import re

import numpy as np
import onnxruntime as ort
import pdfplumber
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


# ============================================================
# V2.10 - Lightweight RAG Engine
# Main improvements:
# 1. Sentence embeddings are pre-computed once at startup.
# 2. Section headings are removed from final answers.
# 3. Exam-start and semester-start questions are handled safely.
# 4. Existing semester, illness, ID-card and fallback logic is kept.
# ============================================================

FALLBACK_RESPONSE = (
    "I couldn't find that in the university documents. "
    "Please contact the student support office."
)

THRESHOLD = 0.45


print("Downloading lightweight model...")

model_path = hf_hub_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    filename="onnx/model.onnx",
)

tokenizer_path = hf_hub_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    filename="tokenizer.json",
)

tokenizer = Tokenizer.from_file(tokenizer_path)
tokenizer.enable_truncation(max_length=256)

session = ort.InferenceSession(model_path)


def embed(texts):
    """Create normalized MiniLM embeddings for a list of texts."""
    vectors = []

    for text in texts:
        enc = tokenizer.encode(text)

        input_ids = np.array([enc.ids], dtype=np.int64)
        attention_mask = np.array(
            [enc.attention_mask],
            dtype=np.int64,
        )
        token_type_ids = np.zeros_like(input_ids)

        outputs = session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )

        token_embeddings = outputs[0][0]
        mask = attention_mask[0][:, None]

        vector = (token_embeddings * mask).sum(axis=0) / max(mask.sum(), 1)

        norm = np.linalg.norm(vector)

        if norm != 0:
            vector = vector / norm

        vectors.append(vector)

    return np.array(vectors)


def load_pdf_text(path="university_docs.pdf"):
    """Read all available text from the university handbook."""
    text = ""

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


def chunk_text(text):
    """
    Build one chunk for each numbered handbook section.

    Example heading:
    1. EXAMINATIONS
    """
    chunks = []
    current_chunk = []

    for line in text.split("\n"):
        line = line.strip()

        if not line:
            continue

        if re.match(r"^\d+\.\s+[A-Z]", line):
            if current_chunk:
                chunks.append(" ".join(current_chunk))

            current_chunk = [line]
        else:
            current_chunk.append(line)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def remove_heading(text):
    """Remove a numbered or all-capital section heading from an answer."""
    text = text.strip()

    # Example: "3. COURSE REGISTRATION Course registration takes place..."
    text = re.sub(
        r"^\d+\.\s+[A-Z][A-Z0-9/&(),'\- ]{2,}?\s+(?=[A-Z][a-z])",
        "",
        text,
    )

    # Example: "COURSE REGISTRATION Course registration takes place..."
    text = re.sub(
        r"^[A-Z][A-Z0-9/&(),'\- ]{2,}?\s+(?=[A-Z][a-z])",
        "",
        text,
    )

    return re.sub(r"\s+", " ", text).strip()


def split_sentences(chunk):
    """Split a chunk into clean answer sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", chunk)
    cleaned = []

    for sentence in sentences:
        sentence = remove_heading(sentence)

        if sentence:
            cleaned.append(sentence)

    return cleaned


def contains_any(text, words):
    return any(word in text for word in words)


def find_document_sentence(required_words, excluded_words=None):
    """
    Find a clean sentence in the handbook using simple word conditions.
    This is used only for precise questions that should not be guessed.
    """
    excluded_words = excluded_words or []

    for sentence in all_sentences:
        lower_sentence = sentence.lower()

        if (
            all(word in lower_sentence for word in required_words)
            and not any(word in lower_sentence for word in excluded_words)
        ):
            return sentence

    return None


print("Loading handbook and building index...")

raw_text = load_pdf_text()
chunks = chunk_text(raw_text)

# Chunk embeddings are used for top-3 retrieval.
chunk_vectors = embed(chunks)

# Pre-compute every sentence and its embedding once at startup.
sentences_by_chunk = []
all_sentences = []
sentence_locations = []

for chunk_index, chunk in enumerate(chunks):
    chunk_sentences = split_sentences(chunk)
    sentences_by_chunk.append(chunk_sentences)

    for sentence in chunk_sentences:
        all_sentences.append(sentence)
        sentence_locations.append(chunk_index)

if all_sentences:
    all_sentence_vectors = embed(all_sentences)
else:
    all_sentence_vectors = np.empty((0, chunk_vectors.shape[1]))

sentence_locations = np.array(sentence_locations, dtype=np.int64)


def get_rag_response(user_input):
    """Return a grounded answer from the university handbook."""
    if not user_input or not user_input.strip():
        return "Please enter a question."

    original_question = user_input.strip()
    question = original_question.lower()
    expanded_question = original_question

    exam_context = contains_any(question, ["exam", "examination"])
    illness_query = (
        contains_any(question, ["sick", "ill", "illness"])
        or (
            contains_any(question, ["miss", "missed"])
            and exam_context
        )
    )
    student_id_query = (
        contains_any(
            question,
            [
                "student id",
                "id card",
                "identification",
                "without my id",
            ],
        )
        or ("bring" in question and exam_context)
    )

    # --------------------------------------------------------
    # Precise question: When do exams start?
    # Return the two examination months stated in the handbook.
    # --------------------------------------------------------
    exam_start_query = (
        exam_context
        and contains_any(question, ["start", "begin", "held", "take place"])
        and "timetable" not in question
        and "schedule" not in question
    )

    if exam_start_query:
        semester_1 = re.search(
            r"Semester\s*1\s+(?:exams?|examinations?)\s+"
            r"(?:take\s+place|are\s+held)\s+in\s+([A-Za-z]+)",
            raw_text,
            re.IGNORECASE,
        )
        semester_2 = re.search(
            r"Semester\s*2\s+(?:exams?|examinations?)\s+"
            r"(?:take\s+place|are\s+held)\s+in\s+([A-Za-z]+)",
            raw_text,
            re.IGNORECASE,
        )

        if semester_1 and semester_2:
            return (
                f"Semester 1 examinations take place in "
                f"{semester_1.group(1)}, and Semester 2 examinations "
                f"take place in {semester_2.group(1)}."
            )

        return FALLBACK_RESPONSE

    # --------------------------------------------------------
    # Precise question: When does the semester start?
    # The handbook does not provide the actual semester start date.
    # A timetable publication date is not a semester start date.
    # --------------------------------------------------------
    if (
        "semester" in question
        and contains_any(question, ["start", "begin"])
        and not exam_context
    ):
        return FALLBACK_RESPONSE

    # Query expansion for known paraphrases.
    if illness_query:
        expanded_question += (
            " illness medical certificate Registrar "
            "resit examination"
        )

    if student_id_query:
        expanded_question += (
            " students must bring student ID card "
            "to every examination"
        )

    question_vector = embed([expanded_question])[0]

    chunk_scores = np.dot(
        chunk_vectors,
        question_vector,
    )

    top_indices = np.argsort(chunk_scores)[-3:][::-1]

    if chunk_scores[top_indices[0]] < THRESHOLD:
        return FALLBACK_RESPONSE

    # Find pre-computed sentences belonging to the top-3 chunks.
    candidate_mask = np.isin(sentence_locations, top_indices)
    candidate_indices = np.where(candidate_mask)[0]

    if len(candidate_indices) == 0:
        return FALLBACK_RESPONSE

    candidate_sentences = [
        all_sentences[index]
        for index in candidate_indices
    ]
    candidate_vectors = all_sentence_vectors[candidate_indices]
    candidate_text = " ".join(candidate_sentences)

    # --------------------------------------------------------
    # Specific Semester 1 examination answer
    # --------------------------------------------------------
    if (
        "semester 1" in question
        and exam_context
    ):
        match = re.search(
            r"Semester 1 examinations take place in ([A-Za-z]+)",
            candidate_text,
            re.IGNORECASE,
        )

        if match:
            return (
                "Semester 1 examinations "
                f"take place in {match.group(1)}."
            )

    # --------------------------------------------------------
    # Specific Semester 2 examination answer
    # --------------------------------------------------------
    if (
        "semester 2" in question
        and exam_context
    ):
        match = re.search(
            r"Semester 2 examinations take place in ([A-Za-z]+)",
            candidate_text,
            re.IGNORECASE,
        )

        if match:
            return (
                "Semester 2 examinations "
                f"take place in {match.group(1)}."
            )

    # --------------------------------------------------------
    # Missed or sick examination answer
    # --------------------------------------------------------
    if illness_query:
        match = re.search(
            r"A student who misses an examination due to illness "
            r"must submit a medical certificate to the Registrar's "
            r"Office within three working days and may apply to sit "
            r"the resit examination\.",
            candidate_text,
            re.IGNORECASE,
        )

        if match:
            return remove_heading(match.group(0))

    # --------------------------------------------------------
    # Student ID examination answer
    # --------------------------------------------------------
    if student_id_query:
        match = re.search(
            r"Students must bring their student ID card "
            r"to every examination\.",
            candidate_text,
            re.IGNORECASE,
        )

        if match:
            return remove_heading(match.group(0))

    # Sentence reranking now uses pre-computed sentence vectors.
    sentence_scores = np.dot(
        candidate_vectors,
        question_vector,
    )

    # Generic keyword overlap boost.
    query_words = {
        word
        for word in re.findall(r"[a-z0-9']+", question)
        if len(word) > 2
    }

    for index, sentence in enumerate(candidate_sentences):
        sentence_words = set(
            re.findall(r"[a-z0-9']+", sentence.lower())
        )

        overlap = len(query_words.intersection(sentence_words))
        sentence_scores[index] += overlap * 0.05

    best_sentence = int(np.argmax(sentence_scores))
    answer = remove_heading(candidate_sentences[best_sentence])

    return answer or FALLBACK_RESPONSE
