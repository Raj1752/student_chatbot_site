import numpy as np
import pdfplumber
import re
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
import onnxruntime as ort


print("Downloading lightweight model...")

model_path = hf_hub_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    filename="onnx/model.onnx"
)

tokenizer_path = hf_hub_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    filename="tokenizer.json"
)

tokenizer = Tokenizer.from_file(tokenizer_path)
tokenizer.enable_truncation(max_length=256)

session = ort.InferenceSession(model_path)


def embed(texts):
    vectors = []

    for text in texts:
        enc = tokenizer.encode(text)

        input_ids = np.array([enc.ids], dtype=np.int64)

        attention_mask = np.array(
            [enc.attention_mask],
            dtype=np.int64
        )

        token_type_ids = np.zeros_like(input_ids)

        outputs = session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids
            }
        )

        token_embeddings = outputs[0][0]
        mask = attention_mask[0][:, None]

        vector = (
            token_embeddings * mask
        ).sum(axis=0) / mask.sum()

        norm = np.linalg.norm(vector)

        if norm != 0:
            vector = vector / norm

        vectors.append(vector)

    return np.array(vectors)


def load_pdf_text(path="university_docs.pdf"):
    text = ""

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


def chunk_text(text):
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


print("Loading handbook and building index...")

raw_text = load_pdf_text()
chunks = chunk_text(raw_text)
chunk_vectors = embed(chunks)

THRESHOLD = 0.45


def get_rag_response(user_input):
    if not user_input.strip():
        return "Please enter a question."

    question = user_input.lower()

    expanded_question = user_input

    illness_words = [
        "miss",
        "missed",
        "sick",
        "ill",
        "illness"
    ]

    if any(word in question for word in illness_words):
        expanded_question += (
            " illness medical certificate "
            "Registrar resit examination"
        )

    question_vector = embed([expanded_question])[0]

    chunk_scores = np.dot(
        chunk_vectors,
        question_vector
    )

    top_indices = np.argsort(
        chunk_scores
    )[-3:][::-1]

    if chunk_scores[top_indices[0]] < THRESHOLD:
        return (
            "I couldn't find that in the university documents. "
            "Please contact the student support office."
        )

    candidate_sentences = []

    for index in top_indices:
        sentences = re.split(
            r"(?<=[.!?])\s+",
            chunks[index]
        )

        for sentence in sentences:
            sentence = sentence.strip()

            if sentence:
                candidate_sentences.append(sentence)

    candidate_text = " ".join(candidate_sentences)

    # Specific Semester 1 answer
    if "semester 1" in question:
        match = re.search(
            r"Semester 1 examinations take place in ([A-Za-z]+)",
            candidate_text,
            re.IGNORECASE
        )

        if match:
            month = match.group(1)

            return (
                "Semester 1 examinations "
                f"take place in {month}."
            )

    # Specific Semester 2 answer
    if "semester 2" in question:
        match = re.search(
            r"Semester 2 examinations take place in ([A-Za-z]+)",
            candidate_text,
            re.IGNORECASE
        )

        if match:
            month = match.group(1)

            return (
                "Semester 2 examinations "
                f"take place in {month}."
            )

    # Missed or sick examination answer
    if any(word in question for word in illness_words):
        match = re.search(
            r"A student who misses an examination due to illness "
            r"must submit a medical certificate to the Registrar's "
            r"Office within three working days and may apply to sit "
            r"the resit examination\.",
            candidate_text,
            re.IGNORECASE
        )

        if match:
            return match.group(0)

    sentence_vectors = embed(candidate_sentences)

    sentence_scores = np.dot(
        sentence_vectors,
        question_vector
    )

    keywords = [
        "semester",
        "december",
        "may",
        "resit",
        "examination",
        "miss",
        "misses",
        "illness",
        "sick",
        "medical",
        "certificate",
        "registrar"
    ]

    for i, sentence in enumerate(candidate_sentences):
        text = sentence.lower()

        for word in keywords:
            if word in text and word in question:
                sentence_scores[i] += 0.10

    best_sentence = int(
        np.argmax(sentence_scores)
    )

    return candidate_sentences[best_sentence]
