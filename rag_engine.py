import numpy as np
import pdfplumber
import json
import re
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
import onnxruntime as ort

# ---- Load lightweight ONNX version of the same model ----
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
    """Turn sentences into meaning-vectors (lightweight way)."""
    vectors = []
    for text in texts:
        enc = tokenizer.encode(text)
        input_ids = np.array([enc.ids], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids)
        outputs = session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        })
        token_embeddings = outputs[0][0]
        mask = attention_mask[0][:, None]
        vec = (token_embeddings * mask).sum(axis=0) / mask.sum()
        vec = vec / np.linalg.norm(vec)
        vectors.append(vec)
    return np.array(vectors)

# ---- Read the handbook PDF ----
def load_pdf_text(path="university_docs.pdf"):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


# ---- Cut into sections ----
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




# ---- Build index at startup ----
print("Loading handbook and building index...")
raw_text = load_pdf_text()
chunks = chunk_text(raw_text)
chunk_vectors = embed(chunks)

# ---- FAQ fallback ----
with open("faq.json", "r") as f:
    faq = json.load(f)
faq_questions = list(faq.keys())
faq_answers = list(faq.values())
faq_vectors = embed(faq_questions)

THRESHOLD = 0.45

# ---- Answer function ----
def get_rag_response(user_input):
    q_vec = embed([user_input])[0]

    scores = np.dot(chunk_vectors, q_vec)
    best = int(np.argmax(scores))
    if scores[best] >= THRESHOLD:
        return chunks[best]

    faq_scores = np.dot(faq_vectors, q_vec)
    best_faq = int(np.argmax(faq_scores))
    if faq_scores[best_faq] >= THRESHOLD:
        return faq_answers[best_faq]

    return "I couldn't find that in the university documents. Please contact the student support office."
