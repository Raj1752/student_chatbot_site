from sentence_transformers import SentenceTransformer
import numpy as np
import pdfplumber
import json

# Load the pre-trained meaning model (nothing to train)
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---- Read the handbook PDF ----
def load_pdf_text(path="university_docs.pdf"):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# ---- Cut into slices (300 chars, 50 overlap) ----
def chunk_text(text, size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + size]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += size - overlap
    return chunks

# ---- Build index at startup ----
print("Loading handbook and building index...")
raw_text = load_pdf_text()
chunks = chunk_text(raw_text)
chunk_vectors = model.encode(chunks, normalize_embeddings=True)

# ---- FAQ fallback ----
with open("faq.json", "r") as f:
    faq = json.load(f)
faq_questions = list(faq.keys())
faq_answers = list(faq.values())
faq_vectors = model.encode(faq_questions, normalize_embeddings=True)

THRESHOLD = 0.45

# ---- Answer function ----
def get_rag_response(user_input):
    q_vec = model.encode([user_input], normalize_embeddings=True)

    # 1) Search handbook
    scores = np.dot(chunk_vectors, q_vec[0])
    best = int(np.argmax(scores))
    if scores[best] >= THRESHOLD:
        return chunks[best]

    # 2) Fallback: FAQ
    faq_scores = np.dot(faq_vectors, q_vec[0])
    best_faq = int(np.argmax(faq_scores))
    if faq_scores[best_faq] >= THRESHOLD:
        return faq_answers[best_faq]

    # 3) Final fallback
    return "I couldn't find that in the university documents. Please contact the student support office."
