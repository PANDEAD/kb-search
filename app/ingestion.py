import os
import json
import fitz  
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

PDF_DIR = "data/pdfs"
INDEX_PATH = "data/indices/faiss.index"
META_PATH = "data/indices/meta.json"

model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def build_index():
    all_texts, metadata = [], []

    for file_name in os.listdir(PDF_DIR):
        if not file_name.endswith(".pdf"):
            continue
        pdf_path = os.path.join(PDF_DIR, file_name)
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            metadata.append({"doc": file_name, "chunk_id": idx, "text": chunk})
            all_texts.append(chunk)

    if not all_texts:
        print("No PDFs found.")
        return

    embeddings = model.encode(all_texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w") as f:
        json.dump(metadata, f)

    print(f"Indexed {len(all_texts)} chunks.")
