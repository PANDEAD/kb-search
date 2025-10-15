import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = "data/indices/faiss.index"
META_PATH = "data/indices/meta.json"

model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_relevant_chunks(query, top_k=5):
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r") as f:
        metadata = json.load(f)

    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), top_k)

    results = [metadata[i] for i in I[0]]
    return results
