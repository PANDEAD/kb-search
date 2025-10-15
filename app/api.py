from fastapi import FastAPI, UploadFile, Form
import shutil, os
from app.ingestion import build_index
from app.retrieval import retrieve_relevant_chunks
from app.synthesis import synthesize_answer

app = FastAPI()

@app.get("/healthz")
def health_check():
    return {"status": "ok"}

@app.post("/ingest")
async def ingest_files(files: list[UploadFile]):
    os.makedirs("data/pdfs", exist_ok=True)
    for file in files:
        with open(f"data/pdfs/{file.filename}", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    build_index()
    return {"message": "Files uploaded and indexed."}

@app.post("/ask")
async def ask_question(query: str = Form(...)):
    chunks = retrieve_relevant_chunks(query)
    answer = synthesize_answer(query, chunks)
    return {"answer": answer, "sources": [c["doc"] for c in chunks]}
