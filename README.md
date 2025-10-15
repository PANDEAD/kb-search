# kb-search
# Knowledge Base Search Engine (RAG)

A lightweight document search and question-answering system built with **FastAPI**, **Streamlit**, and **Gemini API**.  
It uses **Retrieval-Augmented Generation (RAG)** — combining local document retrieval with an LLM for concise, context-based answers.

---

## Features
- Upload multiple PDF documents.
- Automatically extracts, chunks, and embeds text for indexing.
- Query the system in natural language.
- Retrieves relevant document snippets and synthesizes answers using the Gemini API.

---

## Tech Stack
- **Backend:** FastAPI  
- **Frontend:** Streamlit  
- **Vector Search:** FAISS  
- **Embeddings:** Sentence Transformers  
- **PDF Parsing:** PyMuPDF  
- **LLM:** Gemini 1.5 Flash API  

---

