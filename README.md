# RAG HR Chatbot

This project is a **Retrieval-Augmented Generation (RAG) chatbot** built for answering employee queries based on the HR Policy document.  
It demonstrates how HR teams can deploy an internal AI assistant that retrieves policy information and generates concise, citation-backed answers.

---

## 📌 Features
- **PDF ingestion**: extracts and cleans HR policy text.
- **Chunking**: splits into overlapping chunks for semantic search.
- **Embeddings + FAISS**: vector index for fast retrieval.
- **Re-ranking**: combines BM25 + cosine similarity for accuracy.
- **Backend**: FastAPI `/query` endpoint (answer + sources + latency).
- **Frontend**: Streamlit app with a chat-like interface.
- **Caching**: in-memory LRU cache (optional Redis integration).
- **Dockerized**: single `docker-compose up` runs everything.

---

## 🗂 Project Structure
rag-hr-chatbot/
├─ data/
│ ├─ hr_policy.pdf # HR Policy (input)
│ └─ extracted_text/ # intermediate JSON
├─ ingestion/ # text extraction + cleaning + chunking
├─ embeddings/ # embeddings (OpenAI / sentence-transformers)
├─ index/ # FAISS index build + utilities
├─ reranker/ # BM25 + cosine reranker
├─ backend/ # FastAPI backend + cache
├─ frontend/ # Streamlit UI
├─ Dockerfile
├─ docker-compose.yml
├─ requirements.txt
└─ README.md


---

## ⚡ Quickstart (Local Run)

### 1. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

pip install -r requirements.txt

2. Add HR Policy

Place your policy file at:

data/hr_policy.pdf

3. Extract + Chunk + Build Index
# Extract text
python ingestion/extract_text.py --pdf data/hr_policy.pdf --out data/extracted_text/hr_policy_pages.json

# Chunk into overlapping pieces
python ingestion/chunker.py --pages data/extracted_text/hr_policy_pages.json --out data/extracted_text/chunks.json

# Build FAISS index
python index/build_faiss.py --chunks data/extracted_text/chunks.json --out_dir index

4. Run Backend
uvicorn backend.api:app --reload

Check API at: http://localhost:8000/docs

5. Run Frontend
streamlit run frontend/app.py


Visit: http://localhost:8501

🐳 Quickstart with Docker (Recommended)
docker-compose up --build


Backend → http://localhost:8000

Frontend → http://localhost:8501

🔑 Configuration

OpenAI API (optional but recommended)
Set environment variable before running:
export OPENAI_API_KEY="your_api_key"


Uses text-embedding-3-small for embeddings.

Uses gpt-4o-mini for answer generation.

Offline fallback
If no key is provided, uses sentence-transformers for embeddings and a deterministic summarizer for answers.

🎯 Demo Script (5–10 min)

Start services (docker-compose up --build).

Ask: “How many earned leaves do I get?” → Expect “18 per year” with citation.

Ask: “What is the maternity leave policy?” → Expect “26 weeks if employed >1 year” with citation.

Ask a negative case: “Can I get 1 year of paternity leave?” → Expect graceful fallback.

Repeat query to show cache (faster response).

📊 Tech Stack

Python 3.11

FAISS for vector search

rank_bm25 for lexical rerank

FastAPI backend

Streamlit frontend

Docker + docker-compose

✅ Notes

Keep data/ private (contains HR documents).

API is for internal demo use only — add authentication for production.

Ensure no PII (emails, phone numbers) is embedded.

✨ Recruiter Value

This project shows you can:

Ingest & preprocess unstructured text (PDF policies).

Build a retrieval pipeline with FAISS + reranking.

Integrate LLMs safely for grounded answers.

Deliver a full-stack solution (backend + frontend + Docker).

Balance demo simplicity with production-readiness.


---

Would you like me to also **bundle all the code + this README into a single `.zip` file** so you can just send it directly to your recruiter?
