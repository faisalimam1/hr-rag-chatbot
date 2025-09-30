from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os, json, traceback

# local modules
try:
    from embeddings.embed import get_embedding
except Exception:
    get_embedding = None

try:
    from index.faiss_utils import FAISSWrapper
except Exception:
    FAISSWrapper = None

app = FastAPI(title="RAG HR Chatbot Backend")

INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "index")
EMB_PATH = os.path.abspath(os.path.join(INDEX_DIR, "embeddings.npy"))
META_PATH = os.path.abspath(os.path.join(INDEX_DIR, "meta.json"))

# Request model
class QueryReq(BaseModel):
    q: str
    top_k: int = 5

_faiss_wrapper = None
def get_wrapper():
    global _faiss_wrapper
    if _faiss_wrapper is None:
        if FAISSWrapper is None:
            raise RuntimeError("Search backend missing. Ensure index/faiss_utils.py exists.")
        _faiss_wrapper = FAISSWrapper()
    return _faiss_wrapper

@app.get("/health")
def health():
    try:
        if not os.path.exists(META_PATH):
            return {"ok": False, "error": "meta.json not found", "index_size": 0}
        meta = json.load(open(META_PATH, "r", encoding="utf-8"))
        return {"ok": True, "index_size": len(meta)}
    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "error": str(e), "index_size": 0}

@app.post("/query")
def query(req: QueryReq):
    if not req.q or not req.q.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    if get_embedding is None:
        raise HTTPException(status_code=500, detail="Embedding backend not available")

    try:
        q_emb = get_embedding(req.q)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {e}")

    try:
        wrapper = get_wrapper()
        hits = wrapper.search(q_emb, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {e}")

    snippets = []
    for h in hits:
        snippets.append(f"(page {h.get('page')}) {h.get('text')[:400]}")
    answer = "\n\n".join(snippets) if snippets else "No relevant policy text found."

    return {
        "query": req.q,
        "answer": answer,
        "sources": hits,
        "meta": {"top_k": req.top_k, "num_found": len(hits)}
    }
