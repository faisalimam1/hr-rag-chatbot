"""
FastAPI app for RAG HR Chatbot.

Endpoints:
  - GET /health
  - POST /query { "q": "Your question", "top_k": 5 }

Notes:
  - Requires index built under index/
  - Optional: set OPENAI_API_KEY to use OpenAI chat for answer generation.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import os
import json
from embeddings.embed import get_embedding
from index.faiss_utils import FAISSWrapper
from reranker.rerank import rerank_candidates
from backend.cache import LRUCache

# Optional OpenAI for answer generation
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_KEY:
    try:
        import openai
        openai.api_key = OPENAI_KEY
        _USE_OPENAI = True
    except Exception:
        _USE_OPENAI = False
else:
    _USE_OPENAI = False

app = FastAPI(title="RAG HR Chatbot API")
cache = LRUCache(max_size=1000)

# load index on startup
try:
    faiss_wrapper = FAISSWrapper()
except Exception as e:
    faiss_wrapper = None
    print("Warning: FAISS index not loaded:", e)

class QueryReq(BaseModel):
    q: str
    top_k: int = 5

@app.get("/health")
def health():
    return {"ok": True, "index_size": faiss_wrapper.index_size() if faiss_wrapper else 0}

@app.post("/query")
def query(req: QueryReq):
    if not req.q or not req.q.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    q_text = req.q.strip()
    cached = cache.get(q_text)
    if cached:
        cached["meta"]["cached"] = True
        return cached

    if not faiss_wrapper:
        raise HTTPException(status_code=500, detail="Search index not available. Build it first.")

    t0 = time.time()
    # 1) embed query
    q_emb = get_embedding(q_text)
    # 2) search top 20 candidates
    candidates = faiss_wrapper.search(q_emb, top_k=20)
    if len(candidates) == 0:
        answer = "I could not find relevant policy text in the document."
        payload = {"answer": answer, "sources": [], "score": 0.0, "meta": {"latency_ms": int((time.time()-t0)*1000)}}
        cache.set(q_text, payload)
        return payload

    # 3) rerank to top_k
    reranked = rerank_candidates(q_text, candidates, q_emb, top_k=req.top_k)
    # 4) prepare final prompt context (ids + short snippets)
    context_texts = []
    sources = []
    for i, r in enumerate(reranked, start=1):
        snippet = r["text"]
        context_texts.append(f"[{i}] (id:{r['chunk_id']}) (page:{r['page']})\n{snippet}\n")
        sources.append({"id": r["chunk_id"], "page": r["page"], "text": snippet, "score": r.get("combined_score", 0.0)})

    # 5) generate answer (OpenAI if available else fallback summary)
    answer = None
    if _USE_OPENAI:
        system = "You are an HR assistant. Use only the provided policy excerpts to answer the user's question. If the policy does not contain an answer, say you could not find it and recommend escalation to HR."
        user_prompt = f"CONTEXT:\n\n{''.join(context_texts)}\nQUESTION: {q_text}\n\nINSTRUCTIONS:\n- Answer concisely (2-4 sentences).\n- Cite the excerpt ids you used in parentheses at the end.\n"
        try:
            resp = openai.ChatCompletion.create(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=512
            )
            answer = resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            answer = f"(LLM call failed) {e}"
    else:
        # simple deterministic fallback: combine best snippets + short summary
        top_snippets = "\n\n".join([s["text"] for s in sources[:3]])
        answer = f"Based on the policy excerpts: {top_snippets[:700].strip()}..."
        answer = answer.replace("\n", " ")

    payload = {
        "answer": answer,
        "sources": sources,
        "score": float(sum(s.get("score", 0.0) for s in reranked) / max(1, len(reranked))),
        "meta": {"latency_ms": int((time.time()-t0)*1000), "cached": False}
    }
    cache.set(q_text, payload)
    return payload
