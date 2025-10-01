"""
Rerank candidates using BM25 + cosine similarity fusion.

Public function:
  rerank_candidates(query, candidates, query_embedding, top_k=5, alpha=0.6)

Candidates: list of dicts {'chunk_id','text','embedding','page','score','idx'}
"""
import numpy as np
from rank_bm25 import BM25Okapi

def rerank_candidates(query, candidates, query_embedding, top_k=5, alpha=0.6):
    if len(candidates) == 0:
        return []

    texts = [c["text"] for c in candidates]
    tokenized = [t.split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    qtok = query.split()
    bm25_scores = bm25.get_scores(qtok)
    bm25_scores = np.array(bm25_scores, dtype=float)
    # normalize bm25
    if bm25_scores.max() > 0:
        bm25_scores = bm25_scores / (bm25_scores.max() + 1e-12)
    else:
        bm25_scores = bm25_scores

    embs = np.array([c["embedding"] for c in candidates], dtype=float)
    # normalize
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    qv = np.array(query_embedding, dtype=float)
    qv = qv / (np.linalg.norm(qv) + 1e-12)
    cos = (embs @ qv).astype(float)

    combined = alpha * cos + (1.0 - alpha) * bm25_scores
    order = np.argsort(-combined)[:top_k]
    out = []
    for i in order:
        c = candidates[int(i)].copy()
        c["combined_score"] = float(combined[int(i)])
        c["bm25_score"] = float(bm25_scores[int(i)])
        c["cosine_score"] = float(cos[int(i)])
        out.append(c)
    return out
