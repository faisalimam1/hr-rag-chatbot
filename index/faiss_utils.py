#!/usr/bin/env python3
import os
import json
import numpy as np

try:
    import faiss
    _HAVE_FAISS = True
except Exception:
    faiss = None
    _HAVE_FAISS = False

# Correct clean paths (no escaped quotes)
INDEX_PATH = "index/faiss_index.faiss"
EMB_PATH = "index/embeddings.npy"
META_PATH = "index/meta.json"


class FAISSWrapper:
    def __init__(self, index_path=INDEX_PATH, emb_path=EMB_PATH, meta_path=META_PATH):
        self.index_path = index_path
        self.emb_path = emb_path
        self.meta_path = meta_path

        if not os.path.exists(self.emb_path) or not os.path.exists(self.meta_path):
            raise FileNotFoundError("Embeddings or meta not found. Build the index first.")

        # load numpy embeddings
        self.embeddings = np.load(self.emb_path)  # already L2-normalized
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        if _HAVE_FAISS and os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                self.use_faiss = True
            except Exception:
                self.index = None
                self.use_faiss = False
        else:
            self.index = None
            self.use_faiss = False

    def index_size(self):
        if self.use_faiss and self.index is not None:
            return int(self.index.ntotal)
        return int(len(self.meta))

    def search(self, query_embedding, top_k=20):
        qv = np.array(query_embedding, dtype=np.float32)
        qv = qv / (np.linalg.norm(qv) + 1e-12)

        if self.use_faiss and self.index is not None:
            D, I = self.index.search(qv.reshape(1, -1), top_k)
            D = D[0].tolist()
            I = I[0].tolist()
            out = []
            for idx, score in zip(I, D):
                if idx < 0:
                    continue
                meta = self.meta[idx]
                emb = self.embeddings[idx].tolist()
                out.append({
                    "idx": int(idx),
                    "score": float(score),
                    "chunk_id": meta.get("chunk_id"),
                    "page": meta.get("page"),
                    "text": meta.get("text"),
                    "embedding": emb
                })
            return out
        else:
            # fallback: brute-force cosine search with numpy
            embs = self.embeddings
            cos = (embs @ qv).astype(float)
            idxs = np.argsort(-cos)[:top_k]
            out = []
            for idx in idxs:
                meta = self.meta[int(idx)]
                out.append({
                    "idx": int(idx),
                    "score": float(cos[int(idx)]),
                    "chunk_id": meta.get("chunk_id"),
                    "page": meta.get("page"),
                    "text": meta.get("text"),
                    "embedding": self.embeddings[int(idx)].tolist()
                })
            return out
