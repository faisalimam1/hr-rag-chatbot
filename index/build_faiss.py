#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np

# Try to import faiss; if not available we'll use numpy fallback
try:
    import faiss
    _HAVE_FAISS = True
except Exception:
    faiss = None
    _HAVE_FAISS = False

from embeddings.embed import batch_embed

def build_faiss_index(chunks, out_dir="index", embedding_batch=32):
    os.makedirs(out_dir, exist_ok=True)
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embs = batch_embed(texts, batch_size=embedding_batch)
    xb = np.array(embs).astype("float32")
    # normalize for cosine via inner product
    norms = np.linalg.norm(xb, axis=1, keepdims=True) + 1e-12
    xb = xb / norms

    # Save embeddings and meta
    np.save(os.path.join(out_dir, "embeddings.npy"), xb)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    if _HAVE_FAISS:
        d = xb.shape[1]
        print("Building FAISS Index (IndexFlatIP)...")
        index = faiss.IndexFlatIP(d)
        index.add(xb)
        faiss.write_index(index, os.path.join(out_dir, "faiss_index.faiss"))
        print("Saved FAISS index ->", os.path.join(out_dir, "faiss_index.faiss"))
    else:
        print("FAISS not available: saved embeddings.npy and meta.json (numpy fallback).")

    print("Saved embeddings ->", os.path.join(out_dir, "embeddings.npy"))
    print("Saved meta ->", os.path.join(out_dir, "meta.json"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True, help="chunks json from ingestion/chunker.py")
    parser.add_argument("--out_dir", default="index", help="output directory")
    parser.add_argument("--embedding_batch", type=int, default=32)
    args = parser.parse_args()

    with open(args.chunks, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    build_faiss_index(chunks, out_dir=args.out_dir, embedding_batch=args.embedding_batch)

if __name__ == "__main__":
    main()
