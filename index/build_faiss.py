#!/usr/bin/env python3
"""
Build a FAISS index from chunked JSON + embeddings.

Usage:
  python index/build_faiss.py --chunks data/extracted_text/chunks.json --out_dir index/
Outputs:
  - index/faiss_index.faiss
  - index/embeddings.npy
  - index/meta.json
"""
import argparse
import os
import json
import numpy as np
from embeddings.embed import batch_embed
import faiss

def build_index(chunks, out_dir="index", embedding_batch=32):
    os.makedirs(out_dir, exist_ok=True)
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embs = batch_embed(texts, batch_size=embedding_batch)
    xb = np.array(embs).astype("float32")
    # normalize for IP (cosine)
    faiss.normalize_L2(xb)

    d = xb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(xb)
    index_path = os.path.join(out_dir, "faiss_index.faiss")
    faiss.write_index(index, index_path)
    np.save(os.path.join(out_dir, "embeddings.npy"), xb)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Saved index -> {index_path}, embeddings -> {out_dir}/embeddings.npy, meta -> {out_dir}/meta.json")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True, help="chunks json from ingestion/chunker.py")
    parser.add_argument("--out_dir", default="index", help="output directory")
    args = parser.parse_args()

    with open(args.chunks, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    build_index(chunks, out_dir=args.out_dir)

if __name__ == "__main__":
    main()
