#!/usr/bin/env python3
"""
Chunk pages into deterministic character chunks and save chunk metadata.
Usage:
  python ingestion/chunker.py --pages data/extracted_text/hr_policy_pages.json --out data/extracted_text/chunks.json
"""
import json
import argparse
import os
from clean_text import clean_text
from typing import List, Dict
import hashlib

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[Dict]:
    """
    Returns list of dicts: {'chunk_id', 'text', 'start', 'end'}
    """
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + max_chars, L)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({"text": chunk_text, "start": start, "end": end})
        if end == L:
            break
        start = max(0, end - overlap)
    return chunks

def page_to_chunks(page_obj, page_idx, max_chars=1200, overlap=200):
    clean = clean_text(page_obj.get("text", ""))
    chunks = chunk_text(clean, max_chars=max_chars, overlap=overlap)
    out = []
    for i, c in enumerate(chunks):
        # deterministic id
        id_raw = f"p{page_idx}_s{c['start']}_e{c['end']}"
        chunk_id = hashlib.sha1(id_raw.encode("utf-8")).hexdigest()[:10]
        out.append({
            "chunk_id": chunk_id,
            "page": page_idx,
            "start": c["start"],
            "end": c["end"],
            "text": c["text"]
        })
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", required=True, help="Input pages json")
    parser.add_argument("--out", default="data/extracted_text/chunks.json", help="Output chunks json")
    parser.add_argument("--max_chars", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.pages, "r", encoding="utf-8") as f:
        pages = json.load(f)

    chunks = []
    for page in pages:
        page_num = page.get("page_number", 0)
        c = page_to_chunks(page, page_num, max_chars=args.max_chars, overlap=args.overlap)
        chunks.extend(c)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(chunks)} chunks -> {args.out}")

if __name__ == "__main__":
    main()
