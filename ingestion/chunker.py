#!/usr/bin/env python3
# Robust chunker:
# - Accepts input JSON with either "page" or "page_number" keys.
# - Falls back to basic text cleaner if clean_text module missing.
# - Splits text into overlapping chunks for embedding.
# - Writes output JSON to --out.

import json
import argparse
import os
import hashlib
import re

# Try to import the project's clean_text; fallback to local simple cleaner
try:
    from clean_text import clean_text as _clean_text
except Exception:
    def _clean_text(s: str) -> str:
        if not s:
            return ""
        s = re.sub(r"\r\n", "\n", s)
        s = re.sub(r"\n{2,}", "\n", s)
        s = re.sub(r"\s+", " ", s)
        return s.strip()

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200):
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + max_chars, L)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((start, end, chunk))
        if end == L:
            break
        start = max(0, end - overlap)
    return chunks

def page_to_chunks(page_obj, page_idx, max_chars=1200, overlap=200):
    raw_text = page_obj.get("text") or page_obj.get("content") or ""
    clean = _clean_text(raw_text)
    chunks = chunk_text(clean, max_chars=max_chars, overlap=overlap)
    out = []
    for i, (start, end, txt) in enumerate(chunks):
        id_raw = f"p{page_idx}_s{start}_e{end}"
        chunk_id = hashlib.sha1(id_raw.encode("utf-8")).hexdigest()[:12]
        out.append({
            "chunk_id": chunk_id,
            "page": int(page_idx),
            "start": int(start),
            "end": int(end),
            "text": txt
        })
    return out

def load_pages(pages_path):
    with open(pages_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "pages" in data:
        data = data["pages"]
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", required=True, help="Input pages json")
    parser.add_argument("--out", required=True, help="Output chunks json")
    parser.add_argument("--max_chars", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=200)
    args = parser.parse_args()

    pages = load_pages(args.pages)
    if not isinstance(pages, list):
        raise SystemExit("Input pages file must contain a JSON list of pages.")

    all_chunks = []
    for p in pages:
        page_idx = p.get("page") or p.get("page_number") or p.get("pageIndex") or 0
        c = page_to_chunks(p, page_idx, max_chars=args.max_chars, overlap=args.overlap)
        all_chunks.extend(c)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(all_chunks)} chunks -> {args.out}")

if __name__ == "__main__":
    main()
