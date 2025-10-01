#!/usr/bin/env python3
"""
Extract text per page from a PDF into JSON.
Usage:
  python ingestion/extract_text.py --pdf data/hr_policy.pdf --out data/extracted_text/hr_policy_pages.json
"""
import pdfplumber
import json
import argparse
import os

def extract_text_per_page(pdf_path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append({
                "page_number": i + 1,
                "text": text
            })
    return pages

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to PDF")
    parser.add_argument("--out", default="data/extracted_text/hr_policy_pages.json", help="Output JSON")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pages = extract_text_per_page(args.pdf)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(pages)} pages -> {args.out}")

if __name__ == "__main__":
    main()
