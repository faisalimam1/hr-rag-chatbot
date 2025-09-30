#!/usr/bin/env python3
# robust extractor: prefers pdfplumber, falls back to PyPDF2
import argparse, os, json, sys, traceback

def try_pdfplumber(pdf_path):
    try:
        import pdfplumber
    except Exception as e:
        print('pdfplumber not available:', e)
        return None
    try:
        print('Using pdfplumber...')
        pages = []
        with pdfplumber.open(pdf_path) as p:
            for i, page in enumerate(p.pages):
                text = page.extract_text() or ''
                pages.append({'page_number': i+1, 'text': text})
        return pages
    except Exception:
        print('pdfplumber extraction failed:')
        traceback.print_exc()
        return None

def try_pypdf2(pdf_path):
    try:
        from PyPDF2 import PdfReader
    except Exception as e:
        print('PyPDF2 not available:', e)
        return None
    try:
        print('Using PyPDF2 fallback...')
        reader = PdfReader(pdf_path)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ''
            pages.append({'page_number': i+1, 'text': text})
        return pages
    except Exception:
        print('PyPDF2 extraction failed:')
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', required=True, help='input pdf path')
    parser.add_argument('--out', required=True, help='output json path')
    args = parser.parse_args()

    pdf_path = args.pdf
    out_path = args.out
    if not os.path.exists(pdf_path):
        print('ERROR: PDF not found:', pdf_path)
        sys.exit(1)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    pages = try_pdfplumber(pdf_path)
    if pages is None:
        pages = try_pypdf2(pdf_path)

    if pages is None:
        print('ERROR: No extraction backend available. Install pdfplumber or PyPDF2.')
        sys.exit(2)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    print('WROTE:', out_path, 'PAGES:', len(pages))

if __name__ == '__main__':
    main()
