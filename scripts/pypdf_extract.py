from PyPDF2 import PdfReader
import json, os, sys
pdf_path = "data/hr_policy.pdf"
out_dir = "data/extracted_text"
out_path = os.path.join(out_dir, "hr_policy_pages_pypdf.json")
if not os.path.exists(pdf_path):
    print("ERROR: PDF not found:", pdf_path); sys.exit(1)
reader = PdfReader(pdf_path)
pages = []
for i, page in enumerate(reader.pages):
    txt = page.extract_text() or ""
    pages.append({"page_number": i+1, "text": txt})
os.makedirs(out_dir, exist_ok=True)
open(out_path,"w",encoding="utf-8").write(json.dumps(pages,ensure_ascii=False,indent=2))
print("WROTE:", out_path, "PAGES:", len(pages))
