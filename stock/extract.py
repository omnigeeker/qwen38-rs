import sys, os
from pypdf import PdfReader
src, out = sys.argv[1], sys.argv[2]
r = PdfReader(src)
txt = []
for i, p in enumerate(r.pages):
    try:
        t = p.extract_text() or ""
    except Exception as e:
        t = ""
    txt.append(f"\n===PAGE {i+1}===\n" + t)
open(out, "w", encoding="utf-8").write("".join(txt))
print("pages:", len(r.pages), "chars:", sum(len(x) for x in txt))
