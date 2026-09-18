import re, sys
from pypdf import PdfReader
from pypdf.generic import ContentStream

import os
CMAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "UniCNS-UTF16-H.cmap")

def build_map(path):
    txt = open(path, encoding="latin-1").read()
    m = {}
    for blk in re.findall(r"beginbfchar(.*?)endbfchar", txt, re.S):
        for src, dst in re.findall(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", blk):
            m[int(src, 16)] = "".join(chr(int(dst[i:i+4], 16)) for i in range(0, len(dst), 4))
    for blk in re.findall(r"beginbfrange(.*?)endbfrange", txt, re.S):
        for lo, hi, dst in re.findall(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", blk):
            lo, hi = int(lo, 16), int(hi, 16)
            base = int(dst, 16)
            for i in range(hi - lo + 1):
                m[lo + i] = chr(base + i)
    return m

uni2cid = build_map(CMAP)
# invert: cid -> unicode (first wins)
cid2uni = {}
for u, c in uni2cid.items():
    if c not in cid2uni:
        cid2uni[c] = u

def decode(raw, cmap):
    out = []
    for i in range(0, len(raw) - 1, 2):
        cid = (raw[i] << 8) | raw[i + 1]
        out.append(cid2uni.get(cid, ""))
    return "".join(out)

def main(pdf, out):
    r = PdfReader(pdf)
    with open(out, "w", encoding="utf-8") as f:
        for pno, page in enumerate(r.pages, 1):
            f.write(f"\n<<<PAGE {pno}>>>\n")
            try:
                fonts = page["/Resources"]["/Font"]
            except Exception:
                fonts = {}
            name2cmap = {}
            for k, v in fonts.items():
                try:
                    o = v.get_object()
                    if o.get("/Subtype") == "/Type0":
                        name2cmap[k] = True
                    else:
                        name2cmap[k] = False
                except Exception:
                    name2cmap[k] = False
            cur = None
            try:
                cs = ContentStream(page.get_contents(), r)
            except Exception:
                continue
            for operands, op in cs.operations:
                if op == b"Tf" and len(operands) >= 1:
                    cur = operands[0]
                elif op in (b"Tj", b"'", b'"') and operands:
                    s = operands[-1]
                    if cur is not None and name2cmap.get(cur) and isinstance(s, bytes):
                        f.write(decode(s, cid2uni))
                    elif isinstance(s, bytes):
                        try:
                            f.write(s.decode("latin-1"))
                        except Exception:
                            pass
                elif op == b"TJ" and operands:
                    arr = operands[0]
                    for el in arr:
                        if isinstance(el, bytes):
                            if cur is not None and name2cmap.get(cur):
                                f.write(decode(el, cid2uni))
                            else:
                                f.write(el.decode("latin-1"))
                        elif isinstance(el, (int, float)) and el < -150:
                            f.write(" ")
                if op in (b"Td", b"TD", b"T*", b"ET"):
                    f.write("\n")
            f.write("\n")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
