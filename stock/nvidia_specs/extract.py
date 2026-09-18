import re, zlib, sys

def get_streams(data):
    out=[]
    for m in re.finditer(rb'stream\r?\n', data):
        start=m.end()
        e=data.find(b'endstream', start)
        if e==-1: continue
        raw=data[start:e]
        try:
            out.append(zlib.decompress(raw))
        except Exception:
            pass
    return out

def to_unicode_cmaps(data):
    # map from font resource -> dict(hexcode->char) ; simplified: merge all cmaps
    m={}
    for s in get_streams(data):
        if b'beginbfchar' not in s and b'beginbfrange' not in s: continue
        txt=s.decode('latin-1', 'ignore')
        for blk in re.findall(r'beginbfchar(.*?)endbfchar', txt, re.S):
            for a,b in re.findall(r'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>', blk):
                try: m[int(a,16)]=''.join(chr(int(b[i:i+4],16)) for i in range(0,len(b),4))
                except: pass
        for blk in re.findall(r'beginbfrange(.*?)endbfrange', txt, re.S):
            for a,b,c in re.findall(r'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>', blk):
                try:
                    lo,hi,st=int(a,16),int(b,16),int(c,16)
                    for i in range(lo,min(hi,lo+65535)+1): m[i]=chr(st+i-lo)
                except: pass
    return m

def extract(path):
    data=open(path,'rb').read()
    cmap=to_unicode_cmaps(data)
    pieces=[]
    for s in get_streams(data):
        if b'Tj' not in s and b'TJ' not in s: continue
        txt=s.decode('latin-1','ignore')
        for m in re.finditer(r'\[(.*?)\]\s*TJ|\(((?:[^()\\]|\\.)*)\)\s*Tj|<([0-9A-Fa-f\s]+)>\s*Tj', txt, re.S):
            if m.group(1) is not None:
                buf=''
                for sm in re.finditer(r'<([0-9A-Fa-f\s]+)>|\(((?:[^()\\]|\\.)*)\)', m.group(1)):
                    if sm.group(1):
                        h=re.sub(r'\s','',sm.group(1))
                        if len(h)%4==0 and cmap:
                            buf+=''.join(cmap.get(int(h[i:i+4],16),'') for i in range(0,len(h),4))
                        else:
                            buf+=''.join(chr(int(h[i:i+2],16)) for i in range(0,len(h),2) if int(h[i:i+2],16)>31)
                    else:
                        buf+=sm.group(2)
                pieces.append(buf)
            elif m.group(2) is not None:
                pieces.append(m.group(2))
            else:
                h=re.sub(r'\s','',m.group(3))
                if len(h)%4==0 and cmap:
                    pieces.append(''.join(cmap.get(int(h[i:i+4],16),'') for i in range(0,len(h),4)))
                else:
                    pieces.append(''.join(chr(int(h[i:i+2],16)) for i in range(0,len(h),2) if int(h[i:i+2],16)>31))
    return pieces

for p in sys.argv[1:]:
    print('='*30, p)
    print(' | '.join(x for x in extract(p) if x.strip()))
