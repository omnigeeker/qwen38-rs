import re, sys, subprocess, os

POS=[0xf,0x23,0x1d,0x18,0x21,0x10,0x1,0x26,0xa,0x9,0x13,0x1f,0x28,0x1b,0x16,0x17,0x19,0xd,0x6,0xb,
     0x27,0x12,0x14,0x8,0xe,0x15,0x20,0x1a,0x2,0x1e,0x7,0x4,0x11,0x5,0x3,0x1c,0x22,0x25,0xc,0x24]
MASK='3000176000856006061501533003690027800375'
UA='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36'

def solve(arg1):
    out=[None]*len(POS)
    for i in range(len(arg1)):
        for j in range(len(POS)):
            if POS[j]==i+1: out[j]=arg1[i]
    arg2=''.join(x for x in out if x)
    r=''
    for i in range(0,min(len(arg2),len(MASK)),2):
        x=format(int(arg2[i:i+2],16)^int(MASK[i:i+2],16),'02x')
        r+=x
    return r

def fetch(url,out,jar='/tmp/sse_jar.txt'):
    if os.path.exists(jar): os.remove(jar)
    base=['curl','-sL','--compressed','-A',UA,'-c',jar,'-b',jar]
    html=subprocess.run(base+[url],capture_output=True).stdout
    m=re.search(rb"arg1='([0-9A-Fa-f]+)'",html)
    if not m:
        open(out,'wb').write(html); return html[:4]==b'%PDF'
    ck=solve(m.group(1).decode())
    data=subprocess.run(base+['-H','Referer: https://www.sse.com.cn/','--cookie','acw_sc__v2='+ck,url],
                        capture_output=True).stdout
    open(out,'wb').write(data)
    print('saved',len(data),'bytes head',data[:8])
    return data[:4]==b'%PDF'

if __name__=='__main__':
    ok=fetch(sys.argv[1],sys.argv[2])
    print('PDF' if ok else 'BLOCKED/HTML')
