import re,subprocess,sys,os,time
UA='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36'
def solve(url):
    h=subprocess.run(['curl','-s','--max-time','40','--compressed','-A',UA,url],capture_output=True).stdout
    open('/tmp/chal.html','wb').write(h)
    if h[:4]==b'%PDF': return None
    out=subprocess.run(['node','sse_solve.js','/tmp/chal.html'],capture_output=True).stdout.decode()
    m=re.search(r'COOKIE::([^;]+)',out)
    return m.group(1) if m else None
def get(url,out,timeout='900'):
    cookie=solve(url)
    print('cookie',cookie,flush=True)
    for attempt in range(3):
        cmd=['curl','-s','--max-time',timeout,'--compressed','-A',UA,'-H','Referer: https://www.sse.com.cn/']
        if cookie: cmd+=['-H','Cookie: '+cookie]
        cmd+=['-o',out,url]
        subprocess.run(cmd)
        try:
            if open(out,'rb').read(4)==b'%PDF':
                print('OK bytes',os.path.getsize(out),flush=True); return True
        except Exception: pass
        print('retry',attempt,flush=True); time.sleep(3); cookie=solve(url)
    return False
if __name__=='__main__':
    print('RESULT',get(sys.argv[1],sys.argv[2]))
