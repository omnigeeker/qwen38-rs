import re,sys,urllib.request,ssl
ctx=ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
def get(u):
    try:
        r=urllib.request.Request(u, headers={'User-Agent':'Mozilla/5.0'})
        return urllib.request.urlopen(r, timeout=30, context=ctx).read().decode('utf-8','ignore')
    except Exception as e:
        return ''
for page in sys.argv[1:]:
    h=get(page)
    print('###', page, len(h))
    for m in set(re.findall(r'https?://[^"\'\s<>]+?\.pdf', h)):
        if 'nvidia' in m: print('  ', m)
    for m in set(re.findall(r'https?://resources\.nvidia\.com/[^"\'\s<>]+', h)):
        print('  R', m)
