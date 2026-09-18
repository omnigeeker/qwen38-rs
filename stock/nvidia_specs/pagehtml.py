import re,sys,urllib.request,ssl,html
ctx=ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
u=sys.argv[1]
r=urllib.request.Request(u, headers={'User-Agent':'Mozilla/5.0'})
h=urllib.request.urlopen(r, timeout=40, context=ctx).read().decode('utf-8','ignore')
open('page.html','w').write(h)
# strip scripts/styles
b=re.sub(r'(?is)<(script|style|noscript)[^>]*>.*?</\1>',' ',h)
b=re.sub(r'(?is)<br[^>]*>|</(tr|p|div|li|h1|h2|h3|h4|table)>','\n',b)
b=re.sub(r'(?is)</t[dh]>',' | ',b)
t=re.sub(r'(?s)<[^>]+>',' ',b)
t=html.unescape(t)
t=re.sub(r'[ \t]+',' ',t)
lines=[l.strip() for l in t.split('\n')]
lines=[l for l in lines if l and len(l)>1]
print('\n'.join(lines))
