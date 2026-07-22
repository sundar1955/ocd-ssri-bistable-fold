import re
txt=open('Paper_A_refs.bib').read()
# main-cited classification: surname present in the (normalized) main body -> goes in main reflist; else SI-only
_main=open('Paper_A_Main_v1.tex').read()
_mbody=_main[:_main.find('\\section*{References}')]
_nb=re.sub(r'\s+',' ',re.sub(r'[~{}\\"]|--',' ',_mbody))
def _base(au):
    t=au.split(',')[0].split()
    if not t: return ''
    if t[0] in ('van','Le','Du','de','El','von'): return t[1] if len(t)>1 else t[0]
    if len(t)>1 and re.fullmatch(r'([A-Z]\.?){1,4}', t[-1]): return t[0]  # "Best JA" -> surname first
    return t[-1]  # "Janet Best" -> surname is last token
def is_main_cited(au):
    b=re.sub(r'[~{}\\"]','',_base(au))
    return bool(b) and re.search(r'\b'+re.escape(b)+r'\b',_nb) is not None
def entries(s):
    i=0; out=[]
    while True:
        j=s.find('@article{',i)
        if j<0: break
        k=j+len('@article{'); depth=1; p=s.find(',',k); key=s[k:p]
        q=p+1
        while q<len(s) and depth>0:
            if s[q]=='{':depth+=1
            elif s[q]=='}':depth-=1
            q+=1
        out.append((key.strip(), s[p+1:q-1])); i=q
    return out
def field(body,name):
    m=re.search(r'\b'+name+r'\s*=\s*\{', body)
    if not m: return ''
    k=m.end(); depth=1
    while k<len(body) and depth>0:
        if body[k]=='{':depth+=1
        elif body[k]=='}':depth-=1
        k+=1
    return re.sub(r'\s+',' ',body[m.end():k-1]).strip()
from collections import Counter
rows=[]
for key,body in entries(txt):
    au=field(body,'author').replace(' and ',', ')
    ti=field(body,'title'); jo=field(body,'journal'); yr=field(body,'year')
    vo=field(body,'volume'); nu=field(body,'number'); pg=field(body,'pages')
    no=field(body,'note'); doi=field(body,'doi')
    sk=(au.split(',')[0].split()[0] if au else key).lower()
    rows.append({'sk':sk,'yr':yr,'au':au,'ti':ti,'jo':jo,'vo':vo,'nu':nu,'pg':pg,'no':no,'doi':doi,'key':key})
# stable final order, then assign a/b/... to same-(surname,year) collisions
rows.sort(key=lambda r:(r['sk'],r['yr'],r['au'],r['ti']))
cnt=Counter((r['sk'],r['yr']) for r in rows); seen={}
for r in rows:
    k=(r['sk'],r['yr'])
    if cnt[k]>1:
        i=seen.get(k,0); seen[k]=i+1
        r['yr']=r['yr']+chr(ord('a')+i)   # 2003 -> 2003a / 2003b
def fmt(r):
    s=f"{r['au']}. {r['ti']}. \\emph{{{r['jo']}}}. {r['yr']}"
    if r['vo']: s+=f";{r['vo']}"
    if r['nu']: s+=f"({r['nu']})"
    if r['pg']: s+=f":{r['pg']}"
    s+="."
    if r['doi']:
        disp=r['doi'].replace('_',r'\_').replace('#',r'\#').replace('%',r'\%').replace('&',r'\&')
        disp=re.sub(r'([/.-])', r'\1\\allowbreak ', disp)   # allow long DOI to line-break
        s+=f" \\href{{https://doi.org/{r['doi']}}}{{doi:{disp}}}."
    if r['no']: s+=f" {r['no']}."
    return s
main_rows=[r for r in rows if is_main_cited(r['au'])]
si_rows  =[r for r in rows if not is_main_cited(r['au'])]
def emit(rws,title):
    return ["\\section*{"+title+"}","{\\footnotesize\\sloppy","\\begin{enumerate}\\setlength{\\itemsep}{1pt}"]+["\\item "+fmt(r) for r in rws]+["\\end{enumerate}}"]
open('Paper_A_References_v2.tex','w').write('\n'.join(emit(main_rows,"References")))
open('Paper_A_References_SI.tex','w').write('\n'.join(emit(si_rows,"Supplementary References")))
print(f"wrote Paper_A_References_v2.tex: {len(main_rows)} main-cited references")
print(f"wrote Paper_A_References_SI.tex: {len(si_rows)} SI-only references")
for (sk,yr),c in cnt.items():
    if c>1: print(f"  collision {sk} {yr}: {c} -> suffixed a..{chr(ord('a')+c-1)}")
