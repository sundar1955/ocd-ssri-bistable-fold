"""'Role of alpha' figure (2 panels), drug-free.
(a) RHS of the plasticity equation vs G_CS for several alpha: below threshold (health only),
    at onset (fold birth), mild, severe -- saddle moves left / attractor moves right as alpha grows.
(b) Saddle-node bifurcation in alpha: health (Y=0) always stable; a saddle + OCD attractor are born
    at alpha_c (~Y12) and separate as alpha increases. Attractor Y = presenting severity (calibration target)."""
import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import build_fit_gcs_slow as M, gamma_profile_calib as GP, gcs_bistable_population as B
bf=json.load(open('best_fit.json')); THM=9.15; PS=GP.PHI_SAT; TG=bf['tau_G']
M.GAMMA_OBS=2.0; M.b_5HT1B=bf['b']; M.setup(bf['kappa_des'],2.0); GP.KAPPA_DES=bf['kappa_des']
GREEN='#2a8f5a'; BLUE='#1f3a93'; ORANGE='#E69F00'; RED='#D1495B'; PURPLE='#6a3d9a'; GREY='0.55'
def RHS(G,al): return al*B.Sshape(G,THM,PS)-G
def posfps(al):
    Gs=np.linspace(1e-3,2.9,9000); f=np.array([RHS(g,al) for g in Gs]); out=[]
    for i in np.where(np.diff(np.sign(f))!=0)[0]:
        g=Gs[i]-f[i]*(Gs[i+1]-Gs[i])/(f[i+1]-f[i]); st=RHS(g+1e-3,al)-RHS(g-1e-3,al)<0
        if g>0.02: out.append((g,st))
    return out
# --- locate fold-birth alpha_c by bisection ---
lo,hi=3.30e-6,3.60e-6
for _ in range(50):
    mid=0.5*(lo+hi)
    if len(posfps(mid))>=2: hi=mid
    else: lo=mid
ac=hi; print(f"alpha_c = {ac:.4e}  (birth FPs Y={[round(M.Yread(g),1) for g,s in posfps(ac)]})")

fig,(axA,axB,axC)=plt.subplots(1,3,figsize=(13.3,3.77))
# ---------- (a) phase lines vs alpha ----------
Gg=np.linspace(0,2.25,700)
aL=GP.alpha_for(20,THM)[0]; aH=GP.alpha_for(30,THM)[0]
fam=[(0.965*ac,'below threshold ($\\alpha<\\alpha_c$)',GREY,'-'),
     (1.02*ac ,'at onset ($\\alpha=\\alpha_c$)',ORANGE,'-'),
     (aL,'mild ($Y^\\ast\\!=\\!20$)',BLUE,'-'),
     (aH,'severe ($Y^\\ast\\!=\\!30$)',RED,'-')]
axA.axhline(0,color='0.6',lw=0.9,ls=':'); hh=[]; yall=[]
for al,lab,c,ls in fam:
    yv=[RHS(g,al) for g in Gg]; yall+=yv
    axA.plot(Gg,yv,color=c,lw=2.3,ls=ls,zorder=3); hh.append(Line2D([],[],color=c,lw=2.3,label=lab))
    for g,st in posfps(al):
        if g<0.04: continue
        axA.plot(g,0,'o',ms=8.5,mfc=(c if st else 'white'),mec='k',mew=(1.0 if st else 1.6),zorder=6)
axA.plot(0,0,'o',ms=8.5,mfc='k',mec='k',zorder=7)
ylo,yhi=1.12*min(yall),1.12*max(yall)
axA.annotate('',xy=(0.28,0.5*ylo),xytext=(0.62,0.5*ylo),arrowprops=dict(arrowstyle='-|>',color=BLUE,lw=1.3))
axA.text(0.45,0.64*ylo,'saddle $\\leftarrow$ ($\\alpha\\uparrow$)',fontsize=8,color=BLUE,ha='center',va='top')
axA.annotate('',xy=(1.95,0.45*yhi),xytext=(1.55,0.45*yhi),arrowprops=dict(arrowstyle='-|>',color=PURPLE,lw=1.3))
axA.text(1.72,0.58*yhi,'attractor $\\rightarrow$ ($\\alpha\\uparrow$)',fontsize=8,color=PURPLE,ha='center',va='bottom')
axA.set_xlim(-0.03,2.25); axA.set_ylim(ylo,yhi)
axA.set_xlabel('$G_{CS}$  (corticostriatal weight)'); axA.set_ylabel('plasticity rate  $\\tau_G\\,dG_{CS}/dt$')
axA.legend(handles=hh,fontsize=8.0,loc='upper left',framealpha=0.96); axA.grid(alpha=.18)
axA.set_title('(a)  the landscape as vulnerability $\\alpha$ grows',fontsize=10.5)
# ---------- (b) bifurcation in alpha ----------
alr=np.linspace(0.90*ac,GP.alpha_for(36,THM)[0],400); rr=alr/ac
Ysad=[]; Yatt=[]
for al in alr:
    fp=posfps(al); s=[g for g,st in fp if not st and g>0.02]; a=[g for g,st in fp if st and g>0.25]
    Ysad.append(M.Yread(max(s)) if s else np.nan); Yatt.append(M.Yread(max(a)) if a else np.nan)
axB.axvspan(rr[0],1.0,color=GREEN,alpha=0.06); axB.axvspan(1.0,rr[-1],color=RED,alpha=0.05)
axB.axhline(0,color=GREEN,lw=2.6,label='health (stable)')
axB.plot(rr,Yatt,color=PURPLE,lw=2.6,label='OCD attractor (stable) = presenting $Y$')
axB.plot(rr,Ysad,color=BLUE,lw=2.4,ls='--',label='commitment threshold (saddle)')
axB.axhline(8,color='0.5',lw=1.0,ls=':'); axB.text(rr[-1],8.4,'subclinical/clinical ($Y\\approx8$)',fontsize=7.6,color='0.4',ha='right',va='bottom')
axB.plot(1.0,M.Yread(max(g for g,s in posfps(ac))),'*',ms=15,mfc='k',mec='k',zorder=6)
axB.annotate('onset threshold $\\alpha_c$\n(fold: saddle+attractor born)',xy=(1.0,12),xytext=(1.28,4.5),
             fontsize=8,ha='left',arrowprops=dict(arrowstyle='-|>',color='k',lw=1.0))
axB.set_xlabel('vulnerability  $\\alpha/\\alpha_c$'); axB.set_ylabel('$Y_{\\mathrm{model}}$')
axB.set_xlim(rr[0],rr[-1]); axB.set_ylim(-1.5,37); axB.legend(fontsize=8.2,loc='center right',framealpha=0.96); axB.grid(alpha=.18)
axB.set_title('(b)  onset threshold and the two branches',fontsize=10.5)
# ---------- (c) onset transient (once past the tipping point, drug-free) ----------
al28=GP.alpha_for(28,THM)[0]; fp28=posfps(al28)
sad=min(g for g,st in fp28 if not st and g>0.02); ocd=max(g for g,st in fp28 if st and g>0.25)
DT=0.02; G=1.01*sad; t=0.; TT=[0.]; YY=[M.Yread(G)]
while t<95 and G<0.999*ocd:
    G+=DT*(al28*B.Sshape(G,THM,PS)-G)/TG; t+=DT; TT.append(t); YY.append(M.Yread(G))
tau=np.array(TT)/TG
axC.axhline(M.Yread(sad),color='0.6',lw=0.8,ls=':'); axC.text(0.12,M.Yread(sad)+0.6,f'saddle (Y$\\approx${M.Yread(sad):.1f})',fontsize=8,color='0.45',va='bottom')
axC.axhline(M.Yread(ocd),color='0.6',lw=0.8,ls=':'); axC.text(0.12,M.Yread(ocd)-0.6,f'OCD attractor (Y$\\approx${M.Yread(ocd):.0f})',fontsize=8,color='0.45',va='top')
axC.plot(tau,YY,'-',color=RED,lw=2.6)
axC.text(20/TG,5.5,'metastable\nprodrome',fontsize=8.5,color='0.4',ha='center')
axC.annotate('rapid\nescalation',xy=(58/TG,17),xytext=(74/TG,11),fontsize=8.5,color='#8a2718',ha='center',arrowprops=dict(arrowstyle='-|>',color='#8a2718',lw=1))
axC.set_xlim(0,95/TG); axC.set_ylim(0,31); axC.grid(alpha=.18)
axC.set_xlabel('time since crossing  (units of $\\tau_G$)'); axC.set_ylabel('$Y_{\\mathrm{model}}$')
axC.set_title('(c)  onset once past the tipping point',fontsize=10.5)
plt.tight_layout(); plt.savefig('fig_alpha_role.pdf'); plt.savefig('fig_alpha_role.png',dpi=140)
print("wrote fig_alpha_role.pdf/png")
