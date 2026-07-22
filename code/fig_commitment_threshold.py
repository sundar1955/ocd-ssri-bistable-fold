"""Onset / vulnerability figure (3 panels).
(a) The tipping point (unstable saddle = 'commitment threshold' beyond which G_CS drifts spontaneously to the
    OCD attractor) descends as presenting severity rises: severe patients crossed into the runaway regime at a
    Y-BOCS well BELOW the subclinical/clinical boundary.
(b) Treatment runs the same geometry in REVERSE: for a representative mild patient (Y0=20), as SERT occupancy u
    rises the OCD attractor descends and the saddle rises until they collide and annihilate at the saddle-node
    fold -- only health remains (remission).
(c) The onset transient for one patient (Y0=28, drug-free): nudged just past its saddle, the score dwells for
    months near the subclinical/clinical boundary, then escalates within a few months to full OCD."""
import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import build_fit_gcs_slow as M, gamma_profile_calib as GP, gcs_bistable_population as B
C=GP.C; bf=json.load(open('best_fit.json'))
THM=9.15;BB=bf['b'];TG=bf['tau_G'];PS=GP.PHI_SAT;eC_h=M.eC_h
M.GAMMA_OBS=2.0;M.b_5HT1B=BB;M.setup(bf['kappa_des'],2.0);GP.KAPPA_DES=bf['kappa_des']
PURPLE='#6a3d9a';BLUE='#1f3a93'
# --- drug-free machinery (panels a,c) ---
def RHS(G,al): return al*B.Sshape(G,THM,PS)-1.0*G
def fps(Y0):
    al,_,_=GP.alpha_for(Y0,THM); Gs=np.linspace(1e-3,3,8000); f=np.array([RHS(g,al) for g in Gs])
    sad=ocd=None
    for i in np.where(np.diff(np.sign(f))!=0)[0]:
        g=Gs[i]-f[i]*(Gs[i+1]-Gs[i])/(f[i+1]-f[i]); st=(RHS(g+1e-3,al)-RHS(g-1e-3,al))<0
        if (not st) and g>0.02 and sad is None: sad=g
        if st and g>0.25: ocd=g
    return al,sad,ocd
# --- drug (occupancy) machinery (panel b), consistent with Fig 4 ---
def eC_ss(u):
    if u<=1e-6: return eC_h
    D50=10.; D=u*D50/(1-u); tg=np.arange(0,80,0.2); return float(C.ec_series(lambda t:D,D50,tg)[-1])
def RHSd(G,al,ec): return al*B.Sshape(G,THM,PS)-(1.0+BB*(ec-eC_h))*G
def fpsd(al,ec):
    Gs=np.linspace(0.0008,2.7,6000); f=np.array([RHSd(g,al,ec) for g in Gs]); out=[]
    for i in np.where(np.diff(np.sign(f))!=0)[0]:
        g=Gs[i]-f[i]*(Gs[i+1]-Gs[i])/(f[i+1]-f[i]); st=(RHSd(g+1e-3,al,ec)-RHSd(g-1e-3,al,ec))<0
        out.append((g,st))
    return out

fig,(axA,axB,axC)=plt.subplots(1,3,figsize=(15.4,4.8))
# ---- panel (a): commitment threshold vs presenting severity ----
Y0g=np.arange(17,37,0.5); sY=np.array([M.Yread(fps(y)[1]) for y in Y0g])
axA.axhspan(0,8,color='#2ecc71',alpha=0.07)
axA.axhline(8,color='0.5',lw=1.1,ls='--'); axA.text(17.2,8.35,'subclinical / clinical boundary (Y-BOCS $\\approx$8)',fontsize=8,color='0.4',va='bottom')
axA.fill_between(Y0g,sY,Y0g,color='#c0392b',alpha=0.10)
axA.plot(Y0g,Y0g,'-',color=PURPLE,lw=2.0,label='OCD attractor (presenting Y-BOCS)')
axA.plot(Y0g,sY,'-',color=BLUE,lw=2.8,label='commitment threshold (saddle)')
for y in [20,24,28,32]:
    s=M.Yread(fps(y)[1]); axA.plot(y,s,'o',ms=7,mfc='white',mec=BLUE,mew=1.8,zorder=5)
    axA.annotate(f'{s:.1f}',xy=(y,s),xytext=(y-0.2,s-1.4),fontsize=7.5,color=BLUE,ha='center')
axA.text(31,15,'OCD basin:\nspontaneous drift\nto full OCD',fontsize=8.5,color='#8a2718',ha='center',va='center')
axA.text(20.5,3.0,'recovery basin (LTD)',fontsize=8.5,color='#1a6b3a',ha='left',va='center')
axA.set_xlabel('presenting / baseline Y-BOCS$_0$'); axA.set_ylabel('Y-BOCS')
axA.set_xlim(17,36.5); axA.set_ylim(0,37); axA.grid(alpha=.2); axA.legend(fontsize=8.3,loc='upper left')
axA.set_title('(a)  development: the threshold descends with severity',fontsize=10)

# ---- panel (b): treatment reverses it -- branches vs occupancy, colliding at the fold ----
Y0b=20; alb,_,_=GP.alpha_for(Y0b,THM)
uu,Ysad,Yocd=[],[],[]
for u in np.linspace(0,0.95,600):
    pts=fpsd(alb,eC_ss(u))
    sad=[g for g,st in pts if (not st) and g>0.03]
    ocd=[g for g,st in pts if st and g>0.25]
    if sad and ocd:
        uu.append(u); Ysad.append(M.Yread(min(sad))); Yocd.append(M.Yread(max(ocd)))
uu=np.array(uu); Ysad=np.array(Ysad); Yocd=np.array(Yocd)
u_fold=uu[-1]; Y_fold=0.5*(Ysad[-1]+Yocd[-1])
axB.axvspan(u_fold,0.95,color='#2ecc71',alpha=0.08)
axB.fill_between(uu,Ysad,Yocd,color='#c0392b',alpha=0.07)
axB.plot(uu,Yocd,'-',color=PURPLE,lw=2.8,label='OCD attractor (stable)')
axB.plot(uu,Ysad,'--',color=BLUE,lw=2.4,label='commitment threshold (saddle)')
axB.axhline(0,color='k',lw=2.4); axB.plot(0,0,'o',ms=8,mfc='k',mec='k',zorder=6)
axB.text(0.015,0.9,'health (stable)',fontsize=8,color='k',va='bottom')
axB.plot(u_fold,Y_fold,'D',ms=9.5,mfc='#D1495B',mec='k',mew=1.0,zorder=7)
axB.annotate(f'saddle-node fold\n($u^*\\approx{u_fold:.2f}$)',xy=(u_fold,Y_fold),xytext=(u_fold-0.30,Y_fold+8.5),
             fontsize=8,color='#8a2718',ha='center',arrowprops=dict(arrowstyle='-|>',color='#8a2718',lw=1))
axB.text((u_fold+0.95)/2,4.0,'remission\n(only health\nremains)',fontsize=8,color='#1a6b3a',ha='center',va='center')
axB.set_xlabel('SERT occupancy $u$  (treatment)'); axB.set_ylabel('Y-BOCS (fixed point)')
axB.set_xlim(0,0.95); axB.set_ylim(0,25); axB.grid(alpha=.2); axB.legend(fontsize=8.3,loc='upper right')
axB.set_title('(b)  treatment: the threshold rises to meet the attractor',fontsize=10)

# ---- panel (c): onset transient for Y0=28 started just past its saddle ----
al,sad,ocd=fps(28); DT=0.02; G=1.01*sad; t=0.; TT=[0.]; YY=[M.Yread(G)]
while t<95 and G<0.999*ocd:
    G+=DT*(al*B.Sshape(G,THM,PS)-1.0*G)/TG; t+=DT; TT.append(t); YY.append(M.Yread(G))
axC.axhline(M.Yread(sad),color='0.6',lw=0.8,ls=':'); axC.text(2,M.Yread(sad)+0.5,f'saddle (Y$\\approx${M.Yread(sad):.1f})',fontsize=7.5,color='0.45',va='bottom')
axC.axhline(M.Yread(ocd),color='0.6',lw=0.8,ls=':'); axC.text(2,M.Yread(ocd)-0.6,f'OCD attractor (Y$\\approx${M.Yread(ocd):.0f})',fontsize=7.5,color='0.45',va='top')
axC.plot(TT,YY,'-',color='#c0392b',lw=2.6)
axC.text(20,5.5,'metastable\nprodrome',fontsize=8,color='0.4',ha='center')
axC.annotate('rapid escalation\n($\\sim$16 wk)',xy=(58,17),xytext=(74,11),fontsize=8,color='#8a2718',ha='center',
             arrowprops=dict(arrowstyle='-|>',color='#8a2718',lw=1))
axC.set_xlim(0,95); axC.set_ylim(0,31); axC.grid(alpha=.2)
axC.set_xlabel('weeks since crossing the saddle'); axC.set_ylabel('Y-BOCS')
axC.set_title('(c)  onset once past the tipping point (Y-BOCS$_0$=28, drug-free)',fontsize=10)
plt.tight_layout(); plt.savefig('fig_commitment_threshold.pdf'); plt.savefig('fig_commitment_threshold.png',dpi=140)
print(f"(a) saddles ok; (b) fold u*={u_fold:.3f}, Y_fold={Y_fold:.2f}, OCD@u0={Yocd[0]:.1f}, saddle@u0={Ysad[0]:.1f}; (c) saddle Y={M.Yread(sad):.2f}, OCD Y={M.Yread(ocd):.1f}, onset {TT[-1]:.0f} wk")
