"""Horizon analysis of the endpoint distribution (graded-vs-fold section).
The infinite-time (attractor) endpoint shows WHERE the stable states lie; the finite-horizon endpoint shows HOW
FAR the treated-cohort distribution has moved toward them at a trial duration. Because of critical slowing, the
bimodality (BC) and variance inflation (VR) BUILD with time: weak at 12 wk, stronger at 24 wk, full at the
attractor. Pure-fold endpoints (no expectation term) at a fixed clinical occupancy u=0.8, so this isolates the
horizon effect and is comparable to the attractor numbers used elsewhere. Population N(24,5.5)."""
import numpy as np, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import build_fit_gcs_slow as M, gcs_bistable_population as B
import gcs_bistable_calibrate as C, gamma_profile_calib as GP

bf=json.load(open('best_fit.json'))
THM=9.15; BB=bf['b']; KD=bf['kappa_des']; HMAX=2.0; PS=GP.PHI_SAT; eC_h=M.eC_h; TAUG=12.0
M.GAMMA_OBS=2.0; M.b_5HT1B=BB; M.setup(KD,HMAX); GP.KAPPA_DES=KD
_uc={}
def eC_ss(u):
    if u in _uc: return _uc[u]
    D50=10.; D=u*D50/(1-u); tg=np.arange(0,80,0.2); v=float(C.ec_series(lambda t:D,D50,tg)[-1]); _uc[u]=v; return v
def cdrug(u): return 1.0+BB*(eC_ss(u)-eC_h)
_GG=np.linspace(1e-4,2.9,3600)
def RHS(G,al,cd): return al*B.Sshape(G,THM,PS)-cd*G
def attractor(al,cd):
    f=al*np.array([B.Sshape(g,THM,PS) for g in _GG])-cd*_GG; st=[]
    for i in np.where(np.diff(np.sign(f))!=0)[0]:
        g=_GG[i]-f[i]*(_GG[i+1]-_GG[i])/(f[i+1]-f[i])
        if (RHS(g+1e-3,al,cd)-RHS(g-1e-3,al,cd))<0: st.append(g)
    hi=[g for g in st if g>0.25]; return max(hi) if hi else (max(st) if st else 1e-4)

U=0.8; cd=cdrug(U); WKS=[12.,24.]
def integ(al,G0,weeks,dt=0.05):
    G=G0; n=int(weeks/dt)
    for _ in range(n):
        G+=dt*RHS(G,al,cd)/TAUG; G=min(max(G,1e-3),3.2)
    return G

# precompute endpoints per Y0 (12wk, 24wk, attractor)
Y0g=np.arange(10.,39.51,0.2); E12=[]; E24=[]; EI=[]
for y in Y0g:
    al,G0,ok=GP.alpha_for(y,THM)
    if not ok: al2,G0,_=GP.alpha_for(24.,THM); al=al2  # fallback (shouldn't trigger in range)
    E12.append(M.Yread(integ(al,G0,12.))); E24.append(M.Yread(integ(al,G0,24.))); EI.append(M.Yread(attractor(al,cd)))
E12=np.array(E12); E24=np.array(E24); EI=np.array(EI)
def mp(y,tab): return np.interp(y,Y0g,tab)

def bc(x):
    x=np.asarray(x,float); n=len(x); s=(x-x.mean())/x.std(); g1=np.mean(s**3); k=np.mean(s**4)-3
    return (g1**2+1)/(k+3*(n-1)**2/((n-2)*(n-3)))
rng=np.random.default_rng(7); N=12000; y0=np.clip(rng.normal(24,5.5,N),10,39.5)
def stats(tab):
    Y=np.clip(mp(y0,tab),0,40)
    return dict(BC=bc(Y),VR=Y.std(ddof=1)/y0.std(ddof=1),remit=100*np.mean(Y<12),mean=Y.mean())
s12,s24,sinf=stats(E12),stats(E24),stats(EI)
print(f"Population N(24,5.5) at u={U}:")
print(f"  horizon   BC     VR    remission%  meanY")
for lab,s in [("12 wk",s12),("24 wk",s24),("attractor",sinf)]:
    print(f"  {lab:>9}  {s['BC']:.2f}  {s['VR']:.2f}   {s['remit']:>5.0f}     {s['mean']:.1f}")

# figure: (a) endpoint histograms at 12/24/inf ; (b) BC & VR vs horizon
fig,(axA,axB)=plt.subplots(1,2,figsize=(12,4.6))
bins=np.linspace(0,40,26)
for tab,lab,c in [(E12,'12 wk','#f4a582'),(E24,'24 wk','#d6604d'),(EI,'attractor ($t\\to\\infty$)','#67001f')]:
    axA.hist(np.clip(mp(y0,tab),0,40),bins=bins,histtype='step',lw=2.2,color=c,label=lab,density=True)
axA.axvline(12,ls=':',color='0.5',lw=1.2); axA.text(12.4,axA.get_ylim()[1]*0.9,'remission ($Y<12$)',fontsize=8,color='0.4')
axA.set_xlabel('endpoint $Y_{\\mathrm{model}}$'); axA.set_ylabel('density'); axA.set_title('(a)  Endpoint distribution builds bimodality with time',fontsize=11)
axA.legend(fontsize=9); axA.grid(alpha=.2)
hz=[12,24,40]; BCs=[s12['BC'],s24['BC'],sinf['BC']]; VRs=[s12['VR'],s24['VR'],sinf['VR']]
axB.plot(hz,BCs,'-o',color='#6a3d9a',lw=2.4,ms=6,label='BC (bimodality)')
axB.axhline(0.555,ls=':',color='#6a3d9a',lw=1.2); axB.text(12.5,0.565,'bimodal threshold 0.55',fontsize=8,color='#6a3d9a')
axB.plot(hz,VRs,'-s',color='#0a7d3f',lw=2.4,ms=6,label='VR (variance ratio)')
axB.axhline(1.0,ls=':',color='#0a7d3f',lw=1.0)
axB.set_xticks(hz); axB.set_xticklabels(['12 wk','24 wk','attractor']); axB.set_xlabel('trial horizon')
axB.set_ylabel('BC / VR'); axB.set_title('(b)  Discriminator strength vs horizon',fontsize=11); axB.grid(alpha=.25); axB.legend(fontsize=9,loc='center right')
plt.tight_layout(); plt.savefig('fig_horizon_bimodality.pdf'); plt.savefig('fig_horizon_bimodality.png',dpi=140)
json.dump(dict(u=U,s12=s12,s24=s24,sinf=sinf),open('horizon_bimodality.json','w'),indent=1)
print("wrote fig_horizon_bimodality.pdf + horizon_bimodality.json")
