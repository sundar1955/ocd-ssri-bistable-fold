"""
Full-PANEL quantitative fit of the MECHANISTIC PK->circuit->Y-BOCS model.
Breaks the gamma_C / gamma_obs / h_max0 identifiability degeneracy seen in the 2-drug
(FLX+CIT) fit by adding 5 more drugs whose potencies (D50) are PINNED at literature SERT
values -- i.e. new data, NO new freedom on the shared gain chain.

7 drugs / 9 studies:
  FLX  Tollefson 1994   absolute Y-BOCS, 20/40/60 mg           (baseline 23.95)
  CIT  Montgomery       change-from-baseline, 20/40/60 mg      (baseline 23.95)
  ESC  Stein 2007       net-of-placebo, 10 & 20 mg             (baseline 27.0)
  PAR  Stein 2007       net-of-placebo, 40 mg                  (baseline 27.0)
  PAR  Zohar            net-of-placebo, 37.5 mg                (baseline 25.5)
  CMI  Greist           net-of-placebo, 226 mg  <- FALSIFY     (baseline 26.2)
  CMI  Zohar            net-of-placebo, 113.1 mg <- FALSIFY    (baseline 25.5)
  FLV  Askari           net-of-placebo, 100->200 mg titrated   (baseline 28.4)
  SRT  Ghob(irnia)      net-of-placebo, 100->200 mg titrated   (baseline 27.5)

Shared FREE (5): kappa_des, h_max0, tau_desens, gamma_C, gamma_obs.
FREE D50 (2):    D50_flx, D50_cit  (these two have dose-response data).
PINNED D50:      ESC=5, PAR=5, CMI=50, FLV=50, SRT=35 mg  (literature half-SERT-occupancy dose).

Chain identical to build_fit_ssri.py:
  dose(mg) -> SERT block k_SERT -> eC (QSS, autoreceptor theta desensitizes over tau) ->
  vAR fast circuit (eC->cortex via gamma_C) -> caudate phi_d1 -> Y=40*p^gamma_obs.
Each study gets its own G_CS solved so the UNTREATED Y-BOCS = that study's baseline
(same neurobiology, different population severity). Net-of-placebo removes the NS drift.
"""
import numpy as np, time, sys
from scipy.optimize import brentq, differential_evolution, fsolve
from scipy.interpolate import RectBivariateSpline
from var_circuit_validation import idx, S, M, ext, Qmax_v

ie=idx['e']; id1=idx['d1']; id2=idx['d2']; RHO=0.7
kSERT0=40.; kOCT3=8.; K=10.; betaSERT=0.85; w=0.02; eC_h=5.0
phid1_h=7.39; Qd1=float(Qmax_v[id1])

# ---- circuit grid (phi_e, phi_d1 over G_CS and cortical shift s) ----
def M_g(g):
    Mm=M.copy(); Mm[id1,ie]+=g; Mm[id2,ie]+=RHO*g; return Mm
def circuit(G,s,seed):
    Mm=M_g(G)
    def res(phi):
        V=(Mm@phi+ext).copy(); V[ie]+=s; return phi-S(V)
    return fsolve(res,seed,xtol=1e-10)
print("precomputing circuit grid...", flush=True)
Gg=np.linspace(0.0,3.5,36); sg=np.linspace(-4.0,1.5,40)
PHE=np.zeros((36,40)); PHD1=np.zeros((36,40))
for i,G in enumerate(Gg):
    seed=np.array([12,12,40,20,60,30,30,15,28.])
    for j,s in enumerate(sg):
        seed=circuit(G,s,seed); PHE[i,j]=seed[ie]; PHD1[i,j]=seed[id1]
phe_i=RectBivariateSpline(Gg,sg,PHE); phd1_i=RectBivariateSpline(Gg,sg,PHD1)
def phe(G,s):  return float(phe_i(np.clip(G,Gg[0],Gg[-1]),np.clip(s,sg[0],sg[-1]))[0,0])
def phd1(G,s): return float(phd1_i(np.clip(G,Gg[0],Gg[-1]),np.clip(s,sg[0],sg[-1]))[0,0])
print("grid done.", flush=True)

# ---- mechanism ----
F0=None; hmax0=None; theta_h=None
def kSERT(mg,D50): return kSERT0*(1-betaSERT*mg/(mg+D50))
def eC_qss(theta,mg,D50,G,gC):
    def bal(e): return F0*(1+w*phe(G,gC*(e-eC_h))) - (kSERT(mg,D50)+kOCT3)*e - theta*hmax0*e/(K+e)
    return brentq(bal,0.3,300.,xtol=1e-7)
def ybocs(pd1,gamma):
    p=max(0.0,(pd1-phid1_h)/(Qd1-phid1_h)); return 40*p**gamma
def setup(kappa_des,hm0):
    global F0,hmax0,theta_h
    hmax0=hm0; theta_h=1/(1+kappa_des*eC_h)
    F0=(kSERT(0,1.)+kOCT3)*eC_h/(1+w*phe(0.0,0.0)) + theta_h*hmax0*eC_h/(K+eC_h)/(1+w*phe(0.0,0.0))
    return theta_h
def baseline_G(kappa_des,gC,gamma,Ytarget):
    def Yb(G):
        th=theta_h
        for _ in range(60): e=eC_qss(th,0.0,10.,G,gC); th=1/(1+kappa_des*e)
        return ybocs(phd1(G,gC*(e-eC_h)),gamma), e
    try: G=brentq(lambda g: Yb(g)[0]-Ytarget, 0.2, 3.4, xtol=1e-4)
    except ValueError: return None,None
    return G, Yb(G)[1]
def trajectory(dose_fn,D50,G,gC,gamma,kappa_des,tau_d,weeks,eC0):
    """absolute Y-BOCS at requested weeks (no NS drift). dose_fn(t)->mg."""
    dt=0.1; th=theta_h; e=eC0; wk_set=set(np.round(weeks,3)); traj={}
    for t in np.arange(0,weeks[-1]+dt/2,dt):
        e=eC_qss(th,dose_fn(t),D50,G,gC); th+=dt*((1-th)-kappa_des*e*th)/tau_d
        rt=round(t,3)
        if rt in wk_set: traj[rt]=ybocs(phd1(G,gC*(e-eC_h)),gamma)
    return np.array([traj[round(wk,3)] for wk in weeks])
def const(mg):      return lambda t: mg
def titr(d1,d2,tsw): return lambda t: d1 if t<tsw else d2

# ============================ DATA ============================
# FLX Tollefson 1994 (absolute Y-BOCS), CIT Montgomery (change-from-baseline)
WK_T=np.array([0,1,3,5,7,9,11,13.])
TOL={20:np.array([23.6,22.3,21.1,20.2,19.8,19.2,18.9,18.9]),
     40:np.array([23.5,22.7,21.0,19.5,18.1,17.8,18.4,18.1]),
     60:np.array([24.4,22.7,21.4,19.5,17.7,16.9,16.4,16.8])}
TOLn={20:np.array([87,86,84,80,76,77,75,75.]),40:np.array([89,88,84,79,74,68,67,67.]),60:np.array([90,89,85,81,77,74,72,68.])}
TOLsd={20:np.array([5.7,5.8,6.3,6.7,7.0,7.8,8.1,8.3]),40:np.array([5.6,6.1,7.1,7.2,7.0,7.7,7.5,7.9]),60:np.array([5.1,5.6,6.3,6.7,7.1,7.1,7.4,7.8])}
WK_M=np.array([0,1,3,5,7,9,12.])
MONc={20:np.array([0,0.7,1.2,2.1,2.6,2.7,2.8]),40:np.array([0,1.0,1.7,2.8,3.3,3.3,3.3]),60:np.array([0,1.5,2.7,3.8,4.8,4.9,4.8])}
NS=lambda t:-0.67*t/13.0
YB_TC=23.95

# Panel: net-of-placebo (positive = drug better than placebo, points of extra Y-BOCS drop)
STEIN_WK=np.array([1.,2,4,6,8,10,12,16,20,24]); STEIN_WT=np.array([0,0,1,1,1,1,1,1,0.5,0.5])
NET_ESC10=np.array([-0.45,-0.2,0.56,0.58,1.28,2.0,2.97,3.18,2.31,3.1]); SE_ESC10=np.array([0.396,0.58,0.75,0.87,0.948,1.032,1.089,1.223,1.365,1.478])
NET_ESC20=np.array([-0.08,0.58,1.44,1.97,2.34,2.61,3.68,3.67,3.32,3.12]); SE_ESC20=np.array([0.396,0.587,0.735,0.87,0.948,1.032,1.082,1.195,1.344,1.444])
NET_PAR40=np.array([-0.01,0.48,1.66,2.48,2.52,1.97,3.21,4.12,3.35,4.24]); SE_PAR40=np.array([0.396,0.58,0.735,0.87,0.948,1.032,1.089,1.216,1.344,1.457])
YB_STEIN=27.0
ZOHAR_WK=np.array([0.,2,4,6,8,10,12]); YB_ZOHAR=25.5
ZOHAR_PAR_NET=np.array([0.,1.5,2.0,3.0,3.5,4.0,3.0]); SE_ZPAR=np.full(7,0.977)
ZOHAR_CMI_NET=np.array([0.,2.0,3.0,3.5,4.0,4.0,3.0]); SE_ZCMI=np.full(7,1.160)
GREIST_WK=np.arange(0,11.); YB_GREIST=26.2
GREIST_CMI_NET=np.array([0.,0.5,1.5,2.5,3.5,4.5,5.0,5.5,5.8,6.1,6.4]); SE_GCMI=np.ones(11)
ASKARI_WK=np.array([2.,4,6,8]); YB_ASKARI=28.4
NET_ASKARI=np.array([2.3,3.3,5.1,8.4]); SE_ASKARI=np.array([0.94,1.044,1.22,2.048])  # sign flipped to +improvement
GHOB_WK=np.array([4.,8,10]); YB_GHOB=27.5
NET_GHOB=np.array([1.9,4.6,5.7]); SE_GHOB=np.array([1.887,1.888,1.935])              # sign flipped to +improvement

D50_PIN=dict(ESC=5.0,PAR=5.0,CMI=50.0,FLV=50.0,SRT=35.0)
# study weights (from run9_seq)
W=dict(ESC=0.5,PAR=0.5,ZPAR=0.5,CMI=0.5,ZCMI=0.5,FLV=0.3,SRT=0.5)

def net_resid(dose_fn,D50,Ybase,weeks,obs,se,wt,G,gC,gamma,kd,tau,eC0,skip0=True):
    Y=trajectory(dose_fn,D50,G,gC,gamma,kd,tau,weeks,eC0)
    net=Ybase-Y                       # model net improvement vs untreated baseline
    m=slice(1,None) if skip0 else slice(None)
    r=((net[m]-obs[m])/se[m])**2
    if wt is not None: r=wt[m]*r
    return float(np.sum(r)), net

def objective(x, detail=False):
    kd,hm0,tau,gC,gamma,D50f,D50c=x
    setup(kd,hm0)
    parts={}
    # cache G per baseline
    Gcache={}
    def getG(Yt):
        if Yt not in Gcache: Gcache[Yt]=baseline_G(kd,gC,gamma,Yt)
        return Gcache[Yt]
    # --- FLX Tollefson (absolute) ---
    G,eC0=getG(YB_TC)
    if G is None: return 1e6
    s=0.0
    for d in (20,40,60):
        Ym=trajectory(const(d),D50f,G,gC,gamma,kd,tau,WK_T,eC0)+NS(WK_T)
        wt=TOLn[d]/TOLsd[d]**2; s+=np.sum(wt*(Ym-TOL[d])**2)/np.sum(wt)
    parts['FLX']=s
    # --- CIT Montgomery (change-from-baseline) ---
    Yb0=ybocs(phd1(G,gC*(eC0-eC_h)),gamma)
    s=0.0
    for d in (20,40,60):
        Ym=trajectory(const(d),D50c,G,gC,gamma,kd,tau,WK_M,eC0)+NS(WK_M)
        s+=0.5*np.mean(((Yb0-Ym)-MONc[d])**2)
    parts['CIT']=s
    # --- ESC Stein 10/20 ---
    G,eC0=getG(YB_STEIN)
    if G is None: return 1e6
    se=0.0
    for d,obs,sd in [(10,NET_ESC10,SE_ESC10),(20,NET_ESC20,SE_ESC20)]:
        r,_=net_resid(const(d),D50_PIN['ESC'],YB_STEIN,STEIN_WK,obs,sd,STEIN_WT,G,gC,gamma,kd,tau,eC0,skip0=False)
        se+=W['ESC']*r
    parts['ESC']=se
    # --- PAR Stein 40 ---
    r,_=net_resid(const(40),D50_PIN['PAR'],YB_STEIN,STEIN_WK,NET_PAR40,SE_PAR40,STEIN_WT,G,gC,gamma,kd,tau,eC0,skip0=False)
    parts['PAR_stein']=W['PAR']*r
    # --- PAR Zohar 37.5 ---
    G,eC0=getG(YB_ZOHAR)
    r,_=net_resid(const(37.5),D50_PIN['PAR'],YB_ZOHAR,ZOHAR_WK,ZOHAR_PAR_NET,SE_ZPAR,None,G,gC,gamma,kd,tau,eC0)
    parts['PAR_zohar']=W['ZPAR']*r
    # --- CMI Zohar 113.1 (same baseline) ---
    r,_=net_resid(const(113.1),D50_PIN['CMI'],YB_ZOHAR,ZOHAR_WK,ZOHAR_CMI_NET,SE_ZCMI,None,G,gC,gamma,kd,tau,eC0)
    parts['CMI_zohar']=W['ZCMI']*r
    # --- CMI Greist 226 ---
    G,eC0=getG(YB_GREIST)
    r,_=net_resid(const(226.),D50_PIN['CMI'],YB_GREIST,GREIST_WK,GREIST_CMI_NET,SE_GCMI,None,G,gC,gamma,kd,tau,eC0)
    parts['CMI_greist']=W['CMI']*r
    # --- FLV Askari 100->200 @wk4 ---
    G,eC0=getG(YB_ASKARI)
    r,_=net_resid(titr(100,200,4.),D50_PIN['FLV'],YB_ASKARI,ASKARI_WK,NET_ASKARI,SE_ASKARI,None,G,gC,gamma,kd,tau,eC0,skip0=False)
    parts['FLV']=W['FLV']*r
    # --- SRT Ghob 100->200 @wk4 ---
    G,eC0=getG(YB_GHOB)
    r,_=net_resid(titr(100,200,4.),D50_PIN['SRT'],YB_GHOB,GHOB_WK,NET_GHOB,SE_GHOB,None,G,gC,gamma,kd,tau,eC0,skip0=False)
    parts['SRT']=W['SRT']*r
    tot=sum(parts.values())
    if detail: return tot, parts
    return tot

# kd,        h_max0,     tau,      gamma_C,      gamma_obs,  D50_flx,  D50_cit
BOUNDS=[(0.02,0.5),(100.,3000.),(1.,8.),(-1.5,-0.01),(0.4,3.0),(2.,80.),(2.,80.)]
NAMES=['kappa_des','h_max0','tau_desens','gamma_C','gamma_obs','D50_flx','D50_cit']

if __name__=='__main__':
    if '--test' in sys.argv:
        x0=[0.168,2500,3.8,-0.3,1.0,4.75,14.0]
        t0=time.time(); tot,parts=objective(x0,detail=True)
        print(f"test objective={tot:.3f} in {time.time()-t0:.2f}s/eval")
        for k,v in parts.items(): print(f"  {k:12s} {v:.3f}")
    else:
        print("starting differential_evolution (panel)...",flush=True); t0=time.time()
        res=differential_evolution(objective,BOUNDS,maxiter=80,popsize=20,tol=1e-6,seed=1,polish=True,disp=True,workers=-1)
        dt=(time.time()-t0)/60
        tot,parts=objective(res.x,detail=True)
        print(f"\nDONE in {dt:.1f} min; SSE={res.fun:.4f}")
        with open('fit_ssri_panel_result.txt','w') as f:
            f.write(f"SSE={res.fun}\n")
            for n,v,b in zip(NAMES,res.x,BOUNDS):
                rail=''
                if abs(v-b[0])<1e-3*(b[1]-b[0]): rail='  <-- RAILED (lower)'
                if abs(v-b[1])<1e-3*(b[1]-b[0]): rail='  <-- RAILED (upper)'
                line=f"{n} = {v:.5f}{rail}"; f.write(line+"\n"); print(line)
            f.write("\nper-study SSE:\n")
            for k,v in parts.items(): f.write(f"  {k:12s} {v:.4f}\n"); print(f"  {k:12s} {v:.4f}")
        np.save('fit_ssri_panel_bestx.npy',res.x)
        print("wrote fit_ssri_panel_result.txt, fit_ssri_panel_bestx.npy")
