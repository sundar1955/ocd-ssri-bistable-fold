"""Detectability / sample-size for the endpoint-distribution (bimodality) discriminator.
Question a data-holder would ask: at a clinical dose, how many trial endpoints N are needed for the fold's
bimodal endpoint distribution to be told apart from the graded model's unimodal one?

Method (all on MODEL-SIMULATED cohorts, no patient data):
  1. Build the two 'truth' endpoint distributions (fold, graded) at u_clin from the calibrated machinery.
  2. Decision rule = sample bimodality coefficient BC. Calibrate the BC threshold so the graded (unimodal)
     model triggers a false 'bimodal' call at <=5%. Then find the N at which the fold model is correctly
     called bimodal >=80% of the time (power).  Also report the VR>1 one-sided test for comparison.
"""
import numpy as np, json
import build_fit_gcs_slow as M, gcs_bistable_population as B
import gcs_bistable_calibrate as C, gamma_profile_calib as GP

bf=json.load(open('best_fit.json'))
THM=9.15; HMAX=2.0; BB=bf['b']; KD=bf['kappa_des']; PS=GP.PHI_SAT; eC_h=M.eC_h
M.GAMMA_OBS=2.0; M.b_5HT1B=BB; M.setup(KD,HMAX); GP.KAPPA_DES=KD
_ucache={}
def eC_ss(u):
    if u<=1e-6: return eC_h
    if u in _ucache: return _ucache[u]
    D50=10.; D=u*D50/(1-u); tg=np.arange(0,80,0.2); v=float(C.ec_series(lambda t:D,D50,tg)[-1])
    _ucache[u]=v; return v
def cdrug(u): return 1.0+BB*(eC_ss(u)-eC_h)
_GGRID=np.linspace(1e-4,2.9,3600); _Scache={}
def RHS_fold(G,al,u): return al*B.Sshape(G,THM,PS)-cdrug(u)*G
def _Sgrid(al):
    if al not in _Scache: _Scache[al]=al*np.array([B.Sshape(g,THM,PS) for g in _GGRID])
    return _Scache[al]
def fold_equil_from_OCD(al,u):
    f=_Sgrid(al)-cdrug(u)*_GGRID; sc=np.where(np.diff(np.sign(f))!=0)[0]; stables=[]
    for i in sc:
        g=_GGRID[i]-f[i]*(_GGRID[i+1]-_GGRID[i])/(f[i+1]-f[i])
        if (RHS_fold(g+1e-3,al,u)-RHS_fold(g-1e-3,al,u))<0: stables.append(g)
    return max(stables) if stables else 1e-4
def graded_equil(G0,u): return G0/cdrug(u)

def occ(mg,D50): return mg/(mg+D50)
u_clin=occ(40.,10.)                       # ~0.80

# ---- 'truth' endpoint pools (large) ----
rng=np.random.default_rng(11)
Nbig=9000; Yb=np.clip(rng.normal(24.,5.5,Nbig),10.,39.5)
EF=[]; EG=[]
for yb in Yb:
    al,G0,ok=GP.alpha_for(yb,THM)
    if al is None: continue
    EF.append(M.Yread(fold_equil_from_OCD(al,u_clin)))
    EG.append(M.Yread(graded_equil(G0,u_clin)))
EF=np.clip(np.array(EF),0,40); EG=np.clip(np.array(EG),0,40)
BASE=Yb[(Yb>0)][:len(EF)]

def bimod_coef(x):
    x=np.asarray(x,float); n=len(x); s=(x-x.mean())/x.std()
    g1=np.mean(s**3); k=np.mean(s**4)-3
    return (g1**2+1)/(k+3*(n-1)**2/((n-2)*(n-3)))
print(f"u_clin={u_clin:.2f}  truth BC: fold={bimod_coef(EF):.3f} graded={bimod_coef(EG):.3f};"
      f"  truth VR: fold={EF.std()/BASE.std():.2f} graded={EG.std()/BASE.std():.2f}")

# ---- sampling power vs N ----
def power_curve(pool_true, pool_false, base_pool, nrep=2000):
    out=[]
    for N in Ns:
        bc_f=np.empty(nrep); bc_g=np.empty(nrep); vr_f=np.empty(nrep)
        for r in range(nrep):
            xf=pool_true[rng.integers(0,len(pool_true),N)]
            xg=pool_false[rng.integers(0,len(pool_false),N)]
            bb=base_pool[rng.integers(0,len(base_pool),N)]
            bc_f[r]=bimod_coef(xf); bc_g[r]=bimod_coef(xg); vr_f[r]=xf.std(ddof=1)/bb.std(ddof=1)
        # BC threshold set to 95th percentile of graded (=> 5% false-positive), power = P(fold BC > thr)
        thr=np.quantile(bc_g,0.95)
        out.append((N, np.mean(bc_f>thr), np.mean(bc_f>0.55), np.mean(vr_f>1.0), thr))
    return out

Ns=[30,50,75,100,150,200,300,400]
res=power_curve(EF,EG,BASE)
print("\n  N   power(BC>graded-95%)  P(BC>0.55)   P(VR>1)   BC-threshold")
for N,pw,p55,pvr,thr in res:
    print(f"{N:>4d}      {pw:5.2f}              {p55:5.2f}       {pvr:5.2f}      {thr:.3f}")
# smallest N reaching 80% power on the calibrated BC test
hit=[N for N,pw,_,_,_ in res if pw>=0.80]
print(f"\n>=80% power (BC test, 5% false-positive) at N ~ {hit[0] if hit else '>400'} endpoints per arm")
json.dump({'u_clin':u_clin,'Ns':Ns,'power_bc':[r[1] for r in res],'p_vr':[r[3] for r in res],
           'N80':(hit[0] if hit else None)}, open('graded_vs_fold_detectability.json','w'), indent=1)
