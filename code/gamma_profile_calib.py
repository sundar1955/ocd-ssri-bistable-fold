"""
GAMMA PROFILE in the ALPHA-PLACEMENT framework (phi_sat=65 universal; alpha patient-specific,
back-solved from baseline Y-BOCS). For each gamma we fit the 4 SHARED structural params
(theta_M, tau_G, b, h_max0) to the net(drug-placebo) SSRI trajectories, and report:
  SSE*(gamma)            -- trajectory-fit quality
  excluded fraction      -- population mass below the fold (unplaceable) = the GHOST-ZONE COST
  Y_fold(theta_M,gamma)  -- Y-BOCS of the fold at the fitted theta_M
Decision: want a gamma with good SSE AND ~0 excluded fraction (ghost zone closed).
"""
import numpy as np, time
from scipy.optimize import differential_evolution, brentq
import build_fit_gcs_slow as M
import gcs_bistable_population as B
import gcs_bistable_calibrate as C
P=M.P; KAPPA_DES=0.120; TAU_D=2.0; PHI_SAT=65.0   # KAPPA_DES=calibrated (baseline-ref fit)

def alpha_for(Yb, thM):
    """back-solve patient LTP gain alpha so the untreated attractor sits at baseline Yb; phi_sat=65.
       Normalized plasticity: at baseline (eC=eC_h) the sink coefficient is 1, so alpha = G / S(G).
       returns (alpha, G*, placeable_stable_attractor?)."""
    G=M.G_of_Y(Yb); p=B.phd1(G)
    if p<=thM or p>=PHI_SAT: return None,G,False
    Sunit=B.Sshape(G,thM,PHI_SAT)   # baseline-referenced BCM (single source of truth)
    if Sunit<=0: return None,G,False
    alpha=G/Sunit                    # baseline sink coefficient = 1
    dG=1e-3
    f=lambda g: alpha*B.Sshape(g,thM,PHI_SAT)-g
    stable=(f(G+dG)-f(G-dG))/(2*dG)<0        # placed point is the STABLE attractor (not the saddle)
    return alpha,G,stable

def net_model_alpha(st, thM, tau_G, tgrid, eC_arr):
    ys,ws=C.quad(st['Ymean'],st['Ysd'],n=15); eC_h=M.eC_h
    dt=tgrid[1]-tgrid[0]; wk_i=[np.argmin(np.abs(tgrid-w)) for w in st['weeks']]
    acc=np.zeros(len(st['weeks'])); wsum=0.0; wexcl=0.0; b=M.b_5HT1B
    for Yb,w in zip(ys,ws):
        alpha,G0,ok=alpha_for(Yb,thM)
        if not ok: wexcl+=w; continue
        G=G0; Yv=np.empty(len(tgrid))
        for i in range(len(tgrid)):
            Yv[i]=M.Yread(G); e=eC_arr[i]
            G+=dt*(alpha*B.Sshape(G,thM,PHI_SAT)-(1.0+b*(e-eC_h))*G)/tau_G
            G=min(max(G,0.01),3.2)
        acc+=w*(Yb-Yv[wk_i]); wsum+=w
    return (acc/wsum if wsum>0 else acc*np.nan), wexcl/(wsum+wexcl+1e-12)

def objective_alpha(x, studies, detail=False):
    thM,tau_G,b,h_max0=x; M.b_5HT1B=b; M.setup(KAPPA_DES,h_max0)
    tmax=max(s['weeks'][-1] for s in studies)+0.001; tgrid=np.arange(0,tmax,0.15)
    tot=0.0; ec_cache={}; fexcls=[]
    for st in studies:
        key=(id(st['dose_fn']),st['D50'])
        if key not in ec_cache: ec_cache[key]=C.ec_series(st['dose_fn'],st['D50'],tgrid)
        nm,fe=net_model_alpha(st,thM,tau_G,tgrid,ec_cache[key]); fexcls.append(fe)
        if np.any(np.isnan(nm)): return (1e6,fexcls) if detail else 1e6
        m=slice(1,None) if st['skip0'] else slice(None)
        r=st['wt'][m]*((nm[m]-st['net'][m])/st['se'][m])**2
        tot+=st['W']*np.sum(r)/max(np.sum(st['wt'][m]),1e-9)
    return (tot,fexcls) if detail else tot

def Yfold_at(thM, gam):
    """fold = elasticity E(G)=1 of S(G); return its Y-BOCS at fitted theta_M and gamma."""
    old=M.GAMMA_OBS; M.GAMMA_OBS=gam
    def S(G): return B.Sshape(G,thM,PHI_SAT)   # baseline-referenced BCM (single source of truth)
    dS=lambda G:(S(G+1e-4)-S(G-1e-4))/2e-4
    Gs=np.linspace(0.4,2.6,3000); E=np.array([g*dS(g)/S(g) if S(g)>0 else 9 for g in Gs])
    idx=np.where(np.diff(np.sign(E-1))!=0)[0]
    yf=M.Yread(Gs[idx[0]]) if len(idx) else np.nan; M.GAMMA_OBS=old; return yf

STUD=C.build_studies(include_cmi=False)
BOUNDS=[(1.5,22.),(0.8,20.),(0.005,0.05),(150.,4000.)]  # thM,tau_G,b,h_max0 (normalized form, no kappa_h)

class Obj:                       # picklable objective (stores only gamma; workers use module-global STUD)
    def __init__(self,gam): self.gam=gam
    def __call__(self,x): M.GAMMA_OBS=self.gam; return objective_alpha(x,STUD)

if __name__=='__main__':
    print(f"alpha-placement gamma profile: {len(STUD)} SSRI arms, 5 shared params each gamma (parallel)\n")
    print(f"{'gamma':>6} {'SSE*':>9} {'excl.frac':>10} {'Y_fold':>7} {'thM':>6} {'tau_G':>6} {'b':>7}  ghost-zone")
    rows=[]
    for gam in [0.5,0.75,1.0,1.25,1.5,2.0]:
        M.GAMMA_OBS=gam; t0=time.time()
        res=differential_evolution(Obj(gam),BOUNDS,maxiter=14,popsize=10,tol=1e-6,
                                   seed=1,polish=True,workers=1)
        M.GAMMA_OBS=gam; sse,fex=objective_alpha(res.x,STUD,detail=True)
        thM,tau_G,b,h_max0=res.x; yf=Yfold_at(thM,gam); mfx=100*np.mean(fex)
        gz='CLOSED' if yf<=16 else f'Y<{yf:.0f} open'
        print(f"{gam:>6.2f} {sse:>9.3f} {mfx:>9.0f}% {yf:>7.1f} {thM:>6.1f} {tau_G:>6.1f} {b:>7.3f}  {gz}   ({time.time()-t0:.0f}s)",flush=True)
        rows.append((gam,sse,mfx,yf,thM,tau_G,b))
    np.save('gamma_profile_rows.npy',np.array([r[:4] for r in rows]))
    print("\ninterpretation: pick gamma with LOW SSE and LOW excl.frac (ghost zone closed = Y_fold<=16).")
