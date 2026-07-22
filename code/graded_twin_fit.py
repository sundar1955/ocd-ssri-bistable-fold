"""
STAGE 2: build the MATCHED GRADED (no-fold) twin and fit it to the SAME 13 net-improvement
SSRI arms as the canonical fold model, so 'matched on the data' is earned quantitatively.

Design principle (fair 'delete-the-fold' null):
  Fold model  : tau_G dG/dt = alpha*S_BCM(G) - (1+b(eC-eC_h))*G ,  S_BCM non-monotone (hump) -> saddle-node fold.
  Graded twin : tau_G dG/dt = alpha_g*Sm(G)   - (1+b(eC-eC_h))*G ,  Sm(G)=G/(1+G/Gm) MONOTONE saturating
                -> single stable fixed point G*=Gm(alpha_g/c-1), moves SMOOTHLY with drug, NO fold.
Everything else identical: same eC(t) engine, same b, same readout Y=Yread(G), same per-patient baseline
placement (alpha_g set so untreated attractor sits at baseline Y). Free shared params: Gm, tau_G_g (2, i.e.
FEWER than the fold model's 4). Fit to the same net(drug-placebo) arms with the same weighted objective.
"""
import numpy as np, json, time
from scipy.optimize import differential_evolution, brentq
import build_fit_gcs_slow as M
import gcs_bistable_population as B
import gcs_bistable_calibrate as C
import gamma_profile_calib as GP

bf=json.load(open('best_fit.json'))
THM=9.15; HMAX=2.0; BB=bf['b']; KD=bf['kappa_des']; PS=GP.PHI_SAT; eC_h=M.eC_h
M.GAMMA_OBS=2.0; M.b_5HT1B=BB; M.setup(KD,HMAX); GP.KAPPA_DES=KD
STUD=C.build_studies(include_cmi=False)

# ---------------- graded twin: monotone saturating source ----------------
def alpha_g_for(Yb,Gm):
    """place untreated graded attractor at baseline Yb: at c=1, G*=Gm(alpha_g-1)=G0 -> alpha_g=1+G0/Gm."""
    G0=M.G_of_Y(Yb); return 1.0+G0/Gm, G0
def Sm(G,Gm): return G/(1.0+G/Gm)

def net_model_graded(st,Gm,tau_G,tgrid,eC_arr):
    ys,ws=C.quad(st['Ymean'],st['Ysd'],n=15)
    dt=tgrid[1]-tgrid[0]; wk_i=[np.argmin(np.abs(tgrid-w)) for w in st['weeks']]
    acc=np.zeros(len(st['weeks'])); wsum=0.0
    for Yb,w in zip(ys,ws):
        alpha_g,G0=alpha_g_for(Yb,Gm); G=G0; Yv=np.empty(len(tgrid))
        for i in range(len(tgrid)):
            Yv[i]=M.Yread(G); e=eC_arr[i]
            G+=dt*(alpha_g*Sm(G,Gm)-(1.0+BB*(e-eC_h))*G)/tau_G
            G=min(max(G,0.001),3.4)
        acc+=w*(Yb-Yv[wk_i]); wsum+=w
    return acc/wsum

def objective_graded(x,studies,detail=False):
    Gm,tau_G=x
    tmax=max(s['weeks'][-1] for s in studies)+0.001; tgrid=np.arange(0,tmax,0.15)
    tot=0.0; ec_cache={}; parts={}
    for st in studies:
        key=(id(st['dose_fn']),st['D50'])
        if key not in ec_cache: ec_cache[key]=C.ec_series(st['dose_fn'],st['D50'],tgrid)
        nm=net_model_graded(st,Gm,tau_G,tgrid,ec_cache[key])
        m=slice(1,None) if st['skip0'] else slice(None)
        r=st['wt'][m]*((nm[m]-st['net'][m])/st['se'][m])**2
        c=st['W']*np.sum(r)/max(np.sum(st['wt'][m]),1e-9); parts[st['name']]=c; tot+=c
    return (tot,parts) if detail else tot

# ---------------- fold objective (canonical, for side-by-side SSE) ----------------
def objective_fold(thM,tau_G,b,h_max0,studies,detail=False):
    M.b_5HT1B=b; M.setup(KD,h_max0)
    tmax=max(s['weeks'][-1] for s in studies)+0.001; tgrid=np.arange(0,tmax,0.15)
    tot=0.0; ec_cache={}; parts={}
    for st in studies:
        key=(id(st['dose_fn']),st['D50'])
        if key not in ec_cache: ec_cache[key]=C.ec_series(st['dose_fn'],st['D50'],tgrid)
        nm,fe=GP.net_model_alpha(st,thM,tau_G,tgrid,ec_cache[key])
        m=slice(1,None) if st['skip0'] else slice(None)
        r=st['wt'][m]*((nm[m]-st['net'][m])/st['se'][m])**2
        c=st['W']*np.sum(r)/max(np.sum(st['wt'][m]),1e-9); parts[st['name']]=c; tot+=c
    M.b_5HT1B=BB; M.setup(KD,HMAX)   # restore canonical
    return (tot,parts) if detail else tot

if __name__=='__main__':
    # fold SSE at canonical params (thM=9.15,tau_G=12,b=0.0268,h_max0=HMAX)
    t0=time.time()
    fold_sse,fold_parts=objective_fold(THM,bf['tau_G'],BB,HMAX,STUD,detail=True)
    n_pts=sum((len(s['weeks'])-(1 if s['skip0'] else 0)) for s in STUD)
    print(f"FOLD (canonical) chi2={fold_sse:.3f}  RMS-like=sqrt(chi2/Wsum)  ({time.time()-t0:.1f}s)")

    # fit graded twin
    print("\nfitting graded twin (Gm, tau_G_g)...",flush=True); t0=time.time()
    res=differential_evolution(lambda x:objective_graded(x,STUD),[(0.2,6.0),(2.,30.)],
                               maxiter=40,popsize=15,tol=1e-7,seed=1,polish=True,workers=1)
    Gm,tauG=res.x; gr_sse,gr_parts=objective_graded(res.x,STUD,detail=True)
    print(f"GRADED best: Gm={Gm:.4f}  tau_G_g={tauG:.3f}  chi2={gr_sse:.3f}  ({(time.time()-t0):.1f}s)")

    print(f"\n{'arm':16s}{'fold_chi2':>11}{'graded_chi2':>13}")
    for k in fold_parts:
        print(f"{k:16s}{fold_parts[k]:>11.4f}{gr_parts.get(k,float('nan')):>13.4f}")
    print(f"{'TOTAL':16s}{fold_sse:>11.3f}{gr_sse:>13.3f}   (n_pts={n_pts})")
    json.dump({'fold_chi2':fold_sse,'graded_chi2':gr_sse,'Gm':Gm,'tau_G_g':tauG,
               'fold_parts':fold_parts,'graded_parts':gr_parts},
              open('graded_twin_fit.json','w'),indent=2)
    print("\nwrote graded_twin_fit.json")
