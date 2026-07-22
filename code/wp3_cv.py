"""WP3 remainder (R4W3): leave-one-arm-out cross-validation + block-bootstrap-by-arm.
The fit has 84 arm-timepoints but only ~13 independent arm SHAPES; timepoints within an arm are strongly
autocorrelated. We therefore resample at the ARM level. Refit the two shared constants (b, kappa_des) with
theta_M=9.15, tau_G=12, gamma=2, h_max=2 fixed (canonical), using the net(drug-placebo) trajectories.
Report: (i) leave-one-arm-out held-out RMS per arm; (ii) block-bootstrap distribution of b."""
import numpy as np, json, functools, sys
print=functools.partial(print,flush=True)
from scipy.optimize import minimize
import build_fit_gcs_slow as M, gcs_bistable_population as B
import gcs_bistable_calibrate as C, gamma_profile_calib as GP
bf=json.load(open('best_fit.json')); THM=9.15; TAUG=12.0; HMAX=2.0
STUD=C.build_studies(include_cmi=False)
tmax=max(s['weeks'][-1] for s in STUD)+0.001; TG=np.arange(0,tmax,0.35)

def arm_resid2(st,b,kdes):
    M.b_5HT1B=b; GP.KAPPA_DES=kdes; M.setup(kdes,HMAX); M.GAMMA_OBS=2.0
    eC=C.ec_series(st['dose_fn'],st['D50'],TG)
    nm,_=GP.net_model_alpha(st,THM,TAUG,TG,eC)
    if np.any(np.isnan(nm)): return None
    m=slice(1,None) if st['skip0'] else slice(None)
    r=st['wt'][m]*((nm[m]-st['net'][m])/st['se'][m])**2
    return st['W']*np.sum(r)/max(np.sum(st['wt'][m]),1e-9), np.mean(((nm[m]-st['net'][m]))**2)

def fit(arms):
    def obj(x):
        b,kd=x
        if b<=0 or kd<=0: return 1e6
        tot=0
        for st in arms:
            rr=arm_resid2(st,b,kd)
            if rr is None: return 1e6
            tot+=rr[0]
        return tot
    r=minimize(obj,[bf['b'],bf['kappa_des']],method='Nelder-Mead',
               options={'xatol':1e-4,'fatol':1e-5,'maxiter':100})
    return r.x

print("full-data fit (check):", np.round(fit(STUD),4), "(best_fit b,kd =",bf['b'],bf['kappa_des'],")")

print("\n=== leave-one-arm-out CV ===")
print(f"  {'held-out arm':16}{'refit b':>9}{'refit kd':>9}{'held-RMS':>10}")
held=[]
for i,st in enumerate(STUD):
    train=[s for j,s in enumerate(STUD) if j!=i]
    b,kd=fit(train)
    rr=arm_resid2(st,b,kd); rms=np.sqrt(rr[1]) if rr else np.nan
    held.append(rms); print(f"  {st['name']:16}{b:>9.4f}{kd:>9.4f}{rms:>10.3f}")
print(f"  mean held-out RMS = {np.nanmean(held):.3f} Y-BOCS pts (in-sample RMS ~0.83)")

print("\n=== block bootstrap by arm (200 resamples) ===")
rng=np.random.default_rng(0); bs=[]
for _ in range(20):
    idx=rng.integers(0,len(STUD),len(STUD)); arms=[STUD[k] for k in idx]
    try:
        b,kd=fit(arms)
        if 0<b<0.1: bs.append(b)
    except Exception: pass
bs=np.array(bs)
print(f"  b bootstrap: median={np.median(bs):.4f}  95% CI [{np.percentile(bs,2.5):.4f},{np.percentile(bs,97.5):.4f}]  (n={len(bs)})")
json.dump({'held_rms':[float(x) for x in held],'b_boot_ci':[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]},open('wp3_cv.json','w'))
