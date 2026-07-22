"""Quick Gm profile at fixed tau, + test the Gm->0 limit = indirect-response (IDR) null G*=G0/(1+b dEc).
Decide whether to present the saturating twin (interior Gm) or the cleaner IDR limit as THE graded null."""
import numpy as np, json
import build_fit_gcs_slow as M, gcs_bistable_calibrate as C, gamma_profile_calib as GP
bf=json.load(open('best_fit.json')); THM=9.15; HMAX=2.0; BB=bf['b']; KD=bf['kappa_des']; eC_h=M.eC_h
M.GAMMA_OBS=2.0; M.b_5HT1B=BB; M.setup(KD,HMAX); GP.KAPPA_DES=KD
STUD=C.build_studies(include_cmi=False)
from graded_twin_fit import objective_graded

print("Gm profile at tau_G_g=19.0:")
for Gm in [0.02,0.05,0.1,0.2,0.5,1.0,2.0,4.0]:
    print(f"  Gm={Gm:6.2f}  chi2={objective_graded((Gm,19.0),STUD):.3f}")

# IDR limit as its own model: G* = G0/(1+b dEc); integrate tau dG/dt = (1+b dEc)*(G0*... )
# Implement exactly as Gm->0 limit of the saturating source: alpha_g*Sm -> (Gm+G0)/Gm * Gm = G0 (source->const G0)
# so tau dG/dt = G0 - (1+b(eC-eC_h))*G  => equilibrium G0/(1+b dEc). Test tau sweep.
def objective_idr(tau_G,studies):
    tmax=max(s['weeks'][-1] for s in studies)+0.001; tgrid=np.arange(0,tmax,0.15); dt=tgrid[1]-tgrid[0]
    tot=0.0; ec_cache={}
    for st in studies:
        key=(id(st['dose_fn']),st['D50'])
        if key not in ec_cache: ec_cache[key]=C.ec_series(st['dose_fn'],st['D50'],tgrid)
        eC_arr=ec_cache[key]; ys,ws=C.quad(st['Ymean'],st['Ysd'],n=15)
        wk_i=[np.argmin(np.abs(tgrid-w)) for w in st['weeks']]; acc=np.zeros(len(st['weeks'])); wsum=0.
        for Yb,w in zip(ys,ws):
            G0=M.G_of_Y(Yb); G=G0; Yv=np.empty(len(tgrid))
            for i in range(len(tgrid)):
                Yv[i]=M.Yread(G); e=eC_arr[i]
                G+=dt*(G0-(1.0+BB*(e-eC_h))*G)/tau_G; G=min(max(G,0.001),3.4)
            acc+=w*(Yb-Yv[wk_i]); wsum+=w
        nm=acc/wsum; m=slice(1,None) if st['skip0'] else slice(None)
        tot+=st['W']*np.sum(st['wt'][m]*((nm[m]-st['net'][m])/st['se'][m])**2)/max(np.sum(st['wt'][m]),1e-9)
    return tot
print("\nIDR limit G*=G0/(1+b dEc), tau sweep:")
best=(1e9,None)
for tau in [8,10,12,14,16,18,20,22,25]:
    c=objective_idr(tau,STUD); print(f"  tau={tau:5.1f}  chi2={c:.3f}")
    if c<best[0]: best=(c,tau)
print(f"\nIDR best: tau={best[1]}  chi2={best[0]:.3f}   (fold chi2=6.269)")
json.dump({'idr_best_tau':best[1],'idr_chi2':best[0]},open('graded_idr.json','w'))
