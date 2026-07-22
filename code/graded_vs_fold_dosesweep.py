"""Dose-sweep of bimodality coefficient (BC, Hosenfeld), variability ratio (VR, Munkholm),
and remission% for the FOLD population vs the GRADED null. Characterizes where the fold predicts
variance INFLATION (VR>1) vs compression, to frame the Munkholm/Hosenfeld/Stone confrontation honestly."""
import numpy as np, json
import build_fit_gcs_slow as M, gcs_bistable_population as B, gamma_profile_calib as GP
from graded_vs_fold_signatures import fold_equil_from_OCD, graded_equil, eC_ss, cdrug, THM, bimod_coef, VR
rng=np.random.default_rng(7)
Yb=np.clip(rng.normal(24.,5.5,3000),10.,39.5)
pl=[]
for y in Yb:
    a,G0,ok=GP.alpha_for(y,THM)
    if a is not None: pl.append((a,G0,y))
Yb0=np.array([y for _,_,y in pl])
print(f"N placed={len(pl)}  baseline SD={Yb0.std(ddof=1):.2f}")
print(f"{'u':>5}{'remit_fold':>11}{'remit_grad':>11}{'BC_fold':>9}{'BC_grad':>9}{'VR_fold':>9}{'VR_grad':>9}")
rows=[]
for u in [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.98]:
    yf=np.array([M.Yread(fold_equil_from_OCD(a,G0,u)) for a,G0,_ in pl])
    yg=np.array([M.Yread(graded_equil(G0,u)) for a,G0,_ in pl])
    rf=100*np.mean(yf<12); rg=100*np.mean(yg<12)
    bf=bimod_coef(yf); bg=bimod_coef(yg); vf=VR(yf,Yb0); vg=VR(yg,Yb0)
    print(f"{u:>5.2f}{rf:>11.1f}{rg:>11.1f}{bf:>9.3f}{bg:>9.3f}{vf:>9.3f}{vg:>9.3f}")
    rows.append(dict(u=u,remit_fold=rf,remit_grad=rg,BC_fold=bf,BC_grad=bg,VR_fold=vf,VR_grad=vg))
json.dump(rows,open('dosesweep.json','w'),indent=2)
print("\nwrote dosesweep.json")
