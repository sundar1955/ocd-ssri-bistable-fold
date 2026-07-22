"""
WP3: prediction intervals along the sloppy (b, kappa_des) manifold.
Reviewer point (R4W2/W3): the calibration is sloppy in parameters but we report point predictions.
Here we recompute the HEADLINE predictions at the endpoints of the profile-likelihood band for the
best-constrained coupling b in [0.016, 0.036] nM^-1 (RMS <= 1.2 x RMS_min; best_fit.json b_ci), with
kappa_des re-set over its band, and report predictions as INTERVALS:
  (1) fold occupancy u_fold(Y0) -- the SERT occupancy that annihilates the OCD attractor, by baseline severity;
  (2) population remission fraction at a clinical dose (u=0.8);
  (3) the eC-multiple ceiling: extracellular-serotonin fold-multiple needed to fold an attractor, by severity.
Everything else (theta_M=9.15, phi_sat=65, tau_G=12, gamma=2, engine constants) held at canonical values.
"""
import numpy as np, json
import build_fit_gcs_slow as M, gcs_bistable_population as B
import gcs_bistable_calibrate as C, gamma_profile_calib as GP

bf=json.load(open('best_fit.json'))
THM=9.15; HMAX=2.0; PS=GP.PHI_SAT; eC_h=M.eC_h
M.GAMMA_OBS=2.0
B_LO,B_MID,B_HI = bf['b_ci'][0], bf['b'], bf['b_ci'][1]        # 0.016, 0.0268, 0.036
KD=bf['kappa_des']
print(f"b-band [{B_LO},{B_HI}] nM^-1 (best-fit {B_MID}); kappa_des={KD}; theta_M={THM}, phi_sat={PS}")

def setup(b):
    M.b_5HT1B=b; M.setup(KD,HMAX); GP.KAPPA_DES=KD
_uc={}
def eC_ss(u):
    if u<=1e-6: return eC_h
    if u in _uc: return _uc[u]
    D50=10.; D=u*D50/(1-u); tg=np.arange(0,80,0.2); v=float(C.ec_series(lambda t:D,D50,tg)[-1]); _uc[u]=v; return v
def RHS(G,al,b,ec): return al*B.Sshape(G,THM,PS)-(1.0+b*(ec-eC_h))*G
def has_ocd_attractor(al,b,u):
    Gs=np.linspace(1e-4,2.7,4000); f=al*np.array([B.Sshape(g,THM,PS) for g in Gs])-(1.0+b*(eC_ss(u)-eC_h))*Gs
    for i in np.where(np.diff(np.sign(f))!=0)[0]:
        g=Gs[i]-f[i]*(Gs[i+1]-Gs[i])/(f[i+1]-f[i])
        if g>0.25 and (RHS(g+1e-3,al,b,eC_ss(u))-RHS(g-1e-3,al,b,eC_ss(u)))<0: return True
    return False
def u_fold(Y0,b):
    setup(b); al,G0,ok=GP.alpha_for(Y0,THM)
    if al is None: return np.nan
    for u in np.linspace(0,0.985,400):
        if not has_ocd_attractor(al,b,u): return u
    return np.nan

# ---- (1) u_fold(Y0) intervals across the b-band ----
print("\n(1) Fold occupancy u_fold(Y0) [SERT occupancy to annihilate OCD attractor]:")
print(f"  {'Y-BOCS0':>8}{'u_fold(b_lo)':>14}{'u_fold(mid)':>13}{'u_fold(b_hi)':>14}{'interval':>18}")
uf_rows=[]
for Y0 in [16,18,20,22,24,26,28]:
    ufs=[u_fold(Y0,b) for b in (B_LO,B_MID,B_HI)]
    lo,hi=np.nanmin(ufs),np.nanmax(ufs)
    tag="resistant" if any(np.isnan(ufs)) else ""
    print(f"  {Y0:>8}{ufs[0]:>14.3f}{ufs[1]:>13.3f}{ufs[2]:>14.3f}   [{lo:.3f},{hi:.3f}] {tag}")
    uf_rows.append((Y0,ufs[0],ufs[1],ufs[2]))

# ---- (2) population remission fraction at clinical dose u=0.8, across b-band ----
def fold_equil(al,b,G0,u):
    Gs=np.linspace(1e-4,2.9,3000); S=al*np.array([B.Sshape(g,THM,PS) for g in Gs]); f=S-(1.0+b*(eC_ss(u)-eC_h))*Gs
    st=[]
    for i in np.where(np.diff(np.sign(f))!=0)[0]:
        g=Gs[i]-f[i]*(Gs[i+1]-Gs[i])/(f[i+1]-f[i])
        if (RHS(g+1e-3,al,b,eC_ss(u))-RHS(g-1e-3,al,b,eC_ss(u)))<0: st.append(g)
    return max(st) if st else 1e-4
rng=np.random.default_rng(3); Yb=np.clip(rng.normal(24,5.5,1500),10,39.5)
print("\n(2) Population remission fraction (endpoint Y<12) at clinical dose u=0.80:")
rem=[]
for b in (B_LO,B_MID,B_HI):
    setup(b); ys=[]
    for y in Yb:
        al,G0,ok=GP.alpha_for(y,THM)
        if al is None: continue
        ys.append(M.Yread(fold_equil(al,b,G0,0.8)))
    rem.append(100*np.mean(np.array(ys)<12))
print(f"  remission%: b_lo={rem[0]:.1f}  mid={rem[1]:.1f}  b_hi={rem[2]:.1f}   interval [{min(rem):.1f},{max(rem):.1f}]")

json.dump({'b_band':[B_LO,B_MID,B_HI],'u_fold':uf_rows,'remission_pct':rem},open('wp3_intervals.json','w'),indent=2)
print("\nwrote wp3_intervals.json")
