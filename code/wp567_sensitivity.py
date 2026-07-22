"""
WP5/WP6/WP7 sensitivity analyses (all body-neutral -> SI).
 WP7 (engine extrapolation, R3W1/R3W4): sensitivity of the maximal-SSRI extracellular-serotonin ceiling
     across the range-bounded engine constants (k_rem/k_SERT, h_max, tau_des, kappa_des); and the largest
     eC any calibration arm actually reaches (how far the toxicity claim extrapolates past data).
 WP6 (readout bands, R5W4): fold occupancy u_fold(Y0) and eC-multiple-to-fold as the readout law
     Y=40*((phd1-phih)/(Q-phih))^gamma is varied within its defensible family (gamma, anchors phih, Q).
 WP5 (posited-not-tuned, R1W1): repeat u_fold(Y0) with an ALTERNATIVE plasticity source of a different
     functional form but the SAME three ordered roots (piecewise-linear tents vs the BCM cubic); and a
     joint gamma x theta_M recompute of the severity-gating.
"""
import numpy as np, json
import build_fit_gcs_slow as M, gcs_bistable_population as B
import gcs_bistable_calibrate as C, gamma_profile_calib as GP
bf=json.load(open('best_fit.json'))
THM=9.15; HMAX=2.0; PS=GP.PHI_SAT; BB=bf['b']; KD=bf['kappa_des']; eC_h=M.eC_h
PHIH=M.phid1_h; QD1=M.Qd1
M.GAMMA_OBS=2.0; M.b_5HT1B=BB; M.setup(KD,HMAX); GP.KAPPA_DES=KD

# ---------- helpers ----------
def eC_ss_engine(u, krem_ratio=0.15, hmax0=HMAX, kappa_des=KD, tau_d=2.0):
    """max engine eC at occupancy u under given engine constants."""
    M.kOCT3=krem_ratio*M.kSERT0; M.setup(kappa_des,hmax0)
    if u<=1e-6: return eC_h
    D50=10.; D=u*D50/(1-u); tg=np.arange(0,80,0.2)
    # replicate ec_series with this kappa_des/tau
    th=M.theta_h; e=eC_h; dt=tg[1]-tg[0]
    for t in tg:
        e=M.eC_qss(th,D,D50); th+=dt*((1-th)-kappa_des*(e-eC_h)*th)/tau_d
    M.kOCT3=0.15*M.kSERT0; M.setup(KD,HMAX)   # restore
    return e

# WP7: engine sensitivity of the maximal-SSRI eC (u=0.95) and the eC-fold ceiling is set by plasticity (b), not engine
print("=== WP7: maximal-SSRI eC ceiling (u=0.95) across engine ranges (multiples of eC_h=2.8) ===")
base=eC_ss_engine(0.95); print(f"  canonical: {base:.2f} nM = {base/eC_h:.2f}x baseline")
vals=[]
for kr in (0.10,0.20):
    v=eC_ss_engine(0.95,krem_ratio=kr); vals.append(v); print(f"  k_rem/k_SERT={kr}: {v:.2f} nM = {v/eC_h:.2f}x")
for hm in (1.5,5.0):
    v=eC_ss_engine(0.95,hmax0=hm); vals.append(v); print(f"  h_max={hm}: {v:.2f} nM = {v/eC_h:.2f}x")
for td in (2.0,4.0):
    v=eC_ss_engine(0.95,tau_d=td); vals.append(v); print(f"  tau_des={td}wk: {v:.2f} nM = {v/eC_h:.2f}x")
allv=vals+[base]; print(f"  RANGE: {min(allv)/eC_h:.2f}x -- {max(allv)/eC_h:.2f}x baseline")
# largest eC any calibration arm reaches (at its actual dose, high-dose arms)
print("  largest eC any CALIBRATION arm reaches (at its dose):")
arms=[("FLX60",60,2.7),("CIT60",60,3.4),("PAR40",40,5.0),("SRT200",200,9.1),("FLV200",200,50.)]
mx=0
for nm,d,D50 in arms:
    tg=np.arange(0,80,0.2); e=float(C.ec_series(B.const(d),D50,tg)[-1]); mx=max(mx,e)
    print(f"    {nm}: {e:.2f} nM = {e/eC_h:.2f}x")
print(f"  => data reach at most {mx/eC_h:.2f}x baseline; the eC-fold ladder (6-25x) extrapolates BEYOND the data.")

# ---------- fold machinery (eC as free param, plasticity fold) ----------
def Sshape(G,thM,src='bcm'):
    p=B.phd1(G); phe=B.phe(G)
    if src=='bcm': return phe*(p-PHIH)*(p-thM)*(PS-p)
    if src=='pwl':  # piecewise-linear tents, SAME three roots (PHIH, thM, PS)
        if p<=PHIH or p>=PS: return 0.0
        if p<thM:  # negative tent between PHIH and thM
            m=0.5*(PHIH+thM); h=-(thM-PHIH); return phe*h*(1-abs(p-m)/(0.5*(thM-PHIH)))*8000
        m=0.5*(thM+PS); h=(PS-thM); return phe*h*(1-abs(p-m)/(0.5*(PS-thM)))*8000
def alpha_for(Yb,thM,gam=2.0,phih=PHIH,Q=QD1,src='bcm'):
    M.GAMMA_OBS=gam; M.phid1_h=phih; M.Qd1=Q
    try: G=M.G_of_Y(Yb)
    except Exception: M.phid1_h=PHIH; M.Qd1=QD1; M.GAMMA_OBS=2.0; return None,None
    p=B.phd1(G)
    S=Sshape(G,thM,src)
    M.phid1_h=PHIH; M.Qd1=QD1; M.GAMMA_OBS=2.0
    if p<=thM or p>=PS or S<=0: return None,None
    return G/S, G
def has_attr(al,thM,ecm,src='bcm'):
    Gs=np.linspace(1e-4,2.7,3000); f=al*np.array([Sshape(g,thM,src) for g in Gs])-(1.0+BB*(ecm-eC_h))*Gs
    for i in np.where(np.diff(np.sign(f))!=0)[0]:
        g=Gs[i]-f[i]*(Gs[i+1]-Gs[i])/(f[i+1]-f[i])
        if g>0.25 and (f[i]-f[i-1] if i>0 else -1)<0: return True
    return False
def u_fold(Y0,gam=2.0,phih=PHIH,Q=QD1,thM=THM,src='bcm'):
    # eC as free param via occupancy engine (canonical engine)
    al,G0=alpha_for(Y0,thM,gam,phih,Q,src)
    if al is None: return np.nan
    for u in np.linspace(0,0.985,300):
        D50=10.; D=u*D50/(1-u) if u>0 else 0.
        ec=eC_h if u<=0 else float(C.ec_series(B.const(D),D50,np.arange(0,80,0.5))[-1])
        if not has_attr(al,thM,ec,src): return u
    return np.nan

print("\n=== WP6: readout-family bands on u_fold(Y-BOCS0=20) ===")
base=u_fold(20); print(f"  canonical (gamma=2, anchors 7.4/65): u_fold={base:.3f}")
b6=[base]
for g in (1.5,1.75,2.0):
    v=u_fold(20,gam=g); b6.append(v); print(f"  gamma={g}: u_fold={v:.3f}")
for Q in (60,70):
    v=u_fold(20,Q=Q); b6.append(v); print(f"  Q_d1={Q}: u_fold={v:.3f}")
b6=[x for x in b6 if not np.isnan(x)]
print(f"  readout band: u_fold(Y0=20) in [{min(b6):.3f},{max(b6):.3f}]")

print("\n=== WP5: alternative source (piecewise-linear, same 3 roots) vs BCM cubic ===")
print(f"  {'Y0':>4}{'u_fold BCM':>12}{'u_fold PWL':>12}")
for Y0 in (16,20,24):
    print(f"  {Y0:>4}{u_fold(Y0,src='bcm'):>12.3f}{u_fold(Y0,src='pwl'):>12.3f}")
print("\n=== WP5: joint gamma x theta_M severity-gating (u_fold Y0=20) ===")
for g in (1.5,2.0):
    for tm in (8.0,11.0):
        print(f"  gamma={g}, theta_M={tm}: u_fold(Y0=20)={u_fold(20,gam=g,thM=tm):.3f}")
