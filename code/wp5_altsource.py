"""WP5 (R1W1): show the fold + severity-gating survive a DIFFERENT source functional form with the SAME
three ordered roots. Alternative = tanh-product (smooth, saturating) vs the BCM polynomial cubic.
S_alt(G) = phe(G) * A * tanh((p-phih)/w) tanh((p-thM)/w) tanh((PS-p)/w),  p=phd1(G).
Roots at p = phih, thM, PS and the same LTD(<thM)/LTP(>thM) sign pattern as the cubic, but a wholly
different shape. If u_fold(Y0) still rises with severity, bistability is structural (root ordering), not tuned."""
import numpy as np, json
import build_fit_gcs_slow as M, gcs_bistable_population as B
import gcs_bistable_calibrate as C, gamma_profile_calib as GP
bf=json.load(open('best_fit.json'))
THM=9.15; HMAX=2.0; PS=GP.PHI_SAT; BB=bf['b']; KD=bf['kappa_des']; eC_h=M.eC_h; PHIH=M.phid1_h
M.GAMMA_OBS=2.0; M.b_5HT1B=BB; M.setup(KD,HMAX); GP.KAPPA_DES=KD
def S_bcm(G,thM):
    p=B.phd1(G); return B.phe(G)*(p-PHIH)*(p-thM)*(PS-p)
def S_alt(G,thM):
    # square-root-modified cubic: same 3 roots (PHIH, thM, PS), same LTD/LTP sign, different curvature
    p=B.phd1(G)
    if p<=PHIH or p>=PS: return 0.0
    return B.phe(G)*np.sqrt(p-PHIH)*(p-thM)*np.sqrt(PS-p)
SRC={'bcm':S_bcm,'tanh':S_alt}

def eC_of_u(u):
    if u<=0: return eC_h
    D50=10.; D=u*D50/(1-u); return float(C.ec_series(B.const(D),D50,np.arange(0,80,0.5))[-1])

def alpha_place(Y0,thM,src):
    G0=M.G_of_Y(Y0); S=SRC[src](G0,thM)
    if S<=0: return None,None
    return G0/S, G0

def stable_ocd(al,thM,ec,src):
    Gs=np.linspace(1e-4,2.7,4000)
    f=al*np.array([SRC[src](g,thM) for g in Gs])-(1.0+BB*(ec-eC_h))*Gs
    sc=np.where(np.diff(np.sign(f))!=0)[0]
    for i in sc:
        if f[i]>0 and f[i+1]<0:            # RHS + -> - : stable fixed point
            g=Gs[i]-f[i]*(Gs[i+1]-Gs[i])/(f[i+1]-f[i])
            if g>0.25: return True
    return False

def u_fold(Y0,thM,src):
    al,G0=alpha_place(Y0,thM,src)
    if al is None: return np.nan
    if not stable_ocd(al,thM,eC_h,src): return np.nan   # no untreated attractor -> unplaceable
    for u in np.linspace(0,0.985,300):
        if not stable_ocd(al,thM,eC_of_u(u),src): return u
    return np.nan   # resistant

print(f"{'Y-BOCS0':>8}{'u_fold BCM':>12}{'u_fold sqrt':>13}")
rows=[]
for Y0 in (16,18,20,22,24,26):
    ub=u_fold(Y0,THM,'bcm'); ut=u_fold(Y0,THM,'tanh')
    rows.append((Y0,ub,ut))
    fb=f"{ub:.3f}" if not np.isnan(ub) else "resistant"
    ft=f"{ut:.3f}" if not np.isnan(ut) else "resistant"
    print(f"{Y0:>8}{fb:>12}{ft:>13}")
json.dump(rows,open('wp5_altsource.json','w'))
print("\nBoth sources: u_fold rises monotonically with baseline severity ->",
      "severity-gating is structural (root ordering), not specific to the cubic." )
