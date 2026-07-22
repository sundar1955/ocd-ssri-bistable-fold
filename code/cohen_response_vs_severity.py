"""COHEN DIAGNOSTIC (reviewer issue #1): is model DRUG-ATTRIBUTABLE response flat in baseline
severity while REMISSION is severity-gated? Frozen canonical model (best_fit.json), alpha patient-
specific (phi_sat=65). Drug mechanism only (= drug-minus-placebo contrast, since expectation is
severity-independent). Endpoint 12 wk. Reports response>=25%, >=35%, and remission(Y<12) vs Y0."""
import numpy as np, json
import build_fit_gcs_slow as M, gamma_profile_calib as GP, gcs_bistable_population as B
C=GP.C; bf=json.load(open('best_fit.json'))
THM=bf['theta_M']; TAUG=bf['tau_G']; BB=bf['b']; KD=bf['kappa_des']; HMAX=bf['h_max']; PS=GP.PHI_SAT
M.b_5HT1B=BB; M.setup(KD,HMAX); M.GAMMA_OBS=bf['gamma']; GP.KAPPA_DES=KD
eC_h=M.eC_h
print(f"canonical: thM={THM} tau_G={TAUG} b={BB} kap_des={KD} hmax={HMAX} gamma={bf['gamma']} phi_sat={PS} eC_h={eC_h}")

def endpoint(Y0, dose, D50, weeks=12.0, dt=0.05):
    a,G0,ok=GP.alpha_for(Y0,THM)
    if not ok: return None
    tgrid=np.arange(0,weeks+dt/2,dt)
    eC=C.ec_series(B.const(dose),D50,tgrid)
    G=G0
    for i in range(len(tgrid)):
        G+=dt*(a*B.Sshape(G,THM,PS)-(1.0+BB*(eC[i]-eC_h))*G)/TAUG
        G=min(max(G,0.01),3.2)
    return M.Yread(G)

# --- deterministic %reduction(Y0) at adequate doses (FLX D50=2.7) ---
for dose,lbl in [(40,'FLX 40mg (u=0.94)'),(60,'FLX 60mg (u=0.96)')]:
    print(f"\n==== {lbl}, endpoint 12 wk ====")
    print(f"{'Y0':>4} {'Yend':>6} {'%red':>6} {'resp25':>7} {'resp35':>7} {'remit(Y<12)':>12}")
    for Y0 in range(14,39,2):
        Ye=endpoint(Y0,dose,2.7)
        if Ye is None: print(f"{Y0:>4}   (unplaceable)"); continue
        red=100*(Y0-Ye)/Y0
        print(f"{Y0:>4} {Ye:>6.1f} {red:>6.1f} {('YES' if red>=25 else 'no'):>7} "
              f"{('YES' if red>=35 else 'no'):>7} {('YES' if Ye<12 else 'no'):>12}")

# --- population response RATES binned by baseline (FLX 40mg), N patients ~ TruncNormal ---
def rates(dose,D50,Ymean,Ysd,N=4000,seed=1):
    rng=np.random.default_rng(seed); Ybs=np.clip(rng.normal(Ymean,Ysd,N),10,39.5)
    red=[]; Yend=[]; Yb=[]
    for y0 in Ybs:
        ye=endpoint(y0,dose,D50)
        if ye is None: continue
        red.append(100*(y0-ye)/y0); Yend.append(ye); Yb.append(y0)
    red=np.array(red); Yend=np.array(Yend); Yb=np.array(Yb)
    return Yb,red,Yend
print("\n\n==== POPULATION response RATES vs baseline severity (FLX 40mg, N=4000, baseline ~ N(26,6)) ====")
Yb,red,Yend=rates(40,2.7,26,6)
bins=[(10,18),(18,22),(22,26),(26,30),(30,40)]
print(f"{'baseline bin':>14} {'n':>5} {'resp>=25%':>9} {'resp>=35%':>9} {'remit Y<12':>11} {'mean%red':>9}")
for a,c in bins:
    m=(Yb>=a)&(Yb<c); n=int(m.sum())
    if n==0: continue
    print(f"  [{a:>2},{c:>2})     {n:>5} {100*np.mean(red[m]>=25):>8.0f}% {100*np.mean(red[m]>=35):>8.0f}% "
          f"{100*np.mean(Yend[m]<12):>10.0f}% {np.mean(red[m]):>8.1f}")
print(f"{'POOLED':>14} {len(Yb):>5} {100*np.mean(red>=25):>8.0f}% {100*np.mean(red>=35):>8.0f}% "
      f"{100*np.mean(Yend<12):>10.0f}% {np.mean(red):>8.1f}")

# ================= WITH severity-independent expectation eps(t)=4(1-e^{-t/5.5}) =================
def endpoint_Gonly(Y0, dose, D50, weeks=12.0, dt=0.05):
    a,G0,ok=GP.alpha_for(Y0,THM)
    if not ok: return None
    tgrid=np.arange(0,weeks+dt/2,dt); eC=C.ec_series(B.const(dose),D50,tgrid); G=G0
    for i in range(len(tgrid)):
        G+=dt*(a*B.Sshape(G,THM,PS)-(1.0+BB*(eC[i]-eC_h))*G)/TAUG; G=min(max(G,0.01),3.2)
    return M.Yread(G)
EPS12=4.0*(1-np.exp(-12/5.5))   # expectation at 12 wk, additive, severity-independent
print(f"\n\n==== DRUG-ARM vs PLACEBO-ARM response (with expectation eps(12)={EPS12:.2f} pts), FLX 40mg ====")
print("placebo arm: Y = Y0 - eps ;  drug arm: Y = Ymech - eps ; response = %red>=thr")
rng=np.random.default_rng(2); Ybs=np.clip(rng.normal(26,6,6000),10,39.5)
rows=[]
for y0 in Ybs:
    ym=endpoint_Gonly(y0,40,2.7)
    if ym is None: continue
    Yd=max(0,ym-EPS12); Yp=max(0,y0-EPS12)
    rows.append((y0, 100*(y0-Yd)/y0, 100*(y0-Yp)/y0, Yd))
rows=np.array(rows); Yb=rows[:,0]; rd=rows[:,1]; rp=rows[:,2]; Ydend=rows[:,3]
bins=[(10,18),(18,22),(22,26),(26,30),(30,40)]
for thr in (25,35):
    print(f"\n  --- response threshold >= {thr}% ---")
    print(f"  {'bin':>10} {'n':>5} {'DRUG resp':>9} {'PBO resp':>9} {'CONTRAST':>9} {'remit(Y<12,drug)':>16}")
    for a,c in bins:
        m=(Yb>=a)&(Yb<c); n=int(m.sum())
        if not n: continue
        dr=100*np.mean(rd[m]>=thr); pr=100*np.mean(rp[m]>=thr)
        print(f"  [{a:>2},{c:>2})   {n:>5} {dr:>8.0f}% {pr:>8.0f}% {dr-pr:>+8.0f} {100*np.mean(Ydend[m]<12):>15.0f}%")
    dr=100*np.mean(rd>=thr); pr=100*np.mean(rp>=thr)
    print(f"  {'POOLED':>10} {len(Yb):>5} {dr:>8.0f}% {pr:>8.0f}% {dr-pr:>+8.0f}")
