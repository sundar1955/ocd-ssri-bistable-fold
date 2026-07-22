"""ISSUE #3 (backs option C): what does the drug-attributable responder fraction actually depend on?
Show it is SENSITIVE to the assumed baseline population (mean, SD) but INSENSITIVE to theta_M.
Drug-attributable responder rate = (drug-arm >=X% response) - (placebo-arm >=X% response). Frozen params."""
import numpy as np, json, sys
import build_fit_gcs_slow as M, gamma_profile_calib as GP, gcs_bistable_population as B
def P(*a): print(*a); sys.stdout.flush()
bf=json.load(open('best_fit.json'))
THM0=bf['theta_M']; TAUG=bf['tau_G']; BB=bf['b']; KD=bf['kappa_des']; HMAX=bf['h_max']; PS=GP.PHI_SAT
M.b_5HT1B=BB; M.setup(KD,HMAX); M.GAMMA_OBS=bf['gamma']; eC_h=M.eC_h
EPS12=4.0*(1-np.exp(-12/5.5))     # severity-independent expectation, ~3.55 pts at 12 wk
DOSE,D50=40.0,2.7                 # representative adequate SSRI arm (FLX)
tg=np.arange(0,12.001,0.05); ECdrug=GP.C.ec_series(B.const(DOSE),D50,tg)

def endY(Y0,thM):
    a,G0,ok=GP.alpha_for(Y0,thM)
    if not ok: return None
    G=G0
    for e in ECdrug:
        G+=0.05*(a*B.Sshape(G,thM,PS)-(1.0+BB*(e-eC_h))*G)/TAUG; G=min(max(G,0.01),3.2)
    return M.Yread(G)

def responder_rate(mu,sd,thM,thr=35.,N=4000,seed=3):
    rng=np.random.default_rng(seed); Ybs=np.clip(rng.normal(mu,sd,N),10,39.5)
    dr=[]; pr=[]
    for y0 in Ybs:
        ym=endY(y0,thM)
        if ym is None: continue
        Yd=max(0,ym-EPS12); Yp=max(0,y0-EPS12)
        dr.append(100*(y0-Yd)/y0>=thr); pr.append(100*(y0-Yp)/y0>=thr)
    return 100*(np.mean(dr)-np.mean(pr))   # net drug-minus-placebo responder rate (%)

P(f"frozen: thM0={THM0} b={BB} tau_G={TAUG} dose=FLX{int(DOSE)} eps12={EPS12:.2f}\n")
P("=== A. responder fraction vs assumed baseline POPULATION (mean x SD), theta_M=canonical ===")
P(f"{'':>8}" + "".join(f"SD={s:>2}   " for s in [3,4,5,6]))
for mu in [24,26,28,30]:
    row=[responder_rate(mu,s,THM0) for s in [3,4,5,6]]
    P(f"mean={mu:>2} " + "".join(f"{r:>6.0f}% " for r in row))

P("\n=== B. responder fraction vs theta_M (population fixed at N(28,4)) ===")
P(f"{'theta_M':>8} {'net responder %':>15}")
for thM in [8,10,12,14,16,20,24]:
    P(f"{thM:>8} {responder_rate(28,4,thM):>14.0f}%")

P("\n=== C. same, at >=25% criterion (population N(28,4), theta_M canonical) vs mean ===")
for mu in [24,26,28,30]:
    P(f"  mean={mu:>2}: >=25% net={responder_rate(mu,4,THM0,thr=25):>3.0f}%   >=35% net={responder_rate(mu,4,THM0,thr=35):>3.0f}%")
