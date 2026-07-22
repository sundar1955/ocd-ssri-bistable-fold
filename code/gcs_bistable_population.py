"""
BISTABLE cubic + POPULATION MIXTURE — forward simulation (proof of concept before calibration).

Slow ODE (per patient):
  tau_G dG/dt = alpha * S(G) - (1 + b*(eC(t)-eC_h)) * G
  S(G) = phe(G)*(phd1(G)-phd1_h)*(phd1(G)-theta_M)*(phi_sat-phd1(G))  [baseline-referenced BCM; zero at health]
STRUCTURAL (shared): theta_M, phi_sat, tau_G, b.  PATIENT-SPECIFIC: alpha (back-solved from baseline Y).
Readout Y=Yread(G); eC(t) from the fixed autoreceptor-on-release engine (build_fit_gcs_slow, eC_h=2.8, K=10).

Population: sample baseline Y ~ Normal(mean,SD) truncated; each patient's alpha placed so their OCD attractor sits
at their baseline Y; run SSRI; population MEAN Y(t) should plateau at partial improvement (responders cross the
remission fold -> ~healthy; non-responders stay on OCD branch).
"""
import numpy as np, sys
import build_fit_gcs_slow as M

# fast-circuit readouts on a G grid (phe, phd1) -- reuse module
Gg=np.linspace(0.02,3.2,321)
PHE=np.array([M.P.phe(g,0.) for g in Gg]); PHD1=np.array([M.P.phd1(g,0.) for g in Gg])
def phe(G):  return float(np.interp(np.clip(G,Gg[0],Gg[-1]),Gg,PHE))
def phd1(G): return float(np.interp(np.clip(G,Gg[0],Gg[-1]),Gg,PHD1))
PHI_H=M.phid1_h   # healthy caudate D1 rate (7.39); BASELINE-REFERENCED BCM source (2026-07-12)
def Sshape(G,thM,psat):
    # baseline-referenced BCM: source vanishes at the healthy set-point phi_d1^h (matches Model eq:plast)
    p=phd1(G); return phe(G)*(p-PHI_H)*(p-thM)*(psat-p)

def const(mg): return lambda t: mg
def titr(d1,d2,tsw): return lambda t: d1 if t<tsw else d2

def phisat_for(Ybase, thM, alpha, eC0):
    """PATIENT-SPECIFIC severity knob: back-solve phi_sat_i so the untreated OCD attractor sits at Ybase.
       Shared: alpha, theta_M. Returns (phi_sat_i, G_ocd, stable?, valid?)."""
    G=M.G_of_Y(Ybase); p=phd1(G)
    if p<=thM: return None,G,False,False          # below the saddle -> no OCD attractor
    sink=1.0*G
    # equilibrium: alpha*phe*p*(p-thM)*(phisat-p) = sink  ->  phisat = p + sink/(alpha*phe*p*(p-thM))
    phisat=p + sink/(alpha*phe(G)*p*(p-thM))
    if phisat>65.0: return phisat,G,False,False    # would need supra-max caudate ceiling
    dG=1e-3
    f=lambda g: alpha*Sshape(g,thM,phisat)-1.0*g
    stable=(f(G+dG)-f(G-dG))/(2*dG) < 0
    return phisat,G,stable,True

def patient_traj(Ybase, dose_fn, D50, thM, alpha, tau_G, kappa_des, tau_d, weeks):
    eC0=M.eC_qss(M.theta_h,0.,D50)                 # untreated eC (~eC_h)
    psat,Gocd,stable,valid=phisat_for(Ybase,thM,alpha,eC0)
    if not valid: return None
    dt=0.05; th=M.theta_h; G=Gocd; wk_set=set(np.round(weeks,3)); out={}
    for t in np.arange(0,weeks[-1]+dt/2,dt):
        rt=round(t,3)
        if rt in wk_set: out[rt]=M.Yread(G)
        e=M.eC_qss(th,dose_fn(t),D50,G)
        th+=dt*((1-th)-kappa_des*(e-M.eC_h)*th)/tau_d
        G +=dt*(alpha*Sshape(G,thM,psat)-(1.0+M.b_5HT1B*(e-M.eC_h))*G)/tau_G
        G  =min(max(G,0.01),3.2)
    return np.array([out[round(wk,3)] for wk in weeks]), Gocd, stable

def population(dose_fn, D50, thM, alpha, tau_G, weeks, Ymean=23.95, Ysd=5.5, N=300, seed=0,
              kappa_des=0.15, tau_d=2.0):
    rng=np.random.default_rng(seed)
    Ybs=np.clip(rng.normal(Ymean,Ysd,N), 10., 39.5)
    trajs=[]; Yb_kept=[]; n_excl=0
    for Yb in Ybs:
        r=patient_traj(Yb,dose_fn,D50,thM,alpha,tau_G,kappa_des,tau_d,weeks)
        if r is None: n_excl+=1; continue
        Yt,Gocd,stable=r
        trajs.append(Yt); Yb_kept.append(Yb)
    trajs=np.array(trajs); Yb_kept=np.array(Yb_kept)
    return trajs.mean(0), trajs, n_excl, Yb_kept

def response_rates(trajs, Yb_kept, resp_pct=35., remit_Y=12.):
    """Model-mechanistic (G_CS-driven, expectation-EXCLUDED) response/remission rates:
       response = >=resp_pct% Y-BOCS reduction baseline->endpoint; remission = endpoint Y < remit_Y.
       Reported POOLED and split by baseline severity (low-activity vs high-activity branch = below/above
       population median baseline). PREDICTION ONLY -- no data to validate the strata."""
    Yend=trajs[:,-1]; red=100.*(Yb_kept-Yend)/Yb_kept
    resp=red>=resp_pct; remit=Yend<remit_Y
    med=np.median(Yb_kept); lo=Yb_kept<med; hi=~lo
    def pct(m): return 100.*np.mean(m) if len(m) else float('nan')
    out={'pooled_resp':pct(resp),'pooled_remit':pct(remit),
         'low_resp':pct(resp[lo]),'low_remit':pct(remit[lo]),'low_Ymed':np.median(Yb_kept[lo]),
         'hi_resp':pct(resp[hi]),'hi_remit':pct(remit[hi]),'hi_Ymed':np.median(Yb_kept[hi])}
    # also fine-grained bins for later analysis
    bins=[(10,20),(20,24),(24,28),(28,40)]
    out['bins']=[(a,b,pct(resp[(Yb_kept>=a)&(Yb_kept<b)]),int(((Yb_kept>=a)&(Yb_kept<b)).sum())) for a,b in bins]
    return out

# Imported as a helper library (Sshape, phd1, phe, PHI_H, const, phisat_for, population) by
# gamma_profile_calib and the reviewer-response diagnostics. All functions use the normalized
# plasticity form: tau_G dG/dt = alpha*S(G) - (1 + b*(eC-eC_h))*G  (baseline sink coeff = 1, no kappa_h).
# The historical stand-alone proof-of-concept driver lives in local archive/.
