"""
STAGE 1 PROBE for the graded-vs-fold head-to-head (R5 crux / WP1).
Confirm the CANONICAL fold model's fixed-point topology and fold behaviour before
building the matched graded twin. No new physics -- reuse fig4_phaseline construction.
"""
import numpy as np, json
import build_fit_gcs_slow as M
import gcs_bistable_population as B
import gcs_bistable_calibrate as C
import gamma_profile_calib as GP

bf=json.load(open('best_fit.json'))
THM=9.15; HMAX=2.0; BB=bf['b']; KD=bf['kappa_des']; PS=GP.PHI_SAT; eC_h=M.eC_h
M.GAMMA_OBS=2.0; M.b_5HT1B=BB; M.setup(KD,HMAX); GP.KAPPA_DES=KD
print(f"canonical fold: theta_M={THM} phi_sat={PS} b={BB} tau_G={bf['tau_G']} kappa_des={KD} gamma=2")
print(f"phid1_h={M.phid1_h:.3f}  Qd1={M.Qd1:.3f}  eC_h={eC_h}")

def eC_ss(u):
    if u<=1e-6: return eC_h
    D50=10.; D=u*D50/(1-u); tg=np.arange(0,80,0.2)
    return float(C.ec_series(lambda t:D,D50,tg)[-1])

def RHS(G,al,ec): return al*B.Sshape(G,THM,PS)-(1.0+BB*(ec-eC_h))*G
def fps(al,ec):
    Gs=np.linspace(0.0008,2.7,8000); f=np.array([RHS(g,al,ec) for g in Gs]); out=[]
    for i in np.where(np.diff(np.sign(f))!=0)[0]:
        g=Gs[i]-f[i]*(Gs[i+1]-Gs[i])/(f[i+1]-f[i])
        st=(RHS(g+1e-3,al,ec)-RHS(g-1e-3,al,ec))<0
        out.append((g,st))
    return out

# occupancy -> eC map
print("\nu -> eC_ss(u):")
for u in [0,0.2,0.4,0.6,0.8,0.9,0.95,0.98]:
    print(f"  u={u:.2f}  eC={eC_ss(u):.3f} nM   (drug sink coeff 1+b*(eC-eC_h)={1+BB*(eC_ss(u)-eC_h):.4f})")

# fixed-point topology + fold occupancy across baseline severity
print("\nbaseline Y0 -> fixed points (G, stable?) and fold occupancy u_fold:")
for Y0 in [16,18,20,22,24,26,28,30]:
    al,G0,ok=GP.alpha_for(Y0,THM)
    if al is None:
        print(f"  Y0={Y0}: UNPLACEABLE (below fold / ghost zone)"); continue
    # drug-free fixed points
    fp0=fps(al,eC_ss(0))
    # find fold occupancy: smallest u with no stable OCD attractor (g>0.25)
    ufold=None
    for u in np.linspace(0,0.985,600):
        if not [g for g,s in fps(al,eC_ss(u)) if s and g>0.25]:
            ufold=u; break
    # endpoint Y at max dose (u=0.95) integrated from OCD start
    Y_at095=integrate_endpoint(al,0.95) if False else None
    fpstr="; ".join(f"G={g:.3f}{'S' if s else 'u'}->Y={M.Yread(g):.1f}" for g,s in fp0)
    uf=f"{ufold:.3f}" if ufold is not None else ">0.985 (resistant)"
    print(f"  Y0={Y0}: alpha={al:.3e}  FPs[{fpstr}]  u_fold={uf}")
