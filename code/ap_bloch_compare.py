"""SI: bridge AP PK (dose->D2 occ->Black-Leff eff.blockade) to the current fold model, and compare against
Bloch 2006 (refractory OCD on stable SRI: response 35% Y-BOCS reduction; placebo 4%, pooled AP 32%, risperidone 50%)."""
import numpy as np, json, sys
from scipy.optimize import fsolve
sys.path.insert(0,'/Users/sundar/Antigravity Projects/OCD-MDD/Code/Paper2_8D')
import px_ap_pk_model as PK
from var_circuit_validation import idx, S, M, ext
import build_fit_gcs_slow as Mod, gamma_profile_calib as GP, gcs_bistable_population as B
bf=json.load(open('best_fit.json'))
THM=9.15;BB=bf['b'];PS=GP.PHI_SAT;eC_h=Mod.eC_h;phih=Mod.phid1_h
Mod.GAMMA_OBS=2.0; Mod.b_5HT1B=BB; Mod.setup(bf['kappa_des'],2.0); GP.KAPPA_DES=bf['kappa_des']
ie=idx['e']; id1=idx['d1']; id2=idx['d2']; RHO=0.7
def M_g(g): Mm=M.copy(); Mm[id1,ie]+=g; Mm[id2,ie]+=RHO*g; return Mm
Gg=np.linspace(0.02,3.2,55); aAPg=np.linspace(0,26,36); PHE=np.zeros((55,36)); PHD1=np.zeros((55,36))
for i,G in enumerate(Gg):
    seed=np.array([12,12,40,20,60,30,30,15,28.]); Mm=M_g(G)
    for j,a in enumerate(aAPg):
        def res(phi): V=(Mm@phi+ext).copy(); V[id2]+=a; return phi-S(V)
        seed=fsolve(res,seed,xtol=1e-10); PHE[i,j]=seed[ie]; PHD1[i,j]=seed[id1]
def S_AP(G,a):
    i=np.argmin(abs(Gg-G)); pe=np.interp(a,aAPg,PHE[i]); pd=np.interp(a,aAPg,PHD1[i]); return pe*(pd-phih)*(pd-THM)*(PS-pd)
def has_ocd(al,a,ec):
    f=np.array([al*S_AP(g,a)-(1.0+BB*(ec-eC_h))*g for g in Gg])
    for k in np.where(np.diff(np.sign(f))!=0)[0]:
        g=Gg[k]-f[k]*(Gg[k+1]-Gg[k])/(f[k+1]-f[k])
        if g>0.25 and (al*S_AP(g+1e-2,a)-(1.0+BB*(ec-eC_h))*(g+1e-2))<(al*S_AP(g-1e-2,a)-(1.0+BB*(ec-eC_h))*(g-1e-2)): return True
    return False
# --- PK: effective blockade at clinical augmentation doses ---
print("PK (dose -> D2 occ -> Black-Leff eff.blockade -> alpha_AP=0.222*eff):")
drugs=[('risperidone',2.0),('aripiprazole',10.0),('haloperidol',3.0)]
effs={}
for d,dose in drugs:
    occ=PK.d2_occupancy_hill(dose,d); tau=PK.DRUG_PARAMS[d]['tau_operational']
    eff=PK.partial_agonist_effective_blockade(occ,tau); effs[d]=eff
    print(f"  {d} {dose}mg: occ={occ*100:.0f}%  tau={tau}  eff.block={eff*100:.0f}%  alpha_AP={0.222*eff:.3f}")
# --- population + bridge aAP = C*eff.block, calibrated so pooled-AP -> 32% ---
rng=np.random.default_rng(0); pop=np.clip(rng.normal(26,5,3000),16,40)
def resp(aAP,ec):  # response = attractor folded (remission)
    out=0
    for Y0 in pop:
        try: al,_,_=GP.alpha_for(Y0,THM)
        except: continue
        if not has_ocd(al,aAP,ec): out+=1
    return 100*out/len(pop)
eC_ref=eC_h  # refractory: stable SRI ineffective (attractor unfolded), like Modarresi's flat comparator
print(f"\nplacebo arm (SSRI+placebo, aAP=0): model response = {resp(0.0,eC_ref):.0f}%  (Bloch 4%)")
# pooled-AP eff.block ~ mean of risperidone-like; calibrate C so pooled response ~32%
eff_pooled=0.50  # typical augmentation eff.blockade
for C in [6,8,10,12,14,16]:
    r=resp(C*eff_pooled,eC_ref)
    print(f"  bridge C={C}: aAP={C*eff_pooled:.1f} -> pooled-AP response {r:.0f}%")

print("\n--- calibrated bridge C=8 (aAP = 8 x eff.blockade); drug-specific vs Bloch ---")
C=8.0
bloch={'risperidone':50,'aripiprazole':None,'haloperidol':50}
for d,dose in drugs:
    aAP=C*effs[d]; r=resp(aAP,eC_ref)
    bl=bloch[d]; bls=f"Bloch {bl}%" if bl else "not in Bloch 2006"
    print(f"  {d} {dose}mg: eff={effs[d]*100:.0f}% -> aAP={aAP:.1f} -> model {r:.0f}%   ({bls})")
# map fold-threshold aAP for median refractory patient (Y0=26) back to occupancy
def foldaAP(Y0):
    al,_,_=GP.alpha_for(Y0,THM); lo,hi=0.0,26.
    if not has_ocd(al,lo,eC_ref): return 0.0
    for _ in range(26):
        mid=.5*(lo+hi)
        if not has_ocd(al,mid,eC_ref): hi=mid
        else: lo=mid
    return hi
for Y0 in [24,26,28,30]:
    a=foldaAP(Y0); occ=(a/C)  # eff.blockade; full antagonist occ=eff
    print(f"  Y0={Y0}: fold needs aAP={a:.1f} -> eff.block={100*a/C:.0f}% -> D2 occ~{100*a/C:.0f}% (full-antag)")
