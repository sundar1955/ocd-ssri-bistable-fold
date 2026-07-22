"""AP fold requirement: AP adds drive aAP to D2-MSN -> lowers phe(G),phd1(G) -> shrinks the plasticity SOURCE
S_AP(G)=phe*(phd1-phih)(phd1-thM)(phisat-phd1). Find aAP that annihilates the OCD attractor, by severity.
Parallel to fold-eC (sink) and fold-alpha (memantine, gain)."""
import numpy as np, json
from scipy.optimize import fsolve
from var_circuit_validation import idx, S, M, ext
import build_fit_gcs_slow as Mod, gamma_profile_calib as GP, gcs_bistable_population as B
bf=json.load(open('best_fit.json'))
THM=9.15; BB=bf['b']; PS=GP.PHI_SAT; eC_h=Mod.eC_h; phih=Mod.phid1_h
Mod.GAMMA_OBS=2.0; Mod.b_5HT1B=BB; Mod.setup(bf['kappa_des'],2.0); GP.KAPPA_DES=bf['kappa_des']
ie=idx['e']; id1=idx['d1']; id2=idx['d2']; RHO=0.7
def M_g(g): Mm=M.copy(); Mm[id1,ie]+=g; Mm[id2,ie]+=RHO*g; return Mm
# precompute phe_AP, phd1_AP on (G, aAP) grid
Gg=np.linspace(0.02,3.2,60); aAPg=np.linspace(0,26,40)
PHE=np.zeros((60,40)); PHD1=np.zeros((60,40))
for i,G in enumerate(Gg):
    seed=np.array([12,12,40,20,60,30,30,15,28.]); Mm=M_g(G)
    for j,a in enumerate(aAPg):
        def res(phi): V=(Mm@phi+ext).copy(); V[id2]+=a; return phi-S(V)
        seed=fsolve(res,seed,xtol=1e-10); PHE[i,j]=seed[ie]; PHD1[i,j]=seed[id1]
def pheAP(G,a): return float(np.interp(a,aAPg,PHE[np.argmin(abs(Gg-G))]))
def phd1AP(G,a):return float(np.interp(a,aAPg,PHD1[np.argmin(abs(Gg-G))]))
# verify aAP=0 matches B.Sshape
print("verify S_AP(aAP=0) vs B.Sshape:", [ (round(pheAP(G,0)*(phd1AP(G,0)-phih)*(phd1AP(G,0)-THM)*(PS-phd1AP(G,0)),1), round(B.Sshape(G,THM,PS),1)) for G in [1.0,1.5,2.0]])
def S_AP(G,a):
    p=phd1AP(G,a); return pheAP(G,a)*(p-phih)*(p-THM)*(PS-p)
def has_ocd(al,a):
    f=np.array([al*S_AP(g,a)-1.0*g for g in Gg])
    for i in np.where(np.diff(np.sign(f))!=0)[0]:
        g=Gg[i]-f[i]*(Gg[i+1]-Gg[i])/(f[i+1]-f[i])
        if g>0.25 and (al*S_AP(g+1e-2,a)-1.0*(g+1e-2))<(al*S_AP(g-1e-2,a)-1.0*(g-1e-2)): return True
    return False
print(f"\n{'Y0':>3} {'foldaAP':>8} {'phd1@fold':>10} {'phd1_drug-free':>14}")
for Y0 in [24,26,28,30,32]:
    al,_,_=GP.alpha_for(Y0,THM)
    fa=next((a for a in np.linspace(0,26,120) if not has_ocd(al,a)),None)
    pdf=phd1AP(1.5,0)
    print(f"{Y0:>3} {(fa if fa else float('nan')):8.1f} {(phd1AP(1.5,fa) if fa else float('nan')):10.1f} {pdf:14.1f}")

# clean table expression: % reduction in caudate D1 firing (phi_d1) at the attractor that AP must produce to fold
print(f"\n{'Y0':>3} {'phd1_baseline':>13} {'phd1@fold-drive':>16} {'phi_d1 reduction':>16}")
for Y0 in [24,26,28,30]:
    al,_,_=GP.alpha_for(Y0,THM)
    fa=next((a for a in np.linspace(0,26,120) if not has_ocd(al,a)),None)
    G0=Mod.G_of_Y(Y0); pd0=phd1AP(G0,0); pdf=phd1AP(G0,fa)
    print(f"{Y0:>3} {pd0:13.1f} {pdf:16.1f} {100*(pd0-pdf)/pd0:15.0f}%")
