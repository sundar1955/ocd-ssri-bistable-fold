"""ISSUE #2 robustness (backs option C): is the bistable fold an artifact of the phi_e prefactor,
and is health (G_CS=0) stable for ALL theta_M? Frozen params. FULL source vs phi_e->const."""
import numpy as np, json, sys
import build_fit_gcs_slow as M, gamma_profile_calib as GP, gcs_bistable_population as B
def P(*a): print(*a); sys.stdout.flush()
bf=json.load(open('best_fit.json'))
THM=bf['theta_M']; BB=bf['b']; KD=bf['kappa_des']; HMAX=bf['h_max']; PS=GP.PHI_SAT
M.b_5HT1B=BB; M.setup(KD,HMAX); M.GAMMA_OBS=bf['gamma']
eC_h=M.eC_h; PHI_H=B.PHI_H
PHE_C=B.phe(M.G_of_Y(24.0))
def sink(eC): return 1.0+BB*(eC-eC_h)   # normalized decay: coeff=1 at baseline eC=eC_h
P(f"THM={THM} b={BB} phi_sat={PS} eC_h={eC_h} PHI_H={PHI_H:.3f} PHE_C(const)={PHE_C:.3f}")

def Sfull(G,thM):  p=B.phd1(G); return B.phe(G)*(p-PHI_H)*(p-thM)*(PS-p)
def Snophe(G,thM): p=B.phd1(G); return PHE_C*(p-PHI_H)*(p-thM)*(PS-p)

# precompute eC(u) ONCE on a u-grid
US=np.linspace(0,0.98,50)
def eC_of_u(u):
    if u<=1e-6: return eC_h
    D50=10.; D=u*D50/(1-u); tg=np.arange(0,80,0.4); return float(GP.C.ec_series(lambda t:D,D50,tg)[-1])
ECU=np.array([eC_of_u(u) for u in US]); P("eC(u) grid done")

Gg=np.linspace(1e-4,3.0,3000)
def stable_ocd(alpha,Sfun,thM,eC):
    f=alpha*np.array([Sfun(g,thM) for g in Gg])-sink(eC)*Gg
    s=np.where(np.diff(np.sign(f))!=0)[0]
    for i in s:
        g=Gg[i];
        if g>0.15 and (alpha*Sfun(g+1e-4,thM)-sink(eC)*(g+1e-4))<(alpha*Sfun(g-1e-4,thM)-sink(eC)*(g-1e-4)):
            return True
    return False
def analyze(Y0,Sfun,thM=THM):
    G0=M.G_of_Y(Y0); Su=Sfun(G0,thM)
    if Su<=0: return None,None
    alpha=sink(eC_h)*G0/Su
    bist=stable_ocd(alpha,Sfun,thM,eC_h)
    u_fold=None
    for u,eC in zip(US,ECU):
        if not stable_ocd(alpha,Sfun,thM,eC): u_fold=u; break
    return bist,u_fold

P("\n=== TEST 1: does the bistable fold survive dropping the phi_e prefactor? ===")
P(f"{'Y0':>4} | {'FULL bist':>9} {'u_fold':>7} | {'no-phi_e bist':>12} {'u_fold':>7}")
for Y0 in [18,22,26,30,34]:
    ab,au=analyze(Y0,Sfull); bb,bu=analyze(Y0,Snophe)
    P(f"{Y0:>4} | {str(ab):>9} {(f'{au:.2f}' if au is not None else 'none'):>7} | {str(bb):>12} {(f'{bu:.2f}' if bu is not None else 'none'):>7}")

P("\n=== TEST 2: is health (G_CS=0) stable for ALL theta_M? (RHS slope at G=0; <0 = stable) ===")
G0=M.G_of_Y(24.0)
P(f"{'theta_M':>8} | {'FULL slope@0':>13} {'stable':>6} | {'no-phi_e slope@0':>16} {'stable':>6}")
for thM in [6,8,9.15,11,14,18,22]:
    o=[]
    for Sfun in (Sfull,Snophe):
        Su=Sfun(G0,thM); alpha=sink(eC_h)*G0/Su if Su>0 else float('nan')
        eps=1e-5; o.append((alpha*Sfun(eps,thM)-sink(eC_h)*eps)/eps)
    P(f"{thM:>8} | {o[0]:>13.4f} {str(o[0]<0):>6} | {o[1]:>16.4f} {str(o[1]<0):>6}")
