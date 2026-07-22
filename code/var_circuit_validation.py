"""
van Albada & Robinson 2009 (J Theor Biol 257:642-663, PMID 19168074) BGTCS mean-field.
STEP 1: encode the full 9-population circuit and reproduce the healthy fixed point (their Table 3a).

9 populations: e (cortical excit), i (cortical inhib), d1, d2, p1 (GPi/SNr), p2 (GPe),
               c (STN, = their 'varsigma'), s (thalamic relay), r (TRN).
Fixed point: phi_a = S_a( sum_b nu_ab phi_b + ext_a ),  ext only to relay (brainstem drive).
Sigmoid (their Eq 2): S_a(V) = Qmax_a / (1 + exp(-(V - theta_a)/sigma)).
All nu in mV*s (count x efficacy); excitatory +, inhibitory (GABA) -.
"""
import numpy as np
from scipy.optimize import fsolve

# population index
P = ['e','i','d1','d2','p1','p2','c','s','r']
idx = {p:k for k,p in enumerate(P)}
N = 9

# --- Table 2 parameters ---
sigma = 3.8  # mV threshold spread (their sigma')
Qmax = {'e':300,'i':300,'d1':65,'d2':65,'p1':250,'p2':300,'c':500,'s':300,'r':500}
theta = {'e':14,'i':14,'d1':19,'d2':19,'p1':10,'p2':9,'c':10,'s':13,'r':13}  # mV
phi_n = 10.0   # brainstem/sensory stimulus level (s^-1) used for Table 3a
nu_sn = 0.5    # brainstem -> relay

# --- connection strengths nu_ab (from population b -> a), mV*s, SIGNED ---
# built as dict of (target, source): value
nu = {
 ('e','e'):1.6, ('e','i'):-1.9, ('e','s'):0.4,
 ('i','e'):1.6, ('i','i'):-1.9, ('i','s'):0.4,
 ('d1','e'):1.0, ('d1','d1'):-0.3, ('d1','s'):0.1,
 ('d2','e'):0.7, ('d2','d2'):-0.3, ('d2','s'):0.05,
 ('p1','d1'):-0.1, ('p1','p2'):-0.03, ('p1','c'):0.3,
 ('p2','d2'):-0.3, ('p2','p2'):-0.1, ('p2','c'):0.3,
 ('c','e'):0.1, ('c','p2'):-0.04,
 ('s','e'):0.8, ('s','p1'):-0.03, ('s','r'):-0.4,
 ('r','e'):0.15, ('r','s'):0.03,
}
# matrix form
M = np.zeros((N,N))
for (a,b),v in nu.items():
    M[idx[a], idx[b]] = v
ext = np.zeros(N)
ext[idx['s']] = nu_sn*phi_n   # constant external drive to relay

Qmax_v = np.array([Qmax[p] for p in P])
theta_v = np.array([theta[p] for p in P])

def S(V):
    V = np.clip(V, -1e3, 1e3)
    return Qmax_v/(1.0+np.exp(-(V-theta_v)/sigma))

def Sprime(V):
    V = np.clip(V, -1e3, 1e3)
    e = np.exp(-(V-theta_v)/sigma)
    return Qmax_v*(e/sigma)/(1.0+e)**2

def fp_res(phi, Mmat=M, extv=ext):
    V = Mmat@phi + extv
    return phi - S(V)

def jac_relax(phi, Mmat=M, extv=ext):
    """Relaxation-stability proxy: J = -I + diag(S') M  (ignores delays/dendritic filter = Paper II)."""
    V = Mmat@phi + extv
    return -np.eye(N) + (Sprime(V)[:,None])*Mmat

def find_fps(Mmat=M, extv=ext, n_starts=400, seed=0):
    rng = np.random.default_rng(seed)
    fps=[]
    # include a start near the expected physiological point + random broad starts
    guesses=[np.array([12,12,7.4,3.5,69,48,28,14,28.],float)]
    for _ in range(n_starts):
        guesses.append(rng.uniform(0, [300,300,65,65,250,300,500,300,500], N))
    for g in guesses:
        sol,info,ier,msg = fsolve(fp_res, g, args=(Mmat,extv), full_output=True, xtol=1e-11)
        if ier==1 and np.max(np.abs(fp_res(sol,Mmat,extv)))<1e-6 and np.all(sol>=-1e-6):
            if not any(np.linalg.norm(sol-f)<1e-3 for f in fps):
                fps.append(sol)
    return fps

if __name__=='__main__':
    target = {'e':12,'i':12,'d1':7.4,'d2':3.5,'p1':69,'p2':48,'c':28,'s':14,'r':28}
    fps = find_fps()
    print(f"Found {len(fps)} fixed point(s).\n")
    # sort by cortical rate
    fps = sorted(fps, key=lambda x: x[idx['e']])
    for k,phi in enumerate(fps):
        ev = np.linalg.eigvals(jac_relax(phi))
        stable = np.all(ev.real<0)
        print(f"--- FP #{k+1}  (relaxation-stable={stable}) ---")
        print(f"{'pop':>4} {'model':>8} {'Table3a':>8} {'diff':>7}")
        for p in P:
            m=phi[idx[p]]; t=target[p]
            print(f"{p:>4} {m:8.2f} {t:8.1f} {m-t:7.2f}")
        print()
    # identify the physiological (low-rate, stable) one and RMS vs Table 3a
    tvec=np.array([target[p] for p in P])
    best=None; bestrms=1e9
    for phi in fps:
        ev=np.linalg.eigvals(jac_relax(phi))
        if np.all(ev.real<0):
            rms=np.sqrt(np.mean((phi-tvec)**2))
            if rms<bestrms: bestrms=rms; best=phi
    if best is not None:
        print(f"PHYSIOLOGICAL (stable) FP matches Table 3a with RMS = {bestrms:.3f} s^-1")
        # per-population % error
        print("max abs %% error:", np.max(np.abs((best-tvec)/tvec))*100)
    else:
        print("No stable FP found (check encoding).")
