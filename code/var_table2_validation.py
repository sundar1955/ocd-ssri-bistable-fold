"""
VALIDATION of the vAR dispersion solver against Paper II Table 2 (PMID 19154745, p677).
Each Table-2 row: a single GAIN pushed past a threshold -> linear instability at a stated freq.
We work in GAIN space (vAR do too): G0_ab = rho_a * nu_ab at the healthy FP, then replace ONE
entry with the Table-2 threshold value and locate the rightmost dispersion root.
PASS = rightmost root has Re ~ 0 (on the stability boundary) and Im/2pi ~ table frequency.

Dispersion (Paper II eq 14):  det[ Diag_D(s) - K(s) ] = 0
  Diag_D(s) = diag(Dab, ..., Dab) with the e-entry multiplied by cortical wave (1+s/gam)^2 (k=0)
  Dab(s) = (1+s/alpha)(1+s/beta)   (dendritic filter, = 1/L)
  K(s)_ab = G_ab * exp(-s*tau_ab)  (gains carry the receiver slope rho_a; delays tau_ab)
"""
import numpy as np
from scipy.optimize import fsolve
from var_circuit_validation import idx, M as NU, ext, S, Sprime, N

alpha, beta, gam = 160., 640., 125.   # 1/s : dendrite decay, dendrite rise, cortical damping
ie = idx['e']

# ---- healthy fixed point + receiver slopes -> healthy gain matrix ----
seed = np.array([12,12,7.4,3.5,69,48,28,14,28.])
phi0 = fsolve(lambda p: p - S(NU@p+ext), seed, xtol=1e-12)
rho = Sprime(NU@phi0+ext)                 # receiver slope rho_a
G0 = rho[:,None]*NU                        # gains G_ab = rho_a nu_ab

# ---- axonal delays tau_ab (s) ----
T = np.zeros((N,N))
def sd(a,b,ms): T[idx[a],idx[b]] = ms/1000.
sd('e','s',35); sd('i','s',35); sd('s','e',50); sd('r','e',50)
sd('s','r',2);  sd('r','s',2);  sd('s','p1',3)
sd('d1','e',2); sd('d2','e',2); sd('d1','s',2); sd('d2','s',2)
sd('p1','d1',1); sd('p1','p2',1); sd('p1','c',1)
sd('p2','d2',1); sd('p2','p2',1); sd('p2','c',1); sd('c','e',1); sd('c','p2',1)

def dispersion(s, G):
    Dab = (1+s/alpha)*(1+s/beta)
    D = np.full(N, Dab, dtype=complex)
    D[ie] = Dab*(1+s/gam)**2               # cortical wave on the e ROW (receiver)
    K = G*np.exp(-s*T)
    return np.linalg.det(np.diag(D) - K)

def rightmost(G, smax=8.0, wmax=420.):
    """Find rightmost root of det=0 in s=sigma+i*omega."""
    sigs = np.linspace(-60, smax, 130)
    ws   = np.linspace(0., wmax, 260)
    Z = np.array([[np.log(abs(dispersion(sg+1j*w, G))+1e-300) for w in ws] for sg in sigs])
    roots=[]
    for i in range(1,len(sigs)-1):
        for j in range(1,len(ws)-1):
            if (Z[i,j]<=Z[i-1,j] and Z[i,j]<=Z[i+1,j] and
                Z[i,j]<=Z[i,j-1] and Z[i,j]<=Z[i,j+1]):
                def F(x):
                    v=dispersion(x[0]+1j*x[1], G); return [v.real, v.imag]
                try:
                    sol=fsolve(F,[sigs[i],ws[j]],xtol=1e-11)
                    if abs(dispersion(sol[0]+1j*sol[1],G))<1e-4:
                        roots.append(sol[0]+1j*abs(sol[1]))
                except Exception: pass
    if not roots: return None
    # dedupe
    uniq=[]
    for r in roots:
        if not any(abs(r-u)<1e-4 for u in uniq): uniq.append(r)
    return max(uniq, key=lambda r: r.real)

# ---- print a few healthy gains as a sanity check vs Figs 2-3 ----
print("healthy gains (sanity vs Figs 2-3):")
for (a,b) in [('e','e'),('e','i'),('e','s'),('s','e'),('p2','c'),('c','p2'),('r','e'),('r','s')]:
    print(f"  G_{a}{b} = {G0[idx[a],idx[b]]:+7.3f}")
print(f"healthy rightmost root: {rightmost(G0)}\n")

# ---- Table 2 test cases: (receiver, source, threshold gain, expected f Hz, note) ----
tests = [
 ('d1','e', -59.0,  5.8, 'cortico-D1 (theta)'),
 ('d1','s', -17.0, 20.0, 'thal-D1'),
 ('d2','e',  41.0,  5.0, 'cortico-D2'),
 ('c','e',   23.0,  6.1, 'hyperdirect cortico-STN'),
 ('p2','c',  21.0, 46.0, 'STN->GPe (gamma, STN-GPe loop)'),
 ('r','e',    4.1,  3.3, 'cortico-TRN (delta/theta)'),
 ('r','s',    1.8, 30.0, 'relay->TRN (intrathalamic)'),
]
print(f"{'gain':>7} {'target':>7} {'expect f':>9} | {'Re(s*)':>9} {'f=Im/2pi':>9}  {'note'}")
for a,b,gval,fexp,note in tests:
    G = G0.copy(); G[idx[a],idx[b]] = gval
    s = rightmost(G)
    if s is None:
        print(f"  G_{a}{b:>2} {gval:7.1f} {fexp:9.1f} |   (no root)   {note}")
        continue
    f = s.imag/(2*np.pi)
    print(f"  G_{a}{b:>2} {gval:7.1f} {fexp:9.1f} | {s.real:+9.3f} {f:9.2f}  {note}")
