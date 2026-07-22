"""
STEP 2b: does an ASYMMETRIC increase of BOTH corticostriatal weights (dD1, dD2) produce a FOLD?
Literature supports driving BOTH pathways (cortex projects to D1 and D2: Kress 2013, Huerta-Ocampo 2014);
the D1/D2 asymmetry is the free knob. Scan a family of ratios rho = dD2/dD1 and look for hysteresis (fold).

For each rho: dD1 = g, dD2 = rho*g. Forward continuation (from healthy) + reverse (from top) over g in [0,6].
Fold/bistability <=> forward and reverse cortical branches disagree (hysteresis loop).
Also multi-start at each g to directly count coexisting stable fixed points.
"""
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from var_circuit_validation import (idx, S, jac_relax, fp_res, M, ext, find_fps)

def M_ratio(g, rho):
    Mm = M.copy()
    Mm[idx['d1'],idx['e']] += g
    Mm[idx['d2'],idx['e']] += rho*g
    return Mm

def _track(gs, rho, seed):
    """Follow ONE branch by natural continuation (previous solution as next seed)."""
    out={}; g=seed.copy()
    for gc in gs:
        Mm=M_ratio(gc,rho)
        sol=fsolve(fp_res,g,args=(Mm,ext),xtol=1e-11)
        if np.max(np.abs(fp_res(sol,Mm,ext)))<1e-6: g=sol
        ev=np.linalg.eigvals(jac_relax(sol,Mm,ext))
        out[round(gc,4)]=(sol[idx['e']], np.all(ev.real<0))
    return out

def continue_branch(rho, gmax=6.0, ng=601):
    """Forward from healthy PHYSIOLOGICAL FP; reverse seeded from the FORWARD ENDPOINT (same branch).
    This isolates the physiological branch and excludes the always-present high-saturated FPs (~280-300)."""
    gs = np.linspace(0, gmax, ng)
    phys0 = sorted(find_fps(M,ext),key=lambda x:x[idx['e']])[0]  # low physiological healthy FP
    fwd = _track(gs, rho, phys0)
    # reverse: rebuild the full state vector at gmax from forward, then track back
    Mtop=M_ratio(gmax,rho)
    top_full = fsolve(fp_res, sorted(find_fps(M,ext),key=lambda x:x[idx['e']])[0], args=(Mtop,ext))
    rev = _track(gs[::-1], rho, top_full)
    return gs, fwd, rev

def count_stable_fps(g, rho, n_starts=200, phys_cut=150.0):
    """Count PHYSIOLOGICAL stable FPs only (phi_e < phys_cut), excluding the always-present
    all-saturated state (~280-300 s^-1, every population near Qmax) which is not clinical."""
    Mm=M_ratio(g,rho)
    fps=find_fps(Mm,ext,n_starts=n_starts)
    ns=sum(1 for f in fps if f[idx['e']]<phys_cut and np.all(np.linalg.eigvals(jac_relax(f,Mm,ext)).real<0))
    return ns

if __name__=='__main__':
    rhos=[0.0,0.25,0.5,0.75,1.0]
    fig,ax=plt.subplots(figsize=(8,5.5))
    colors=plt.cm.viridis(np.linspace(0,0.85,len(rhos)))
    fold_found=False
    for rho,c in zip(rhos,colors):
        gs,fwd,rev=continue_branch(rho)
        ef=[fwd[round(g,4)][0] for g in gs]
        er=[rev[round(g,4)][0] for g in gs]
        hyst=max(abs(a-b) for a,b in zip(ef,er))
        ax.plot(gs,ef,'-',color=c,lw=1.6,label=f'rho={rho}  (fwd)')
        ax.plot(gs,er,'--',color=c,lw=1.0)
        # direct multistability check at a few g
        maxns=max(count_stable_fps(g,rho) for g in [1.5,3.0,4.5,6.0])
        tag='FOLD' if (hyst>1.0 and maxns>=2) else 'no fold'
        if hyst>1.0 and maxns>=2: fold_found=True
        print(f"rho={rho:4.2f}: phi_e healthy={ef[0]:.2f} fwd@6={ef[-1]:.2f} "
              f"max|fwd-rev|={hyst:.2f}  max#stableFP={maxns}  -> {tag}")
    ax.axhline(12,ls=':',color='gray',lw=0.8,label='healthy 12 s$^{-1}$')
    ax.set_xlabel(r'$\Delta\nu_{d_1e}$ (= $G_{CS}$ on D1; D2 gets $\rho\times$)')
    ax.set_ylabel(r'cortex $\phi_e$ (s$^{-1}$)')
    ax.set_title('Asymmetric corticostriatal lesion: cortex vs G_CS for D2/D1 ratios rho\n(solid=forward, dashed=reverse; separation=fold)')
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig('var_gcs_asymmetric.png',dpi=140)
    print(f"\nFOLD anywhere? {'YES' if fold_found else 'NO'}   saved var_gcs_asymmetric.png")
