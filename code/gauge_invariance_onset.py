"""ISSUE #4 (option B): in the NORMALIZED plasticity form the constitutive decay is fixed to 1
(baseline sink = 1 at eC=eC_h), which fixes the old (tau_G,kappa_h,b) scaling gauge. What remains
is that tau_G is a pure TIME-RESCALING of the AUTONOMOUS dynamics: (i) the t->inf attractor (hence
the response/remission classification) is INDEPENDENT of tau_G, and (ii) the untreated prodrome
(constant eC=eC_h, no external clock) rescales LINEARLY with tau_G. The SSRI onset is NOT a pure
tau_G-rescaling because the drug forcing eC(t) carries its own pharmacological clock (ramp + tau_des);
but its t->inf treated attractor is still tau_G-invariant. So tau_G sets only the disease clock,
fixed by the observed weeks-scale onset, NOT by the fitted outcome -- a round tau_G=12 wk is fine.
Frozen params. Normalized sink: (1+b(eC-eC_h))G."""
import numpy as np, json, sys
import build_fit_gcs_slow as M, gamma_profile_calib as GP, gcs_bistable_population as B
def P(*a): print(*a); sys.stdout.flush()
bf=json.load(open('best_fit.json'))
THM=bf['theta_M']; PS=GP.PHI_SAT; eC_h=M.eC_h; BB=bf['b']
M.b_5HT1B=BB; M.setup(bf['kappa_des'],bf['h_max']); M.GAMMA_OBS=bf['gamma']
def sink(eC): return 1.0+BB*(eC-eC_h)   # normalized constitutive decay: coeff=1 at baseline

def attractor_and_prodrome(tauG,label):
    Y0=22.0; a,G0,ok=GP.alpha_for(Y0,THM); dt=0.02
    # --- SSRI treated attractor: integrate long (>>tau_G and >> pharmacological clock) to the fixed point ---
    Tlong=max(120.0,40*tauG); tg=np.arange(0,Tlong,dt); eC=GP.C.ec_series(B.const(40.0),2.7,tg)
    G=G0
    for e in eC: G+=dt*(a*B.Sshape(G,THM,PS)-sink(e)*G)/tauG; G=min(max(G,0.01),3.2)
    Ystar_ssri=M.Yread(G)
    # --- untreated prodrome (autonomous, eC=eC_h): time from just-above-saddle to 90% of attractor G0 ---
    Gs=np.linspace(1e-4,min(3.0,G0*1.2),6000); f=a*np.array([B.Sshape(g,THM,PS) for g in Gs])-sink(eC_h)*Gs
    roots=[Gs[i]-f[i]*(Gs[i+1]-Gs[i])/(f[i+1]-f[i]) for i in np.where(np.diff(np.sign(f))!=0)[0]]
    saddle=[r for r in roots if 0.02<r<G0*0.98]
    tpro=np.nan
    if saddle:
        G=saddle[-1]*1.05; t=0.0
        while G<0.9*G0 and t<80*tauG:
            G+=dt*(a*B.Sshape(G,THM,PS)-sink(eC_h)*G)/tauG; t+=dt
        tpro=t
    P(f"{label:>16} | tau_G={tauG:6.2f} wk | SSRI attractor Y*={Ystar_ssri:7.3f} | untreated prodrome(saddle->90%)={tpro:7.2f} wk")
    return Ystar_ssri,tpro

P("=== tau_G is a pure time-gauge: t->inf attractor INVARIANT; autonomous prodrome rescales ~ tau_G ===")
Y0,p0=attractor_and_prodrome(bf['tau_G'],"frozen tau_G")
for c in (2.0,0.5):
    Yc,pc=attractor_and_prodrome(bf['tau_G']*c,f"tau_G x{c}")
    P(f"{'  vs frozen':>16} | dY*={Yc-Y0:+7.4f} (expect ~0: attractor invariant) | prodrome x{pc/p0:.2f} (expect x{c})")
P(f"\nnote: constitutive decay normalized to 1 (baseline sink=1 at eC_h={eC_h} nM); b={BB} nM^-1 sets")
P( "      the drug pull b*(eC-eC_h). The treated attractor Y* (hence response/remission class) is")
P( "      tau_G-invariant, and the AUTONOMOUS untreated prodrome rescales linearly with tau_G. SSRI")
P( "      clinical onset also carries the pharmacological clock (drug ramp + tau_des), so it is not a")
P( "      pure tau_G-rescaling; but the OUTCOME is. Hence tau_G is fixed by the weeks-scale onset, not")
P( "      by the outcome fit, and a round tau_G=12 wk is acceptable.")
