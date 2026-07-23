"""Reviewer-requested rigor checks on the reduced 1-D G_CS plasticity dynamics.
(F-ii) SADDLE-NODE NORMAL-FORM CERTIFICATION at the calibrated fold (Y-BOCS0=20):
       locate (u_fold, G_fold) where f=0 and f_G=0; verify non-degeneracy f_GG != 0 and
       transversality f_u != 0; confirm the lambda ~ (u_fold-u)^{1/2} critical-slowing law.
(F-i)  SLOW-MANIFOLD / QSS CHECK: the drug enters via extracellular serotonin e_C, whose
       autoreceptor desensitization has tau_des ~ 2 wk while G_CS remodels over tau_G ~ 12 wk
       (ratio ~1/6, not asymptotically separated). We integrate G_CS with the TIME-RESOLVED
       e_C(t) (full 2-timescale system) vs the reduced system that uses the desensitized
       steady e_C(u), across baselines, and show the fold structure and endpoints are
       preserved (max endpoint deviation reported)."""
import numpy as np, json
import build_fit_gcs_slow as M, gcs_bistable_population as B
import gcs_bistable_calibrate as C, gamma_profile_calib as GP

bf=json.load(open('best_fit.json'))
THM=9.15; HMAX=2.0; BB=bf['b']; KD=bf['kappa_des']; PS=GP.PHI_SAT; eC_h=M.eC_h; TAUG=bf['tau_G']
M.GAMMA_OBS=2.0; M.b_5HT1B=BB; M.setup(KD,HMAX); GP.KAPPA_DES=KD

_uc={}
def eC_ss(u):
    if u<=1e-6: return eC_h
    if u in _uc: return _uc[u]
    D50=10.; D=u*D50/(1-u); tg=np.arange(0,80,0.2); v=float(C.ec_series(lambda t:D,D50,tg)[-1]); _uc[u]=v; return v
def cdrug(u): return 1.0+BB*(eC_ss(u)-eC_h)
def f(G,al,u): return al*B.Sshape(G,THM,PS)-cdrug(u)*G           # tau_G * dG/dt  (the RHS)
def fG(G,al,u,h=1e-5): return (f(G+h,al,u)-f(G-h,al,u))/(2*h)     # d/dG
def fGG(G,al,u,h=1e-4): return (f(G+h,al,u)-2*f(G,al,u)+f(G-h,al,u))/h**2
def fu(G,al,u,h=1e-4): return (f(G,al,u+h)-f(G,al,u-h))/(2*h)     # d/du

Y0=20.0; al,G0,ok=GP.alpha_for(Y0,THM)
print(f"=== (F-ii) Saddle-node normal-form certification, Y-BOCS0={Y0} (alpha={al:.5g}) ===")

# locate the fold: the u at which the OCD-attractor & saddle merge (f=0, f_G=0 simultaneously).
# scan u upward; at each u find the two upper roots (attractor>saddle); fold = where they coincide.
_GG=np.linspace(1e-4,2.9,7000)
def upper_pair(u):
    fv=al*np.array([B.Sshape(g,THM,PS) for g in _GG])-cdrug(u)*_GG; idx=np.where(np.diff(np.sign(fv))!=0)[0]
    roots=[_GG[i]-fv[i]*(_GG[i+1]-_GG[i])/(fv[i+1]-fv[i]) for i in idx]
    hi=[r for r in roots if r>0.25]
    return (max(hi),min(hi)) if len(hi)>=2 else None
lo,hi=0.0,0.999
for _ in range(60):
    m=0.5*(lo+hi); p=upper_pair(m)
    if p and (p[0]-p[1])>1e-4: lo=m
    else: hi=m
u_fold=0.5*(lo+hi)
pr=upper_pair(lo)                      # just below fold: attractor & saddle still distinct
G_fold=0.5*(pr[0]+pr[1]) if pr else float('nan')
# refine G_fold as the point where f_G=0 near the merge at u just below fold
gg=np.linspace(pr[1]-0.05,pr[0]+0.05,4000); i=np.argmin(np.abs([fG(g,al,lo) for g in gg])); G_fold=gg[i]

print(f"  fold located at  u_fold = {u_fold:.4f},  G_fold = {G_fold:.4f}")
print(f"  tangency:      f (G_fold,u_fold) = {f(G_fold,al,u_fold):+.3e}   (target 0)")
print(f"  tangency:      f_G(G_fold,u_fold) = {fG(G_fold,al,u_fold):+.3e}   (target 0)")
print(f"  non-degeneracy f_GG(G_fold,u_fold) = {fGG(G_fold,al,u_fold):+.3e}   (must be != 0)")
print(f"  transversality f_u (G_fold,u_fold) = {fu(G_fold,al,u_fold):+.3e}   (must be != 0)")
a_nf=0.5*fGG(G_fold,al,u_fold); b_nf=fu(G_fold,al,u_fold)
print(f"  => topological normal form  dx/dt ~ a*x^2 + b*(u-u_fold)/tau_G,  a={a_nf:.3e}, b={b_nf:.3e}")
# critical-slowing law: relaxation rate lambda(u) = -f_G(attractor)/tau_G ~ sqrt(u_fold-u)
print("  critical slowing  lambda(u) = -f_G/tau_G  vs  sqrt(u_fold-u):")
for u in [u_fold-0.20,u_fold-0.10,u_fold-0.05,u_fold-0.02,u_fold-0.005]:
    p=upper_pair(u); Ga=p[0]
    lam=-fG(Ga,al,u)/TAUG
    print(f"    u={u:.3f}  (u_fold-u)={u_fold-u:.3f}  lambda={lam:.4f}/wk  lambda/sqrt(du)={lam/np.sqrt(u_fold-u):.4f}")

print("\n=== (F-i) Slow-manifold / QSS check: time-resolved e_C(t) vs desensitized steady e_C(u) ===")
# reduced: constant e_C = e_C_ss(u).  full: e_C(t) from the serotonin engine (includes tau_des ramp).
def endpoint_reduced(al,G0,u,weeks,dt=0.02):
    G=G0; cd=cdrug(u)
    for _ in range(int(weeks/dt)):
        G+=dt*(al*B.Sshape(G,THM,PS)-cd*G)/TAUG; G=min(max(G,1e-3),3.4)
    return G
def endpoint_full(al,G0,u,weeks,dt=0.02):
    D50=10.; D=u*D50/(1-u) if u<1 else 1e6
    tg=np.arange(0,weeks+dt,dt); ec=C.ec_series(lambda t:D,D50,tg)   # time-resolved e_C(t)
    G=G0
    for i in range(len(tg)-1):
        cd=1.0+BB*(ec[i]-eC_h)
        G+=dt*(al*B.Sshape(G,THM,PS)-cd*G)/TAUG; G=min(max(G,1e-3),3.4)
    return G
maxdev=0.0
print("  Y0   u     G_reduced  G_full   |dY|(endpoint, 12wk)")
for Y0t in [18,20,22,24,26]:
    a2,G02,okk=GP.alpha_for(float(Y0t),THM)
    if not okk: continue
    for u in [0.5,0.8,0.95]:
        Gr=endpoint_reduced(a2,G02,u,12.); Gf=endpoint_full(a2,G02,u,12.)
        dY=abs(M.Yread(Gr)-M.Yread(Gf)); maxdev=max(maxdev,dY)
        print(f"  {Y0t:>2}  {u:.2f}   {Gr:7.4f}   {Gf:7.4f}    {dY:.3f}")
print(f"\n  max |endpoint Y-BOCS deviation| (full vs reduced, 12 wk) = {maxdev:.3f} pts")
# does the bistability threshold u_fold move under the full time-resolved e_C? Compare steady states.
def steady_full(al,G0,u): return endpoint_full(al,G0,u,400.)
def folds_reduced(al,u):
    p=upper_pair_al(al,u); return p is None
def upper_pair_al(al,u):
    fv=al*np.array([B.Sshape(g,THM,PS) for g in _GG])-cdrug(u)*_GG
    idx=np.where(np.diff(np.sign(fv))!=0)[0]
    roots=[_GG[i]-fv[i]*(_GG[i+1]-_GG[i])/(fv[i+1]-fv[i]) for i in idx]
    hi=[r for r in roots if r>0.25]; return (max(hi),min(hi)) if len(hi)>=2 else None
# u_fold from long-time full integration (does the OCD attractor survive?) for Y0=20
us=np.linspace(0.5,0.99,50); uf_full=np.nan
for u in us:
    Gs=steady_full(al,G0,u)
    if M.Yread(Gs)<12:   # collapsed to health
        uf_full=u; break
print(f"  u_fold (reduced, Y0=20) = {u_fold:.3f};  u_fold (full time-resolved e_C, Y0=20) = {uf_full:.3f}")
json.dump(dict(u_fold=float(u_fold),G_fold=float(G_fold),
               f=float(f(G_fold,al,u_fold)),fG=float(fG(G_fold,al,u_fold)),
               fGG=float(fGG(G_fold,al,u_fold)),fu=float(fu(G_fold,al,u_fold)),
               maxdev_full_vs_reduced=float(maxdev),u_fold_full=float(uf_full)),
          open('fold_certification.json','w'),indent=1)
print("\nwrote fold_certification.json")
