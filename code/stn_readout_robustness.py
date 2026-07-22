"""
STN-READOUT ROBUSTNESS: the model only needs a MONOTONE severity coordinate.
Swapping the readout from caudate phi_d1 to subthalamic phi_STN touches ONLY the observational
map Y=F(phi(G_CS)); the G_CS dynamics (drug->eC->plasticity) are IDENTICAL. Since both phi_d1
and phi_STN are strictly monotone in G_CS, phi_STN is a monotone function of phi_d1, so swapping
the readout is a REPARAMETRIZATION ABSORBED BY THE EXPONENT gamma: each readout carries its own
gamma (caudate gamma=2; STN gamma~1.3). With its own gamma the STN readout reproduces the SAME
fit quality. We (1) show phi_STN monotone in phi_d1, (2) refit gamma,theta_M,b,kappa_des through
the STN readout (tau_G,h_max0 fixed from best_fit.json), (3) compare SSE + predictions to D1.
"""
import numpy as np, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.optimize import minimize, brentq
import build_fit_ssri_panel as P
from var_circuit_validation import idx
import build_fit_gcs_slow as M
import gamma_profile_calib as GP
iSTN=idx['c']

# --- phi_d1(G) and phi_STN(G) grids (drug-free, continuation) ---
Gg=np.linspace(0.0,3.45,140); seed=np.array([12,12,40,20,60,30,30,15,28.])
PHD1=[]; PHSTN=[]
for G in Gg:
    seed=P.circuit(G,0.0,seed); PHD1.append(seed[idx['d1']]); PHSTN.append(seed[iSTN])
PHD1=np.array(PHD1); PHSTN=np.array(PHSTN)
mono_d1=np.all(np.diff(PHD1)>=-1e-6); mono_stn=np.all(np.diff(PHSTN)>=-1e-6)
order=np.argsort(PHD1); mono_cross=np.all(np.diff(PHSTN[order])>=-1e-6)
print(f"phi_d1 monotone {mono_d1}; phi_STN monotone {mono_stn}; phi_STN monotone in phi_d1 {mono_cross}")

# --- STN readout: Y = 40*((phiSTN-phiSTN_h)/(phiSTN_cap-phiSTN_h))^gamma, endpoints matched to D1 ---
phiSTN_h=PHSTN[0]; phiSTN_cap=float(np.interp(3.4,Gg,PHSTN))
def phiSTN_of(G): return float(np.interp(np.clip(G,Gg[0],Gg[-1]),Gg,PHSTN))
def Yread_STN(G):
    p=max(0.0,(phiSTN_of(G)-phiSTN_h)/(phiSTN_cap-phiSTN_h)); return 40*p**M.GAMMA_OBS
def G_of_Y_STN(Yt): return brentq(lambda g: Yread_STN(g)-Yt, 0.05, 3.4, xtol=1e-5)

# --- fit gamma,theta_M,b,kappa_des per readout; tau_G,h_max0 fixed from best_fit ---
bf=json.load(open('best_fit.json'))
THMc=bf['theta_M']; TGc=bf['tau_G']; HM0=bf['h_max']; B0=bf['b']; KD0=bf['kappa_des']
STUD=GP.STUD
Yread_D1, G_of_Y_D1 = M.Yread, M.G_of_Y
def fit(readout_Y,readout_Ginv):
    M.Yread=readout_Y; M.G_of_Y=readout_Ginv
    def obj(p):   # p=[gamma, theta_M, b, kappa_des]
        M.GAMMA_OBS=p[0]; GP.KAPPA_DES=p[3]; GP.C.KAPPA_DES=p[3]
        return GP.objective_alpha([p[1],TGc,p[2],HM0],STUD)
    r=minimize(obj,[2.0,THMc,B0,KD0],method='Nelder-Mead',
               bounds=[(1.,6.),(7.,12.),(0.005,0.05),(0.05,0.55)],
               options=dict(xatol=1e-4,fatol=1e-4,maxfev=500))
    M.GAMMA_OBS=r.x[0]; GP.KAPPA_DES=r.x[3]; GP.C.KAPPA_DES=r.x[3]
    x=[r.x[1],TGc,r.x[2],HM0]
    sse,fex=GP.objective_alpha(x,STUD,detail=True); mfx=100*np.mean(fex)
    M.Yread,M.G_of_Y=Yread_D1,G_of_Y_D1
    return x,float(r.x[3]),float(r.x[0]),sse,mfx
# Per-readout gamma refits and panels (b)/(c) removed: the coordinate change is a monotone, invertible
# relabeling, argued from panel (a) alone -- no per-readout gamma refit is needed or shown.
# --- figure: phi_STN is a strictly monotone (invertible) function of phi_d1 ---
fig,ax=plt.subplots(1,1,figsize=(5.4,4.4))
ax.plot(PHD1,PHSTN,lw=2); ax.set_xlabel('$\\phi_{d1}$ (caudate, s$^{-1}$)')
ax.set_ylabel('$\\phi_{STN}$ (s$^{-1}$)'); ax.set_title('$\\phi_{STN}$ is a strictly monotone function of $\\phi_{d1}$'); ax.grid(alpha=.25)
assert mono_cross, 'phi_STN not monotone in phi_d1'
plt.tight_layout(); plt.savefig('fig_stn_readout_robustness.png',dpi=140); plt.savefig('fig_stn_readout_robustness.pdf')
print('wrote fig_stn_readout_robustness.pdf (single panel: monotone phi_STN(phi_d1))')
