"""S9 identifiability figure (normalized form, NO kappa_h): (a) marginal b-profile (kappa_des co-fit) at
tau_G=12 wk -> b well-constrained; (b) tau_G profile (refit b) -> RMS flat over [8,15] wk, worse outside;
pinned 12 wk. Model: tau_G dG/dt = alpha*S - (1 + b(eC-eC_h))*G."""
import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import build_fit_gcs_slow as M, gamma_profile_calib as GP, gcs_bistable_calibrate as C
STUD=GP.STUD; THM=9.15; HMAX=2.0
bf=json.load(open('best_fit.json')); B0=bf['b']; KD0=bf['kappa_des']; TG0=bf['tau_G']
def rms(b,kdes,tauG):
    GP.KAPPA_DES=kdes; C.KAPPA_DES=kdes; M.GAMMA_OBS=2.0; M.b_5HT1B=b; M.setup(kdes,HMAX)
    tmax=max(s['weeks'][-1] for s in STUD)+0.001; tg=np.arange(0,tmax,0.15); res=[]
    for st in STUD:
        ec=C.ec_series(st['dose_fn'],st['D50'],tg); nm,_=GP.net_model_alpha(st,THM,tauG,tg,ec)
        m=slice(1,None) if st['skip0'] else slice(None); rr=np.asarray(nm)[m]-np.asarray(st['net'])[m]
        wt=np.asarray(st['wt'])[m]; res.extend(list(rr[wt>0]))
    return np.sqrt(np.mean(np.array(res)**2))
rmin=rms(B0,KD0,TG0); thr=1.2*rmin
fig,ax=plt.subplots(1,2,figsize=(11,4.0))
# (a) b marginal profile at tau_G=12 (kappa_des co-fit)
bs=np.linspace(0.013,0.042,30)
rb=np.array([minimize_scalar(lambda kd:rms(b,kd,TG0),bounds=(0.05,0.55),method='bounded',options={'xatol':2e-4}).fun for b in bs])
ins=bs[rb<=thr]
ax[0].plot(bs,rb,'-',color='C0',lw=2); ax[0].axhline(thr,ls='--',color='0.5',lw=1)
ax[0].axvspan(ins.min(),ins.max(),color='C0',alpha=0.15)
ax[0].plot(B0,rmin,'o',color='C3',ms=6)
ax[0].set_xlabel('$b$  (5-HT$_{1B}$ drug coupling, nM$^{-1}$)'); ax[0].set_ylabel('RMS residual (Y-BOCS pts)')
ax[0].set_title('(a) $b$ at $\\tau_G{=}12$ wk: CI [%.3f, %.3f]'%(ins.min(),ins.max())); ax[0].grid(alpha=.25)
ax[0].text(0.03,0.95,'$1.2\\times$RMS$_{min}$',transform=ax[0].transAxes,fontsize=8,color='0.4',va='top')
# (b) tau_G profile (refit b)
taus=np.linspace(5,22,18); rt=[]
for tw in taus:
    bstar=minimize_scalar(lambda b:rms(b,KD0,tw),bounds=(0.01,0.05),method='bounded',options={'xatol':1e-4}).x
    rt.append(rms(bstar,KD0,tw))
rt=np.array(rt)
ax[1].plot(taus,rt,'-',color='C1',lw=2); ax[1].axhline(thr,ls='--',color='0.5',lw=1)
ax[1].axvspan(8,15,color='C1',alpha=0.15); ax[1].axvline(12,ls=':',color='C3',lw=1.5)
ax[1].set_xlabel('$\\tau_G$  (intrinsic remodeling time, wk)'); ax[1].set_ylabel('RMS residual (Y-BOCS pts)')
ax[1].set_title('(b) $\\tau_G$ profile: flat over [8,15] wk, pinned 12'); ax[1].grid(alpha=.25)
ax[1].text(12.2,rt.max(),'pinned\n(trial dur.)',fontsize=7.5,color='C3',va='top')
plt.tight_layout(); plt.savefig('fig_identifiability.pdf'); plt.savefig('fig_identifiability.png',dpi=140)
print('wrote fig_identifiability; RMS_min=%.3f  b_CI=[%.3f,%.3f]  tauG RMS range=[%.3f,%.3f]'%(rmin,ins.min(),ins.max(),rt.min(),rt.max()))
