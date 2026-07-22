"""SI figure (S9): response versus remission across baseline severity -- the Cohen contrast.
Frozen model (best_fit.json), fluoxetine 40/60 mg, 12-week drug-attributable endpoint.
(a) %Y-BOCS reduction vs baseline Y0 with the 25% and 35% response criteria;
(b) endpoint Y-BOCS vs baseline Y0 with the remission region (Y<12).
Message: response is achieved across a broad baseline range (especially at 25%), whereas
remission is reached only at low baseline -- response and remission gate at different severities."""
import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import build_fit_gcs_slow as M, gamma_profile_calib as GP, gcs_bistable_population as B
C=GP.C; bf=json.load(open('best_fit.json'))
THM=bf['theta_M']; BB=bf['b']; KD=bf['kappa_des']; HMAX=bf['h_max']; PS=GP.PHI_SAT; TAUG=bf['tau_G']
M.b_5HT1B=BB; M.setup(KD,HMAX); M.GAMMA_OBS=bf['gamma']; eC_h=M.eC_h

def endpoint(Y0, dose, D50, weeks=12.0, dt=0.05):
    a,G0,ok=GP.alpha_for(Y0,THM)
    if not ok: return np.nan
    tg=np.arange(0,weeks+dt/2,dt); eC=C.ec_series(B.const(dose),D50,tg); G=G0
    for e in eC:
        G+=dt*(a*B.Sshape(G,THM,PS)-(1.0+BB*(e-eC_h))*G)/TAUG; G=min(max(G,0.01),3.2)
    return M.Yread(G)

Y0=np.arange(14,38.1,0.5)
arms=[(40,'FLX 40 mg (u=0.94)','C0'),(60,'FLX 60 mg (u=0.96)','C3')]
red={}; yend={}
for dose,_,_ in arms:
    ye=np.array([endpoint(y,dose,2.7) for y in Y0])
    yend[dose]=ye; red[dose]=100*(Y0-ye)/Y0

fig,ax=plt.subplots(1,2,figsize=(11,4.1))
# (a) % reduction vs baseline
for dose,lbl,c in arms:
    ax[0].plot(Y0,red[dose],'-',color=c,lw=2,label=lbl)
for thr,ls in [(25,'--'),(35,':')]:
    ax[0].axhline(thr,color='0.4',ls=ls,lw=1.2)
    ax[0].text(37.6,thr+0.6,f'{thr}% response',ha='right',va='bottom',fontsize=8,color='0.35')
ax[0].set_xlabel('baseline Y-BOCS'); ax[0].set_ylabel('drug-attributable Y-BOCS reduction at 12 wk (%)')
ax[0].set_title('(a) Response declines with baseline severity'); ax[0].grid(alpha=.25)
ax[0].legend(fontsize=8,loc='upper right'); ax[0].set_xlim(14,38); ax[0].set_ylim(0,60)
# (b) endpoint Y vs baseline, remission region
ax[1].axhspan(0,12,color='C2',alpha=0.12)
ax[1].axhline(12,color='C2',ls='--',lw=1.2); ax[1].text(14.4,12.5,'remission (Y-BOCS $<$ 12)',fontsize=8,color='C2',va='bottom')
for dose,lbl,c in arms:
    ax[1].plot(Y0,yend[dose],'-',color=c,lw=2,label=lbl)
ax[1].set_xlabel('baseline Y-BOCS'); ax[1].set_ylabel('endpoint Y-BOCS at 12 wk')
ax[1].set_title('(b) Remission is reached only at low baseline'); ax[1].grid(alpha=.25)
ax[1].legend(fontsize=8,loc='upper left'); ax[1].set_xlim(14,38); ax[1].set_ylim(0,38)
plt.tight_layout(); plt.savefig('fig_cohen_severity.pdf'); print('wrote fig_cohen_severity.pdf')
# report the crossing baselines
def cross(x,y,lvl,dec=True):
    for i in range(len(x)-1):
        if (y[i]-lvl)*(y[i+1]-lvl)<0: return x[i]+(lvl-y[i])*(x[i+1]-x[i])/(y[i+1]-y[i])
    return np.nan
for dose,_,_ in arms:
    print(f" FLX{dose}: 25%red at Y0<={cross(Y0,red[dose],25):.1f}, 35% at Y0<={cross(Y0,red[dose],35):.1f}, "
          f"remission(Yend<12) at Y0<={cross(Y0,yend[dose],12):.1f}")
