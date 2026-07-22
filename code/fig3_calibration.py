"""Main Figure 3 (R1): (a) fluoxetine dose-response trajectories; (b) predicted-vs-observed scatter at
literature D50 for the 13 calibration arms, with Kronig 1999 sertraline overlaid as an OUT-OF-SAMPLE
(held-out) validation. Adopted fit uses literature D50; the D50-optimized scatter is in Supplement S9.1."""
import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import build_fit_gcs_slow as M, gcs_bistable_calibrate as C, gamma_profile_calib as GP
STUD=GP.STUD; THM=9.15
bf=json.load(open('best_fit.json')); TG=bf['tau_G']; BB=bf['b']; KD=bf['kappa_des']; BCI=bf.get('b_ci')
HMAX=2.0; LIT={'FLX':2.7,'CIT':3.4,'ESC':1.7,'PAR':5.0,'SRT':9.1,'FLV':50.}
COL={'FLX':'#1f77b4','CIT':'#ff7f0e','ESC':'#2ca02c','PAR':'#d62728','SRT':'#9467bd','FLV':'#8c564b'}
GP.KAPPA_DES=KD; C.KAPPA_DES=KD; M.GAMMA_OBS=2.0; M.b_5HT1B=BB; M.setup(KD,HMAX)

def net_curve(st,tg,b=BB,d50=None):
    M.b_5HT1B=b; M.setup(KD,HMAX); eC0=M.eC_h
    ec=C.ec_series(st['dose_fn'], st['D50'] if d50 is None else d50, tg); ys,ws=C.quad(st['Ymean'],st['Ysd'],n=15)
    acc=np.zeros(len(tg)); wsum=0.0
    for Yb,w in zip(ys,ws):
        a,G0,ok=GP.alpha_for(Yb,THM)
        if not ok: continue
        G=G0; Yv=np.empty(len(tg))
        for i in range(len(tg)):
            Yv[i]=M.Yread(G); G+=(tg[1]-tg[0])*(a*GP.B.Sshape(G,THM,GP.PHI_SAT)-(1.0+b*(ec[i]-eC0))*G)/TG; G=min(max(G,0.01),3.2)
        acc+=w*(Yb-Yv); wsum+=w
    return acc/wsum if wsum>0 else acc*np.nan

def scatter_pts(dm):
    M.b_5HT1B=BB; M.setup(KD,HMAX)   # restore adopted b (3a loop left a CI value)
    pts={d:[[],[]] for d in COL}; ao=[]; ap=[]
    for st in STUD:
        d=st['drug']; ec=C.ec_series(st['dose_fn'],dm[d],np.arange(0,max(s['weeks'][-1] for s in STUD)+0.001,0.15))
        nm,_=GP.net_model_alpha(st,THM,TG,np.arange(0,max(s['weeks'][-1] for s in STUD)+0.001,0.15),ec)
        m=slice(1,None) if st['skip0'] else slice(None)
        for o,p,w in zip(np.array(st['net'])[m],np.array(nm)[m],np.asarray(st['wt'])[m]):
            if w>0: pts[d][0].append(o); pts[d][1].append(p); ao.append(o); ap.append(p)
    return pts,np.sqrt(np.mean((np.array(ap)-np.array(ao))**2))

# Kronig 1999 sertraline (PMID 10211919): held-out. 50mg wk0-3, titrate 100/150/165(mean max) after.
def kron_dose(t): return 50. if t<3 else (100. if t<6 else (150. if t<8 else 165.))
KWK=[0,1,2,3,4,6,8,10,12]; KNET=[0,0.1,0.0,1.8,1.2,2.2,4.0,4.8,4.2]   # net drop, digitized Fig 1
def kron_pts():
    M.b_5HT1B=BB; M.setup(KD,HMAX)
    kron=C.mk("SRT_Kronig","SRT",kron_dose,9.1,KWK,KNET,[1.0]*len(KWK),25.2,3.8,1.0)
    tg=np.arange(0,12.001,0.15); ec=C.ec_series(kron['dose_fn'],kron['D50'],tg)
    nm,_=GP.net_model_alpha(kron,THM,TG,tg,ec)
    o=np.array(kron['net'])[1:]; p=np.array(nm)[1:]
    return o,p,np.sqrt(np.mean((p-o)**2))

fig=plt.figure(figsize=(8.6,3.6)); gsp=GridSpec(1,2,width_ratios=[1.4,1],wspace=0.30)
# (a) FLX dose-response: distinct colours, staggered data error bars, model b-CI bars at even weeks
ax0=fig.add_subplot(gsp[0]); CF={'20':'#0072B2','40':'#E69F00','60':'#009E73'}; ORD=['20','40','60']
flx={s['name'][8:]:s for s in STUD if s['drug']=='FLX'}; tgf=np.arange(0,13.001,0.05)
cur={}; env={}
for d in ORD:
    st=flx[d]; mid=net_curve(st,tgf,b=BB); a=net_curve(st,tgf,b=BCI[0]); bb=net_curve(st,tgf,b=BCI[1])
    cur[d]=mid; env[d]=(np.minimum(a,bb),np.maximum(a,bb)); ax0.plot(tgf,mid,'-',color=CF[d],lw=2,zorder=3)
for wk,d in {2:'20',4:'40',6:'60',8:'20',10:'40',12:'60'}.items():
    i=int(np.argmin(abs(tgf-wk))); mid=cur[d][i]; lo,hi=env[d][0][i],env[d][1][i]
    ax0.errorbar(wk,mid,yerr=[[mid-lo],[hi-mid]],fmt='s',color=CF[d],ms=5,mfc='white',mew=1.3,capsize=2.5,zorder=6)
deb={'20':[3,7,9,11,13],'40':[5,9,11,13],'60':[1,7,9,11,13]}
for d in ORD:
    st=flx[d]; wk=np.array(st['weeks']); net=np.array(st['net']); se=np.array(st['se'])
    ax0.plot(wk[1:],net[1:],'o',color=CF[d],ms=4.5,zorder=5)
    for w in deb[d]:
        j=int(np.argmin(abs(wk-w))); ax0.errorbar(wk[j],net[j],yerr=se[j],fmt='none',ecolor=CF[d],capsize=2.5,elinewidth=1,zorder=5)
ax0.set_xticks([3,6,9,12]); ax0.set_xlim(0,13.3); ax0.set_xlabel('week'); ax0.set_ylabel('net $\\Delta$Y-BOCS')
ax0.set_title('(a) fluoxetine dose-response',fontsize=10); ax0.grid(alpha=.25)
dh=[Line2D([],[],color=CF[d],lw=2,label=f'{d} mg') for d in ORD]
mh=[Line2D([],[],marker='o',color='0.3',ls='',label='data'),Line2D([],[],marker='s',color='0.3',ls='',mfc='white',mew=1.3,label='model')]
lg=ax0.legend(handles=dh,title='dose',fontsize=7.5,loc='upper left'); ax0.add_artist(lg)
ax0.legend(handles=mh,fontsize=6.5,loc='lower right')
# (b) predicted-vs-observed scatter at literature D50 + Kronig held-out
ax=fig.add_subplot(gsp[1]); pts,rms=scatter_pts(LIT); lim=[-1.5,8.5]
ax.plot(lim,lim,'-',color='0.6',lw=1,zorder=1)
for off in (1.0,-1.0):
    ax.plot(lim,[lim[0]+off,lim[1]+off],'--',color='0.72',lw=0.8,zorder=1)
for d in COL:
    if pts[d][0]: ax.scatter(pts[d][0],pts[d][1],s=20,color=COL[d],alpha=0.8,edgecolor='w',lw=0.3,label=d,zorder=3)
ko,kp,krms=kron_pts()
ax.scatter(ko,kp,s=60,marker='*',facecolor='none',edgecolor='k',lw=1.2,zorder=6,label='Kronig (held-out)')
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal'); ax.grid(alpha=.25)
ax.set_xlabel('observed net $\\Delta$Y-BOCS'); ax.set_ylabel('model net $\\Delta$Y-BOCS')
ax.set_title('(b) literature $D_{50}$',fontsize=10)
ax.legend(fontsize=6.0,ncol=2,loc='upper left')
ax.text(0.96,0.05,f'RMS $={rms:.2f}$\nheld-out $={krms:.2f}$',transform=ax.transAxes,fontsize=7.6,ha='right',va='bottom',
        bbox=dict(boxstyle='round',fc='white',ec='0.7',alpha=0.9))
plt.tight_layout(); plt.savefig('fig3_calibration.pdf'); plt.savefig('fig3_calibration.png',dpi=145)
print(f"wrote fig3_calibration.pdf  (calibration RMS={rms:.3f}, Kronig held-out RMS={krms:.3f})")
