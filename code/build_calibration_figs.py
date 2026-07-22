"""Calibration figures for Paper A (gamma=2). Reads best-fit params from best_fit.json (written by
profile_likelihood.py) or falls back to anchored values. Produces:
  fig_calib_main.pdf  -- representative arms (FLX Tollefson 3 doses + CIT Montgomery) with CI band
  fig_calib_grid.pdf  -- ALL 13 arms (Supplement)
  table_calib_rms.txt -- per-arm RMS
Model net-trajectory on a fine grid = population mean of (Yb - Y(t)) over the arm's baseline-Y distribution."""
import numpy as np, json, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import build_fit_gcs_slow as M, gamma_profile_calib as GP
B=GP.B; PHI_SAT=GP.PHI_SAT; STUD=GP.STUD; THM=9.15; HMAX=2.0
bf=json.load(open('best_fit.json')) if os.path.exists('best_fit.json') else {'tau_G':12,'b':0.0268,'kappa_des':0.108,'h_max':2.0,'theta_M':9.15,'eC_h':2.8}
GP.KAPPA_DES=bf.get('kappa_des',0.285); GP.C.KAPPA_DES=bf.get('kappa_des',0.285)  # joint-fit kappa_des
TG,BB=bf['tau_G'],bf['b']; BCI=bf.get('b_ci',None)
print(f"using tau_G={TG:.3f} b={BB:.4f}")
# optimized per-drug D50 (dashed line): opt_d50_fit.json written by experiment_d50_perdrug.py
OPTD=json.load(open('opt_d50_fit.json'))['opt'] if os.path.exists('opt_d50_fit.json') else {}

def net_curve(st,tg,tG=TG,b=BB,hmax=HMAX,d50=None):
    """population-mean net improvement (Yb-Y(t)) on fine grid tg."""
    M.GAMMA_OBS=2.0; M.b_5HT1B=b; M.setup(GP.KAPPA_DES,hmax); eC0=M.eC_h
    ec=GP.C.ec_series(st['dose_fn'], st['D50'] if d50 is None else d50, tg); ys,ws=GP.C.quad(st['Ymean'],st['Ysd'],n=15)
    acc=np.zeros(len(tg)); wsum=0.0
    for Yb,w in zip(ys,ws):
        a,G0,ok=GP.alpha_for(Yb,THM)
        if not ok: continue
        G=G0; Yv=np.empty(len(tg))
        for i in range(len(tg)):
            Yv[i]=M.Yread(G); e=ec[i]
            G+=(tg[1]-tg[0])*(a*B.Sshape(G,THM,PHI_SAT)-(1.0+b*(e-eC0))*G)/tG
            G=min(max(G,0.01),3.2)
        acc+=w*(Yb-Yv); wsum+=w
    return acc/wsum if wsum>0 else acc*np.nan

def plot_arm(ax,st):
    tg=np.arange(0,st['weeks'][-1]+0.001,0.1); y=net_curve(st,tg)
    ax.plot(tg,y,'-',color='C0',lw=2,zorder=3)                    # literature D50 (solid)
    dopt=OPTD.get(st['drug'])
    if dopt is not None:                                          # optimized D50 (dashed)
        ax.plot(tg,net_curve(st,tg,d50=dopt),'--',color='C3',lw=1.8,zorder=3)
    if BCI:  # band from b CI (tau_G held)
        env=np.array([net_curve(st,tg,b=bv) for bv in BCI])
        ax.fill_between(tg,env.min(0),env.max(0),color='C0',alpha=0.18,zorder=1)
    m=slice(1,None) if st['skip0'] else slice(None)
    ax.errorbar(np.array(st['weeks'])[m],np.array(st['net'])[m],yerr=np.array(st['se'])[m],
                fmt='o',color='k',ms=4,capsize=2,zorder=4)
    ax.set_title(st['name'],fontsize=8); ax.grid(alpha=.25)
    if dopt is not None:                                          # per-panel D50 box
        txt=f"$D_{{50}}$ (mg)\n— lit {st['D50']:g}\n- - opt {dopt:.1f}"
        ax.text(0.035,0.965,txt,transform=ax.transAxes,fontsize=6.0,va='top',ha='left',
                bbox=dict(boxstyle='round,pad=0.3',fc='white',ec='0.6',alpha=0.9))

# per-arm RMS table: literature D50 vs optimized per-drug D50 (same wt filter)
rows=[]
for st in STUD:
    tg=np.arange(0,st['weeks'][-1]+0.001,0.1)
    m=slice(1,None) if st['skip0'] else slice(None)
    wk=np.array(st['weeks'])[m]; net=np.array(st['net'])[m]; wt=np.asarray(st['wt'])[m]
    def rmsval(d50):
        y=net_curve(st,tg,d50=d50); res=(np.interp(wk,tg,y)-net)[wt>0]
        return len(res),float(np.sqrt(np.mean(res**2)))
    dr=st['drug']; dopt=OPTD.get(dr)
    n,rl=rmsval(None); _,ro=rmsval(dopt)
    rows.append((st['name'],n,st['D50'],dopt,rl,ro))
def poolrms(idx):  # N-weighted pooled RMS
    num=sum(n*r[idx]**2 for _,n,_,_,*r in [(x[0],x[1],x[2],x[3],x[4],x[5]) for x in rows]); return None
# N-weighted pooled
NL=sum(n for _,n,_,_,_,_ in rows)
pl=np.sqrt(sum(n*rl**2 for _,n,_,_,rl,_ in rows)/NL)
po=np.sqrt(sum(n*ro**2 for _,n,_,_,_,ro in rows)/NL)
with open('table_calib_rms.txt','w') as fh:
    fh.write(f"{'arm':16s}{'N':>3}{'D50lit':>8}{'D50opt':>8}{'RMSlit':>8}{'RMSopt':>8}\n")
    for nm,n,dl,do,rl,ro in rows:
        fh.write(f"{nm:16s}{n:>3}{dl:>8.1f}{(do or 0):>8.1f}{rl:>8.3f}{ro:>8.3f}\n")
    fh.write(f"{'POOLED':16s}{NL:>3}{'':>8}{'':>8}{pl:>8.3f}{po:>8.3f}\n")
print(f"wrote table_calib_rms.txt  (pooled lit={pl:.3f} opt={po:.3f})")

# MAIN: FLX Tollefson doses + CIT Montgomery
main=[s for s in STUD if s['name'].startswith('FLX_Toll')]+[s for s in STUD if s['name'].startswith('CIT_Mont')][:1]
n=len(main); fig,ax=plt.subplots(1,n,figsize=(3.1*n,3.0),squeeze=False)
for a,st in zip(ax[0],main): plot_arm(a,st); a.set_xlabel('week')
ax[0,0].set_ylabel('net improvement (Y-BOCS)')
plt.tight_layout(); plt.savefig('fig_calib_main.pdf'); plt.savefig('fig_calib_main.png',dpi=140); print("wrote fig_calib_main")

# GRID: all 13
nc=4; nr=int(np.ceil(len(STUD)/nc)); fig,ax=plt.subplots(nr,nc,figsize=(3.0*nc,2.4*nr),squeeze=False)
for k,st in enumerate(STUD): plot_arm(ax[k//nc][k%nc],st)
for k in range(len(STUD),nr*nc): ax[k//nc][k%nc].axis('off')
for a in ax[-1]: a.set_xlabel('week')
plt.tight_layout(); plt.savefig('fig_calib_grid.pdf'); plt.savefig('fig_calib_grid.png',dpi=140); print("wrote fig_calib_grid")
