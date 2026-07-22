"""Population-sensitivity analysis: how the assumed baseline Y-BOCS distribution N(mu,sigma) controls the
population-level predictions (net responder fraction, remission fraction, mean drug-attributable Delta Y-BOCS,
and the endpoint-distribution BC / VR). Single-patient response is threshold-like in baseline (the fold), so
every population number is that threshold integrated over the baseline distribution. We precompute the treated
endpoint map endY(Y0) ONCE, then all (mu,sigma) cells are cheap resampling.
Frozen params; representative adequate SSRI arm (FLX 40 mg, D50=2.7). EPS12 = severity-independent expectation."""
import numpy as np, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import build_fit_gcs_slow as M, gamma_profile_calib as GP, gcs_bistable_population as B

bf=json.load(open('best_fit.json'))
THM=bf['theta_M']; TAUG=bf['tau_G']; BB=bf['b']; KD=bf['kappa_des']; HMAX=bf['h_max']; PS=GP.PHI_SAT
M.b_5HT1B=BB; M.setup(KD,HMAX); M.GAMMA_OBS=bf['gamma']; eC_h=M.eC_h
EPS12=4.0*(1-np.exp(-12/5.5))                 # ~3.55 pts expectation at 12 wk (severity-independent)
DOSE,D50=40.0,2.7
tg=np.arange(0,12.001,0.05); ECdrug=GP.C.ec_series(B.const(DOSE),D50,tg)
REMIT=12.0

def endY_raw(Y0):
    a,G0,ok=GP.alpha_for(Y0,THM)
    if not ok: return np.nan
    G=G0
    for e in ECdrug:
        G+=0.05*(a*B.Sshape(G,THM,PS)-(1.0+BB*(e-eC_h))*G)/TAUG; G=min(max(G,0.01),3.2)
    return M.Yread(G)

# ---- precompute endY(Y0) on a fine grid (the expensive step, done once) ----
Y0g=np.arange(10.0,39.51,0.25)
endYg=np.array([endY_raw(y) for y in Y0g])
ok=~np.isnan(endYg); Y0g=Y0g[ok]; endYg=endYg[ok]
def endY(y): return np.interp(y,Y0g,endYg)

def bimod_coef(x):
    x=np.asarray(x,float); n=len(x); s=(x-x.mean())/x.std()
    g1=np.mean(s**3); k=np.mean(s**4)-3
    return (g1**2+1)/(k+3*(n-1)**2/((n-2)*(n-3)))

_rng=np.random.default_rng(7)
def metrics(mu,sd,N=8000):
    y0=np.clip(_rng.normal(mu,sd,N),10,39.5)
    em=endY(y0)                                   # treated endpoint before expectation
    Ytr=np.clip(em-EPS12,0,None)                  # treated observed endpoint (with expectation)
    Ypb=np.clip(y0-EPS12,0,None)                  # placebo observed endpoint
    dr35=100*(y0-Ytr)/y0>=35; pr35=100*(y0-Ypb)/y0>=35
    dr25=100*(y0-Ytr)/y0>=25; pr25=100*(y0-Ypb)/y0>=25
    resp35=100*(dr35.mean()-pr35.mean()); resp25=100*(dr25.mean()-pr25.mean())
    remit=100*np.mean(Ytr<REMIT)                  # treated-arm remission fraction (Y<12)
    dY=np.mean(y0-em)                             # net drug-attributable mean improvement (EPS cancels)
    return dict(resp35=resp35,resp25=resp25,remit=remit,dY=dY,
                BC=bimod_coef(Ytr),VR=Ytr.std(ddof=1)/y0.std(ddof=1))

# ---- grid table ----
MUS=[22,24,26,28,30]; SDS=[3,4,5,6,7]
print("=== net >=35% responder fraction (%) over (mean x SD) ===")
print("      "+"".join(f"SD={s} " for s in SDS))
GRID={}
for mu in MUS:
    row=[]
    for sd in SDS:
        m=metrics(mu,sd); GRID[(mu,sd)]=m; row.append(m['resp35'])
    print(f"mu={mu:>2} "+"".join(f"{r:>5.0f} " for r in row))
print("\n(mu,sd): resp35 / remit / meanDY / BC / VR  at selected cells")
for cell in [(28,4),(24,5.5),(26,5),(24,4),(24,7)]:
    m=metrics(*cell)
    print(f"  N{cell}: resp35={m['resp35']:.0f}%  remit={m['remit']:.0f}%  dY={m['dY']:.1f}  BC={m['BC']:.2f}  VR={m['VR']:.2f}")

# ---- figure: (a) mean sweep @ sd=5.5, (b) SD sweep @ mu=24, (c) heatmap resp35 ----
plt.rcParams.update({'font.size':11})
fig,(axA,axB,axC)=plt.subplots(1,3,figsize=(15.5,4.8))
mus=np.arange(22,30.01,0.5)
rA=[metrics(mu,5.5) for mu in mus]
axA.axhspan(16,22,color='0.8',alpha=0.6,label='Bloch 2010 ARD (16--22%)')
axA.plot(mus,[m['resp35'] for m in rA],'-o',color='#c0392b',ms=3,lw=2.4,label='net responder ($\\geq$35%)')
axA.plot(mus,[m['remit'] for m in rA],'-s',color='#1f3a93',ms=3,lw=2.4,label='remission ($Y<12$)')
axA.axvline(25,ls=':',color='0.4',lw=1.2); axA.text(25.1,axA.get_ylim()[1]*0.02,'typical trial mean',rotation=90,fontsize=8,va='bottom',color='0.4')
axA.set_xlabel('baseline population mean $\\mu$ (SD fixed $=5.5$)'); axA.set_ylabel('Percentage')
axA.set_title('(a)  Mean sweep: severity-gating',fontsize=12); axA.grid(alpha=.25); axA.legend(fontsize=8.5,loc='upper right')

sds=np.arange(3,7.01,0.25)
rB24=[metrics(24,sd) for sd in sds]; rB28=[metrics(28,sd) for sd in sds]
axB.axhspan(16,22,color='0.85',alpha=0.7,label='Bloch band')
axB.plot(sds,[m['resp35'] for m in rB24],'-o',color='#c0392b',ms=3,lw=2.4,label='$\\mu=24$ (mild population)')
axB.plot(sds,[m['resp35'] for m in rB28],'-s',color='#1f3a93',ms=3,lw=2.4,label='$\\mu=28$ (severe population)')
axB.set_xlabel('baseline population SD $\\sigma$'); axB.set_ylabel('net responder $\\geq$35% (%)')
axB.set_title('(b)  Width effect depends on severity',fontsize=12); axB.grid(alpha=.25); axB.legend(fontsize=8.5,loc='center right')

Z=np.array([[GRID[(mu,sd)]['resp35'] for sd in SDS] for mu in MUS])
im=axC.imshow(Z,origin='lower',aspect='auto',cmap='RdYlBu_r',extent=[SDS[0]-0.5,SDS[-1]+0.5,MUS[0]-1,MUS[-1]+1])
for i,mu in enumerate(MUS):
    for j,sd in enumerate(SDS):
        axC.text(sd,mu,f"{Z[i,j]:.0f}",ha='center',va='center',fontsize=8.5,color='k')
axC.set_xlabel('SD $\\sigma$'); axC.set_ylabel('mean $\\mu$'); axC.set_title('(c)  net responder $\\geq$35% (%)',fontsize=12)
axC.set_xticks(SDS); axC.set_yticks(MUS); fig.colorbar(im,ax=axC,fraction=0.046)
plt.tight_layout(); plt.savefig('fig_population_sensitivity.pdf'); plt.savefig('fig_population_sensitivity.png',dpi=140)
print("\nwrote fig_population_sensitivity.pdf")
