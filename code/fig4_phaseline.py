"""Figure 4 (R2), THREE panels: treatment as motion of a bistable fixed point, across baseline severity.
(a) Y-BOCS0=20 -- detailed single-patient fold (four doses; saddle-node snapshot).
(b) Y-BOCS0=24 -- at the cusp: attractor reaches the fold only near-maximal occupancy (u~0.96).
(c) Y-BOCS0=28 -- resistant: attractor persists even at u=0.95.
Panels (b,c) show u=0,0.8,0.95. Marker/parameter definitions live in the manuscript caption."""
import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import build_fit_gcs_slow as M, gamma_profile_calib as GP, gcs_bistable_population as B
C=GP.C; bf=json.load(open('best_fit.json'))
THM=9.15; HMAX=2.0; BB=bf['b']; KD=bf['kappa_des']; PS=GP.PHI_SAT; eC_h=M.eC_h
M.GAMMA_OBS=2.0; M.b_5HT1B=BB; M.setup(KD,HMAX); GP.KAPPA_DES=KD
PURPLE='#6a3d9a'; BLUE='#1f3a93'; ORANGE='#E69F00'; RED='#D1495B'
def eC_ss(u):
    if u<=1e-6: return eC_h
    D50=10.; D=u*D50/(1-u); tg=np.arange(0,80,0.2); return float(C.ec_series(lambda t:D,D50,tg)[-1])
def RHS(G,al,ec): return al*B.Sshape(G,THM,PS)-(1.0+BB*(ec-eC_h))*G
def fps(al,ec):
    Gs=np.linspace(0.0008,2.7,6000); f=np.array([RHS(g,al,ec) for g in Gs]); out=[]
    for i in np.where(np.diff(np.sign(f))!=0)[0]:
        g=Gs[i]-f[i]*(Gs[i+1]-Gs[i])/(f[i+1]-f[i]); st=(RHS(g+1e-3,al,ec)-RHS(g-1e-3,al,ec))<0; out.append((g,st))
    return out
Gg=np.linspace(0,2.35,760)
fig,axs=plt.subplots(1,3,figsize=(12.0,3.76))

# ---------- panel (a): Y0=20 detailed fold ----------
axA=axs[0]; Y0=20; al,G0,ok=GP.alpha_for(Y0,THM)
uf=next(u for u in np.linspace(0,0.98,600) if not [g for g,s in fps(al,eC_ss(u)) if s and g>0.25])
u_at=round(uf-0.006,3); u_past=round(uf+0.05,3)
famA=[(0.0,'$u=0$  (drug-free)',BLUE,'-',3.0),(0.45,'$u=0.45$',ORANGE,'-',2.2),
      (u_at,f'$u={u_at:.2f}$  (at fold)',RED,'-',2.2),(u_past,f'$u={u_past:.2f}$  (past fold)','0.55','--',2.2)]
axA.axhline(0,color='0.6',lw=0.9,ls=':'); cA=[]
for u,lab,c,ls,lw in famA:
    ec=eC_ss(u); axA.plot(Gg,[RHS(g,al,ec) for g in Gg],color=c,lw=lw,ls=ls,zorder=3)
    cA.append(Line2D([],[],color=c,lw=lw,ls=ls,label=lab))
axA.plot(0,0,'o',ms=8.5,mfc='k',mec='k',zorder=8)          # health (stable)
for u in (0.0,0.45):                                       # saddle + OCD attractor for each pre-fold dose
    for g,st in fps(al,eC_ss(u)):
        if g<0.05: continue
        axA.plot(g,0,'o',ms=9,mfc=(PURPLE if st else 'white'),mec=('#3a2060' if st else 'k'),
                 mew=(1.1 if st else 1.6),zorder=8)
ecf=eC_ss(u_at); rr=[g for g,s in fps(al,ecf) if g>0.5]; gm=float(np.mean(rr)) if len(rr)>=2 else 1.16
axA.plot(gm,RHS(gm,al,ecf),'D',ms=10,mfc=RED,mec='k',mew=0.9,zorder=9)
for g0,dirn in [(0.42,-1),(1.19,1)]:
    axA.annotate('',xy=(g0+0.13*dirn,-0.016),xytext=(g0,-0.016),
                 arrowprops=dict(arrowstyle='-|>',color='0.45',lw=1.1,mutation_scale=10),zorder=4)
axA.set_ylim(-0.185,0.105); axA.set_xlim(-0.03,2.12)
legA=axA.legend(handles=cA,fontsize=8.2,loc='upper left',framealpha=0.96); axA.add_artist(legA)
# on-figure fixed-point marker key (shared convention across Figs 2b, 4, 7)
fpkey=[Line2D([],[],ls='',marker='o',mfc='k',mec='k',ms=7,label='health (stable)'),
       Line2D([],[],ls='',marker='o',mfc='white',mec='k',mew=1.4,ms=7,label='saddle (threshold)'),
       Line2D([],[],ls='',marker='o',mfc=PURPLE,mec='#3a2060',ms=7,label='OCD attractor'),
       Line2D([],[],ls='',marker='D',mfc=RED,mec='k',ms=7,label='saddle-node (fold)')]
axA.legend(handles=fpkey,fontsize=7.3,loc='lower right',framealpha=0.96,title='fixed points',title_fontsize=7.3)

# ---------- panels (b,c): Y0=24, 28 severity ----------
US=[(0.0,BLUE),(0.8,ORANGE),(0.95,RED)]
def panel(ax,Y0,legloc):
    al,G0,ok=GP.alpha_for(Y0,THM); ax.axhline(0,color='0.6',lw=0.8,ls=':'); pk=0
    for u,c in US:
        ec=eC_ss(u); y=np.array([RHS(g,al,ec) for g in Gg]); ax.plot(Gg,y,color=c,lw=2.2,zorder=3); pk=max(pk,y.max())
        for g,st in fps(al,ec):
            if g<0.05: continue
            ax.plot(g,0,'o',ms=9,mfc=(PURPLE if st else 'white'),mec=('#3a2060' if st else 'k'),
                    mew=(1.1 if st else 1.6),zorder=6)
    ax.plot(0,0,'o',ms=8.5,mfc='k',mec='k',zorder=7)
    ax.set_xlim(-0.03,2.35); ax.set_ylim(-0.20,min(0.13,pk*1.12)+0.005)
    h=[Line2D([],[],color=c,lw=2.2,label=f'$u={u:g}$') for u,c in US]
    ax.legend(handles=h,fontsize=8.5,loc=legloc,framealpha=0.96,ncol=1)
panel(axs[1],24,'upper right'); panel(axs[2],28,'lower center')

for ax in axs: ax.set_xlabel('$G_{CS}$  (corticostriatal weight)'); ax.grid(alpha=.2)
axs[0].set_ylabel('plasticity rate  $\\tau_G\\,dG_{CS}/dt$')
for ax,(lab,Y0) in zip(axs,[('(a)',20),('(b)',24),('(c)',28)]):
    ax.text(0.025,0.035,f'{lab}  $Y_{{\\mathrm{{model}},0}}={Y0}$',transform=ax.transAxes,
            fontsize=10.5,va='bottom',ha='left',fontweight='bold')
plt.tight_layout(); plt.savefig('fig4_phaseline.pdf'); plt.savefig('fig4_phaseline.png',dpi=150)
print(f"Y0=20 fold u={uf:.3f}; wrote fig4_phaseline.pdf (3 panels, no inset)")
