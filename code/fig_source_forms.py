"""SI figure: plasticity source form -- what creates the fold, what only positions it.
(a) phase lines for linear / frozen-BCM-quadratic / our-cubic (drug-free)
(b) bifurcation: OCD attractor Y vs SSRI occupancy u -- cubic (fold) vs indirect-response null (graded slide)
(c) sliding-theta_M: drug-free Y(t) -- frozen (persists) vs sliding (homeostatic self-cure)"""
import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import build_fit_gcs_slow as M, gcs_bistable_population as B, gcs_bistable_calibrate as C
bf=json.load(open('best_fit.json')); BB=bf['b']; KD=bf['kappa_des']; PS=65.0; TH=9.15; phih=7.39; eC_h=M.eC_h; TAUG=12.0
M.b_5HT1B=BB; M.setup(KD,2.0); M.GAMMA_OBS=2.0
BLUE='#1f3a93'; ORANGE='#E69F00'; RED='#D1495B'; GREEN='#2a8f5a'; PURPLE='#6a3d9a'; GREY='0.55'
def pd(G): return B.phd1(G)
def S_lin(G):  p=pd(G); return B.phe(G)*(p-phih)
def S_quad(G): p=pd(G); return B.phe(G)*(p-phih)*(p-TH)
def S_cub(G):  p=pd(G); return B.phe(G)*(p-phih)*(p-TH)*(PS-p)
try:
    gf=json.load(open('graded_freeb_fit.json')); Gm=gf['Gm']; B_NULL=gf['b']; TAUG_NULL=gf['tau_G_g']
except FileNotFoundError:
    Gm=0.200; B_NULL=BB; TAUG_NULL=19.0
def Sm(G): return G/(1+G/Gm)                      # indirect-response (graded null)
def fps(S,a,gmax=2.75):
    Gs=np.linspace(0.0,gmax,7000); f=np.array([a*S(g)-g for g in Gs]); out=[]
    for i in np.where(np.diff(np.sign(f))!=0)[0]:
        g=Gs[i]-f[i]*(Gs[i+1]-Gs[i])/(f[i+1]-f[i]); st=(a*S(g+1e-3)-(g+1e-3))-(a*S(g-1e-3)-(g-1e-3))<0
        if g>0.02: out.append((g,st))
    return out
def eC_ss(u):
    if u<=1e-6: return eC_h
    D50=10.; D=u*D50/(1-u); tg=np.arange(0,80,0.2); return float(C.ec_series(lambda t:D,D50,tg)[-1])

fig,axs=plt.subplots(1,3,figsize=(15.6,4.7))
# ---------- (a) phase lines ----------
axA=axs[0]; Gg=np.linspace(0,2.6,700); G0=M.G_of_Y(24)
variants=[('linear-Hebbian',S_lin,BLUE),('BCM quadratic (frozen $\\theta_M$)',S_quad,ORANGE),('our cubic (+ soft ceiling)',S_cub,RED)]
axA.axhline(0,color='0.6',lw=0.9,ls=':')
hh=[]
for lab,S,c in variants:
    a=G0/S(G0); y=np.array([a*S(g)-g for g in Gg]); y=y/np.max(np.abs(y))
    axA.plot(Gg,y,color=c,lw=2.3,zorder=3); hh.append(Line2D([],[],color=c,lw=2.3,label=lab))
    for g,st in fps(S,a):
        if g<0.05: continue
        axA.plot(g,0,'o',ms=9,mfc=(c if st else 'white'),mec=('k'),mew=(1.0 if st else 1.6),zorder=6)
        axA.annotate(f'Y={M.Yread(g):.0f}',(g,0),xytext=(0,10 if st else -16),textcoords='offset points',
                     fontsize=7.6,ha='center',color=c,fontweight='bold' if st else 'normal')
axA.plot(0,0,'o',ms=8.5,mfc='k',mec='k',zorder=7)
axA.set_xlim(-0.03,2.6); axA.set_ylim(-1.15,1.72); axA.set_xlabel('$G_{CS}$  (corticostriatal weight)')
axA.set_ylabel('plasticity rate  $\\tau_G\\,dG_{CS}/dt$  (rescaled)')
axA.legend(handles=hh,fontsize=8.0,loc='upper left',bbox_to_anchor=(0.0,0.99),framealpha=0.96)
axA.text(0.975,0.99,'filled = stable, open = saddle\nhealth (black) at $G_{CS}=0$',transform=axA.transAxes,
         fontsize=7.4,va='top',ha='right')

# ---------- (b) bifurcation: attractor Y vs occupancy ----------
axB=axs[1]; us=np.linspace(0,0.995,140)
def trace(S,a,gstart,tau,b):                       # each model uses its OWN calibrated b and tau
    ys=[]
    for u in us:
        ec=eC_ss(u); G=gstart; dt=0.04
        for t in np.arange(0,400,dt): G=min(max(G+dt*(a*S(G)-(1+b*(ec-eC_h))*G)/tau,1e-4),3.4)
        ys.append(M.Yread(G)); gstart=G
    return np.array(ys)
ac=G0/S_cub(G0); yc=trace(S_cub,ac,G0,TAUG,BB)
ag=(1+G0/Gm);   yg=trace(Sm,ag,G0,TAUG_NULL,B_NULL)
axB.plot(us,yc,color=RED,lw=2.6,label=f'our cubic (fold), $b={BB:.3f}$',zorder=4)
axB.plot(us,yg,color=GREEN,lw=2.6,ls='--',label=f'indirect-response null, $b={B_NULL:.3f}$',zorder=4)
# mark the fold jump
ij=np.argmax(np.abs(np.diff(yc))); axB.annotate('saddle-node\n(remission jump)',(us[ij],yc[ij]*0.5),
    xytext=(0.44,3.2),fontsize=8,color=RED,ha='left',arrowprops=dict(arrowstyle='-|>',color=RED,lw=1.1))
axB.axhline(12,color='0.5',lw=0.9,ls=':'); axB.text(0.5,12.6,'remission (Y=12)',fontsize=7.4,color='0.4',ha='center')
axB.set_xlabel('SSRI occupancy  $u$'); axB.set_ylabel('endpoint  $Y_{\\mathrm{model}}$'); axB.set_ylim(-1.5,27)
axB.legend(fontsize=8.0,loc='lower left',framealpha=0.96)

# ---------- (c) sliding theta_M ----------
axC=axs[2]
G0q=M.G_of_Y(24); aq=G0q/S_quad(G0q)
# stable attractor of frozen quadratic
st=[g for g,s in fps(S_quad,aq,3.1) if s and g>0.05]; Gstar=max(st)
phi0=phih; tt=np.arange(0,100,0.02)
def run(tauT):
    G=Gstar; thM=TH; ys=[]
    dt=tt[1]-tt[0]
    for t in tt:
        G=min(max(G+dt*(aq*B.phe(G)*(pd(G)-phih)*(pd(G)-thM)-G)/TAUG,1e-4),3.4)
        thM=thM+dt*((pd(G)**2/phi0)-thM)/tauT; ys.append(M.Yread(G))
    return np.array(ys)
axC.plot(tt,run(1e9),color=PURPLE,lw=2.6,label='frozen $\\theta_M$  (our assumption)')
for tau,c in [(500,BLUE),(150,ORANGE),(50,RED)]:
    axC.plot(tt,run(tau),color=c,lw=2.1,ls='--',label=f'sliding $\\theta_M$, $\\tau_\\theta={tau}$ wk')
axC.axhline(12,color='0.5',lw=0.9,ls=':'); axC.text(97,13,'remission (Y=12)',fontsize=7.4,color='0.4',ha='right')
axC.set_xlabel('time (weeks, drug-free)'); axC.set_ylabel('$Y_{\\mathrm{model}}$'); axC.set_xlim(0,100); axC.set_ylim(-1.5,40)
axC.legend(fontsize=8.0,loc='center right',framealpha=0.96)
for ax,lab in zip(axs,['(a)','(b)','(c)']):
    ax.text(0.02,0.975,lab,transform=ax.transAxes,fontsize=12,va='top',ha='left',fontweight='bold'); ax.grid(alpha=.18)
plt.tight_layout(); plt.savefig('fig_source_forms.pdf'); plt.savefig('fig_source_forms.png',dpi=140)
print(f"fold jump at u={us[ij]:.2f} (Y {yc[ij]:.1f}->{yc[ij+1]:.1f}); graded end Y={yg[-1]:.1f}; cubic end Y={yc[-1]:.1f}")
print("wrote fig_source_forms.pdf/png")
