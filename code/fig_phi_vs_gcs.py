"""Figure 2 (restructured per reviewer): (a) severity readout map phi_d1 & G_CS vs Y-BOCS;
(b) NEW schematic of the plasticity rate dG_CS/dt vs G_CS (drug-free) showing the three fixed points ---
health (G_CS=0, stable), the commitment-threshold saddle (unstable), and the elevated OCD attractor (stable) ---
so a reader can visualize the fold. The firing-rate-vs-G_CS panels move to the Supplement (fig_phi_change /
fig_phi_abs)."""
import numpy as np, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import build_fit_ssri_panel as P
from var_circuit_validation import idx
import gcs_bistable_population as B, gamma_profile_calib as GP, build_fit_gcs_slow as M

LAB={'e':'cortex e (OFC proxy)','i':'cortex i','d1':'caudate $D_1$ (severity)','d2':'$D_2$-MSN',
     'p1':'GPi/SNr','p2':'GPe','c':'STN','s':'thalamic relay','r':'TRN'}
Gs=np.linspace(0.0,3.2,161)
seed=np.array([12,12,40,20,60,30,30,15,28.]); PHI={p:[] for p in idx}
for G in Gs:
    seed=P.circuit(G,0.0,seed)
    for p,k in idx.items(): PHI[p].append(seed[k])
PHI={p:np.array(v) for p,v in PHI.items()}
Qd1=65.0; phid1_0=PHI['d1'][0]; ceil=Qd1-phid1_0

# ---------- plasticity fold machinery for panel (b) ----------
bf=json.load(open('best_fit.json')); THM=9.15; PS=GP.PHI_SAT; BB=bf['b']; eC_h=M.eC_h
M.GAMMA_OBS=2.0; M.b_5HT1B=BB; M.setup(bf['kappa_des'],2.0); GP.KAPPA_DES=bf['kappa_des']
Y0=24                                   # representative moderate-OCD patient
al,G0,ok=GP.alpha_for(Y0,THM)
Gg=np.linspace(0.0,2.35,900)
rate=np.array([al*B.Sshape(g,THM,PS)-g for g in Gg])   # tau_G dG/dt, drug-free (sink coeff = 1)
# fixed points = zeros; stable where rate crosses + -> -
fps=[]
for i in np.where(np.diff(np.sign(rate))!=0)[0]:
    g=Gg[i]-rate[i]*(Gg[i+1]-Gg[i])/(rate[i+1]-rate[i])
    stable = rate[i]>0 and rate[i+1]<0
    fps.append((g,stable))
# ensure health (G=0) shown as stable FP
if not any(abs(g)<0.02 for g,_ in fps): fps=[(0.0,True)]+fps

# ================= MAIN Figure 2 =================
plt.rcParams.update({'font.size':12})
fig,(axA,axB)=plt.subplots(1,2,figsize=(13.5,5.2))

# --- panel (a): readout map phi_d1 & G_CS vs Y-BOCS (bigger fonts, bigger legend) ---
phid1=PHI['d1']; Yp=40*np.clip((phid1-phid1_0)/(Qd1-phid1_0),0.,None)**2.0
CR='#c0392b'; CB='#1f3a93'
axA.plot(Yp,phid1,color=CR,lw=3.0,zorder=4)
axA.set_xlabel('Y-BOCS  (severity readout $Y=40\\,p^{\\gamma}$, $\\gamma=2$)',fontsize=13.5)
axA.set_ylabel('$\\phi_{d_1}$  (caudate $D_1$ firing, s$^{-1}$)',color=CR,fontsize=13.5)
axA.tick_params(axis='both',labelsize=12); axA.tick_params(axis='y',labelcolor=CR)
axA.set_xlim(0,40); axA.grid(alpha=.25)
axA.axhline(Qd1,ls=':',color=CR,lw=1.0,alpha=0.6); axA.text(1,Qd1-1.8,f'$Q_{{d_1}}={Qd1:.0f}$',fontsize=11,color=CR,va='top')
axA2=axA.twinx()
axA2.plot(Yp,Gs,color=CB,lw=3.0,ls='--',zorder=4)
axA2.set_ylabel('$G_{CS}$  (corticostriatal weight)',color=CB,fontsize=13.5)
axA2.tick_params(axis='y',labelcolor=CB,labelsize=12)
axA.legend(handles=[Line2D([],[],color=CR,lw=3.0,label='$\\phi_{d_1}$ (left axis)'),
                    Line2D([],[],color=CB,lw=3.0,ls='--',label='$G_{CS}$ (right axis)')],
           fontsize=13,loc='upper left',framealpha=0.96,borderpad=0.8,labelspacing=0.6,handlelength=2.4)
axA.set_title('(a)  Severity readout: $\\phi_{d_1}$ and $G_{CS}$ vs Y-BOCS',fontsize=13.5)

# --- panel (b): dG_CS/dt vs G_CS schematic with the three fixed points ---
axB.axhline(0,color='0.6',lw=1.0,ls=':')
axB.plot(Gg,rate,color='#6a3d9a',lw=3.0,zorder=3)
for g,st in fps:
    if g<-1e-6 or g>2.3: continue
    axB.plot(g,0,'o',ms=13,mfc=('#6a3d9a' if st else 'white'),mec='#3a2060',mew=1.6,zorder=6)
# label the three fixed points
def near(gt):
    c=[g for g,_ in fps if abs(g-gt)<0.35]; return c[0] if c else gt
gh=0.0; gsad=sorted([g for g,s in fps if 0.05<g and not s])[:1]; gocd=sorted([g for g,s in fps if g>0.3 and s],reverse=True)[:1]
axB.annotate('healthy state\n(stable, $G_{CS}=0$)',xy=(gh,0),xytext=(0.18,rate.max()*0.55),fontsize=11.5,
             color='#3a2060',ha='left',arrowprops=dict(arrowstyle='-|>',color='0.45',lw=1.2))
if gsad:
    axB.annotate('commitment threshold\n(unstable saddle)',xy=(gsad[0],0),xytext=(gsad[0]-0.15,rate.min()*0.7),
                 fontsize=11.5,color='#3a2060',ha='center',arrowprops=dict(arrowstyle='-|>',color='0.45',lw=1.2))
if gocd:
    axB.annotate('OCD attractor\n(stable, elevated)',xy=(gocd[0],0),xytext=(gocd[0]-0.55,rate.max()*0.55),
                 fontsize=11.5,color='#3a2060',ha='left',arrowprops=dict(arrowstyle='-|>',color='0.45',lw=1.2))
# flow arrows on the G-axis
for g0,dirn in [(0.42,-1),(1.15,1)]:
    axB.annotate('',xy=(g0+0.16*dirn,0),xytext=(g0,0),arrowprops=dict(arrowstyle='-|>',color='0.5',lw=1.4),zorder=4)
axB.set_xlabel('$G_{CS}$  (corticostriatal weight = disease severity)',fontsize=13.5)
axB.set_ylabel('plasticity rate  $\\tau_G\\,dG_{CS}/dt$',fontsize=13.5)
axB.tick_params(axis='both',labelsize=12); axB.grid(alpha=.2)
axB.set_xlim(-0.03,2.05); axB.set_ylim(rate.min()*1.15,rate.max()*1.15)
axB.set_title('(b)  The bistable landscape (drug-free, Y-BOCS$_0=24$)',fontsize=13.5)
plt.tight_layout(); plt.savefig('fig_phi_vs_gcs.pdf'); plt.savefig('fig_phi_vs_gcs.png',dpi=140)
print("wrote fig_phi_vs_gcs.pdf (new: readout map + dG/dt schematic). FPs:",[(round(g,3),s) for g,s in fps])

# ================= Supplement: firing rates vs G_CS (moved from main) =================
fig2,(s1,s2)=plt.subplots(1,2,figsize=(13,4.7))
for p in idx:
    lw=2.6 if p=='d1' else 1.4; z=5 if p=='d1' else 2
    s1.plot(Gs,PHI[p]-PHI[p][0],lw=lw,zorder=z,label=LAB[p])
s1.axhline(ceil,ls='--',color='0.55',lw=1.1); s1.text(3.15,ceil-1.0,f'caudate $D_1$ ceiling $\\approx{ceil:.0f}$',fontsize=7.5,color='0.4',va='top',ha='right')
s1.set_xlabel('$G_{CS}$'); s1.set_ylabel('$\\phi(G_{CS})-\\phi(0)$ (s$^{-1}$)'); s1.set_title('(a) change from healthy baseline',fontsize=10); s1.grid(alpha=.25); s1.legend(fontsize=7,loc='upper left')
for p in idx:
    lw=2.6 if p=='d1' else 1.4; z=5 if p=='d1' else 2
    s2.plot(Gs,PHI[p],lw=lw,zorder=z,label=LAB[p])
s2.axhline(Qd1,ls='--',color='0.55',lw=1.0); s2.text(0.05,Qd1-1.2,f'$Q_{{d_1}}={Qd1:.0f}$',fontsize=7.5,color='0.4',va='top')
s2.set_xlabel('$G_{CS}$'); s2.set_ylabel('firing rate $\\phi$ (s$^{-1}$)'); s2.set_title('(b) absolute rates',fontsize=10); s2.grid(alpha=.25)
plt.tight_layout(); plt.savefig('fig_phi_abs.pdf'); plt.savefig('fig_phi_abs.png',dpi=140)
print("wrote fig_phi_abs.pdf (supplement: firing rates vs G_CS, 2 panels)")
