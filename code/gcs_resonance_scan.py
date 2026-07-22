"""
OCD resonance scan (solver validated against vAR Table 2).
Question: as G_CS increases (OCD), does the DAMPED resonance spectrum of the STABLE fixed point
shift into / amplify delta (1-4 Hz) and theta (4-8 Hz)?  (vAR full-PD is stable-with-resonances;
we test the analogous statement for OCD.)

For each G_CS we (1) solve the fixed point, (2) build gains G_ab=rho_a nu_ab, (3) find the
low-frequency dispersion roots s=sigma+i*omega and report the LEAST-DAMPED root in 0-40 Hz
(its freq and damping Re -> proximity to imaginary axis = sharper resonance), and
(4) compute the cortical noise-driven power spectrum P(f) ~ |phi_e(f)|^2 and its delta/theta share.
"""
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from var_circuit_validation import idx, M as NU0, ext, S, Sprime, N

alpha, beta, gam = 160., 640., 125.
ie = idx['e']; rho_CS = 0.7

T = np.zeros((N,N))
def sd(a,b,ms): T[idx[a],idx[b]] = ms/1000.
sd('e','s',35); sd('i','s',35); sd('s','e',50); sd('r','e',50)
sd('s','r',2);  sd('r','s',2);  sd('s','p1',3)
sd('d1','e',2); sd('d2','e',2); sd('d1','s',2); sd('d2','s',2)
sd('p1','d1',1); sd('p1','p2',1); sd('p1','c',1)
sd('p2','d2',1); sd('p2','p2',1); sd('p2','c',1); sd('c','e',1); sd('c','p2',1)

def NU(G):
    m = NU0.copy(); m[idx['d1'],ie] += G; m[idx['d2'],ie] += rho_CS*G; return m

def gains(G):
    seed = np.array([12,12,7.4,3.5,69,48,28,14,28.])
    phi = fsolve(lambda p: p - S(NU(G)@p+ext), seed, xtol=1e-12)
    rho = Sprime(NU(G)@phi+ext)
    return rho[:,None]*NU(G), phi

def dispersion(s, Gm):
    Dab=(1+s/alpha)*(1+s/beta); D=np.full(N,Dab,dtype=complex); D[ie]=Dab*(1+s/gam)**2
    return np.linalg.det(np.diag(D) - Gm*np.exp(-s*T))

def lowfreq_roots(Gm, wmax=260.):
    """all roots with 0<=f<=~40Hz; return list of complex s."""
    sigs=np.linspace(-45,5,110); ws=np.linspace(0.,wmax,240)
    Z=np.array([[np.log(abs(dispersion(sg+1j*w,Gm))+1e-300) for w in ws] for sg in sigs])
    roots=[]
    for i in range(1,len(sigs)-1):
        for j in range(1,len(ws)-1):
            if (Z[i,j]<=Z[i-1,j] and Z[i,j]<=Z[i+1,j] and Z[i,j]<=Z[i,j-1] and Z[i,j]<=Z[i,j+1]):
                def F(x): v=dispersion(x[0]+1j*x[1],Gm); return [v.real,v.imag]
                try:
                    sol=fsolve(F,[sigs[i],ws[j]],xtol=1e-11)
                    if abs(dispersion(sol[0]+1j*sol[1],Gm))<1e-4:
                        roots.append(sol[0]+1j*abs(sol[1]))
                except Exception: pass
    uniq=[]
    for r in roots:
        if not any(abs(r-u)<1e-3 for u in uniq): uniq.append(r)
    return uniq

# transfer-function power spectrum of cortical field to relay-noise drive:
# phi = (Diag_D - K)^{-1} * forcing;  forcing only on relay s (brainstem noise).
def cortical_power(Gm, fgrid):
    P=[]
    for f in fgrid:
        s=1j*2*np.pi*f
        Dab=(1+s/alpha)*(1+s/beta); D=np.full(N,Dab,dtype=complex); D[ie]=Dab*(1+s/gam)**2
        Mmat=np.diag(D)-Gm*np.exp(-s*T)
        b=np.zeros(N,dtype=complex); b[idx['s']]=1.0
        try:
            x=np.linalg.solve(Mmat,b); P.append(abs(x[ie])**2)
        except np.linalg.LinAlgError:
            P.append(np.nan)
    return np.array(P)

Gvals=np.linspace(0,3.0,16)
fgrid=np.linspace(0.3,40,400)
print(f"{'G_CS':>5} {'phi_d1':>7} | least-damped root 0-40Hz    | delta%  theta%  alpha-peak(Hz)")
records=[]
for G in Gvals:
    Gm,phi=gains(G)
    roots=lowfreq_roots(Gm)
    # least damped in 0.5-40 Hz
    band=[r for r in roots if 0.3 < r.imag/(2*np.pi) < 40]
    ld=max(band,key=lambda r:r.real) if band else None
    Pf=cortical_power(Gm,fgrid)
    tot=np.trapz(Pf,fgrid)
    dshare=np.trapz(Pf[(fgrid>=1)&(fgrid<4)],fgrid[(fgrid>=1)&(fgrid<4)])/tot
    tshare=np.trapz(Pf[(fgrid>=4)&(fgrid<8)],fgrid[(fgrid>=4)&(fgrid<8)])/tot
    fpk=fgrid[np.nanargmax(Pf)]
    records.append((G,phi[idx['d1']],ld,dshare,tshare,fpk,Pf))
    lds = f"Re={ld.real:+6.2f} f={ld.imag/(2*np.pi):5.2f}Hz" if ld else "   (none)   "
    print(f"{G:5.2f} {phi[idx['d1']]:7.2f} | {lds}      | {dshare*100:5.1f}  {tshare*100:5.1f}   {fpk:5.2f}")

# --- figure: spectra at a few G_CS + resonance-root track ---
fig,ax=plt.subplots(1,2,figsize=(11,4.3))
for G,phi,ld,ds,ts,fpk,Pf in records:
    if np.isclose(G,Gvals,atol=1e-9).any() and round(G,2) in [0.0,0.6,1.2,1.8,2.4,3.0]:
        ax[0].semilogy(fgrid,Pf/np.trapz(Pf,fgrid),lw=1.8,label=f'G_CS={G:.1f}')
ax[0].set_xlabel('frequency (Hz)'); ax[0].set_ylabel('normalized cortical power')
ax[0].axvspan(1,4,color='navy',alpha=.06); ax[0].axvspan(4,8,color='green',alpha=.06)
ax[0].set_xlim(0,40); ax[0].legend(fontsize=8,ncol=2); ax[0].set_title('Cortical power spectrum vs $G_{CS}$')
Gr=[r[0] for r in records]; dd=[r[3]*100 for r in records]; tt=[r[4]*100 for r in records]
ax[1].plot(Gr,dd,'o-',color='navy',label='delta 1-4 Hz %')
ax[1].plot(Gr,tt,'s-',color='green',label='theta 4-8 Hz %')
ax[1].set_xlabel('$G_{CS}$'); ax[1].set_ylabel('relative band power (%)')
ax[1].legend(); ax[1].set_title('delta/theta share vs $G_{CS}$')
plt.tight_layout(); plt.savefig('fig_gcs_resonance.png',dpi=140); plt.savefig('fig_gcs_resonance.pdf')
print("\nwrote fig_gcs_resonance.png / .pdf")
