"""
DATA + HELPER module for the net(drug-placebo) SSRI calibration.
Provides the digitized trial set (build_studies), the feed-forward serotonin engine (ec_series),
and the deterministic Gaussian baseline-Y quadrature (quad). These are imported by
gamma_profile_calib, which performs the CANONICAL calibration in the normalized plasticity form
(4 free params theta_M, tau_G, b, h_max0; NO kappa_h; baseline sink coeff = 1).
 - eC(t) precomputed ONCE per arm (eC decoupled from G_CS -> feedforward) => fast.
 - Placebo handled per-study OUTSIDE (data are net-of-placebo; epsilon cancels). CMI HELD OUT (falsification).
 - Anchored (CANONICAL): kappa_des=0.12(calib), tau_desens=2, eC_h=2.8, K=10, betaSERT=1.0, k_rem=0.15*kSERT0, gamma_obs=2, D50 pinned(Meyer FLX/CIT/PAR/SRT [V]; ESC=1.7=CIT/2 Baldinger2014+Klein2007 [V]; FLV=50 Takano2006 [V]).
The historical stand-alone joint bistable+population calibrator (kappa_h, 6-param DE fit) is preserved
in local archive/gcs_bistable_calibrate_kappah_v1.py.
"""
import numpy as np, time, sys
from scipy.optimize import differential_evolution
import build_fit_gcs_slow as M
import gcs_bistable_population as B
P=M.P
KAPPA_DES=0.120; TAU_D=2.0   # calibrated (baseline-ref fit)

# ---------- net data (improvement, + = better) ----------
TOL_PBO=np.array([24.3,23.3,22.4,23.4,22.4,22.6,23.5,23.6])   # Tollefson placebo arm (wk 0,1,3,5,7,9,11,13)
def toll_net(d):
    dr=P.TOL[d]; return (dr[0]-dr)-(TOL_PBO[0]-TOL_PBO)        # net improvement vs placebo, per week
def const(mg): return lambda t: mg
def titr(d1,d2,tsw): return lambda t: d1 if t<tsw else d2

# ---- VERIFIED replacement arms (digitized weekly net; see verified_ssri_digitized.py / .npz) ----
# These REPLACE the removed FLV_Askari & SRT_Ghob, which Hamanaka 2023 shows are 5-HT3-antagonist
# AUGMENTATION trials (granisetron/setron added to SSRI), NOT SSRI monotherapy -> wrong drug class.
# Greist 1995 SRT (PMID 7702445): pooled fixed 50/100/200 (flat dose-resp, SERT-sat), base 23.8;
#   endpoint wk12 net 2.16 PRECISE (Table 4 pooled SRT -5.57 vs PBO -3.41); Fig1 weekly digitized.
GREIST_SRT_WK =np.array([0,1,2,4,6,8,10,12.])
GREIST_SRT_NET=np.array([0,0.6,0.9,1.4,1.7,1.9,2.05,2.16])
GREIST_SRT_SE =np.array([1.0,1.1,1.1,1.1,1.0,1.0,0.9,0.8])
# Kamijima 2004 PAR (PMID 15298657): titrated 20->40, base 24.3; wk6 4.12 & wk12 4.65 PRECISE; Fig1 digitized.
KAMI_PAR_WK =np.array([0,1,2,4,6,8,10,12.])
KAMI_PAR_NET=np.array([0,0.6,1.5,2.9,4.12,4.35,4.5,4.65])
KAMI_PAR_SE =np.array([1.0,1.1,1.1,1.1,1.0,1.0,1.0,1.0])
# Hollander 2003 FLV-CR (PMID 12823077): flexible 100-300 CR, base 26.6; endpoint-only wk12 net 2.9.
HOLL_FLV_WK =np.array([0,12.])
HOLL_FLV_NET=np.array([0,2.9])
HOLL_FLV_SE =np.array([1.0,1.0])

# STUDIES: each = dict(name,drug,dose_fn,D50,weeks,net_obs,se,wtmask,Ymean,Ysd,W)
def mk(name,drug,dose_fn,D50,weeks,net,se,Ymean,Ysd,W,wtmask=None,skip0=True):
    weeks=np.asarray(weeks,float); net=np.asarray(net,float); se=np.asarray(se,float)
    if wtmask is None: wtmask=np.ones_like(weeks)
    return dict(name=name,drug=drug,dose_fn=dose_fn,D50=D50,weeks=weeks,net=net,se=se,
                wt=np.asarray(wtmask,float),Ymean=Ymean,Ysd=Ysd,W=W,skip0=skip0)

def build_studies(include_cmi=False):
    S=[]
    for d in (20,40,60):   # FLX Tollefson (absolute->net); baseline per-arm ~ TOL[d][0]
        S.append(mk(f"FLX_Toll{d}","FLX",const(d),2.7,P.WK_T,toll_net(d),
                    P.TOLsd[d]/np.sqrt(P.TOLn[d]),P.TOL[d][0],5.5,1.0))
    for d in (20,40,60):   # CIT Montgomery (net = MONc)
        S.append(mk(f"CIT_Mont{d}","CIT",const(d),3.4,P.WK_M,P.MONc[d],
                    np.full(len(P.WK_M),1.5),25.4,3.9,0.7))
    for d,net,se in [(10,P.NET_ESC10,P.SE_ESC10),(20,P.NET_ESC20,P.SE_ESC20)]:  # ESC Stein
        S.append(mk(f"ESC_Stein{d}","ESC",const(d),1.7,P.STEIN_WK,net,se,27.,5.0,0.5,wtmask=P.STEIN_WT,skip0=False))
    S.append(mk("PAR_Stein40","PAR",const(40),5.0,P.STEIN_WK,P.NET_PAR40,P.SE_PAR40,27.,5.0,0.5,wtmask=P.STEIN_WT,skip0=False))
    S.append(mk("PAR_Zohar","PAR",const(37.5),5.0,P.ZOHAR_WK,P.ZOHAR_PAR_NET,P.SE_ZPAR,25.5,4.0,0.5))
    # --- VERIFIED replacements for removed Askari/Ghob (5-HT3-antagonist augmentation trials) ---
    S.append(mk("SRT_Greist","SRT",const(100),9.1,GREIST_SRT_WK,GREIST_SRT_NET,GREIST_SRT_SE,23.8,5.3,0.5))
    S.append(mk("PAR_Kamijima","PAR",titr(20,40,2.),5.0,KAMI_PAR_WK,KAMI_PAR_NET,KAMI_PAR_SE,24.3,4.5,0.5))
    S.append(mk("FLV_Holl","FLV",const(200),50.,HOLL_FLV_WK,HOLL_FLV_NET,HOLL_FLV_SE,26.6,5.0,0.3))
    if include_cmi:
        S.append(mk("CMI_Greist","CMI",const(226),50.,P.GREIST_WK,P.GREIST_CMI_NET,P.SE_GCMI,26.2,5.2,0.5))
        S.append(mk("CMI_Zohar","CMI",const(113.1),50.,P.ZOHAR_WK,P.ZOHAR_CMI_NET,P.SE_ZCMI,25.5,4.0,0.5))
    return S

# ---------- fast engine ----------
def ec_series(dose_fn,D50,tgrid):
    th=M.theta_h; e=M.eC_h; out=np.empty(len(tgrid)); dt=tgrid[1]-tgrid[0]
    for i,t in enumerate(tgrid):
        e=M.eC_qss(th,dose_fn(t),D50); out[i]=e
        th+=dt*((1-th)-KAPPA_DES*(e-M.eC_h)*th)/TAU_D
    return out
def quad(Ymean,Ysd,n=41,lo=13.,hi=35.):
    a=max(lo,Ymean-2.5*Ysd); bb=min(hi,Ymean+2.5*Ysd)
    ys=np.linspace(a,bb,n); w=np.exp(-0.5*((ys-Ymean)/Ysd)**2); w/=w.sum(); return ys,w
