"""Optimize EACH drug's D50 to its own arms (holding shared b,kappa_des at adopted values).
Produces literature-vs-optimized D50 per drug + pooled RMS, for the two-line Figure S4."""
import numpy as np, json
from scipy.optimize import minimize_scalar
import build_fit_gcs_slow as M, gcs_bistable_calibrate as C, gamma_profile_calib as GP
STUD=GP.STUD; THM=9.15; TAUG=12.0; HMAX=2.0; GAM=2.0
B=0.0268; KD=0.108  # normalized-form fit
LIT={'FLX':2.7,'CIT':3.4,'ESC':1.7,'PAR':5.0,'SRT':9.1,'FLV':50.}
BND={'FLX':(1,12),'CIT':(1,12),'ESC':(0.5,12),'PAR':(1,20),'SRT':(3,60),'FLV':(10,150)}
GP.KAPPA_DES=KD; C.KAPPA_DES=KD; M.GAMMA_OBS=GAM; M.b_5HT1B=B; M.setup(KD,HMAX)
tmax=max(s['weeks'][-1] for s in STUD)+0.001; tg=np.arange(0,tmax,0.15)
def sqres(st,d50):
    ec=C.ec_series(st['dose_fn'],d50,tg); nm,_=GP.net_model_alpha(st,THM,TAUG,tg,ec)
    m=slice(1,None) if st['skip0'] else slice(None)
    return list(np.asarray(nm)[m]-np.asarray(st['net'])[m])
by_drug={}
for st in STUD: by_drug.setdefault(st['drug'],[]).append(st)
OPT={}
for drug,arms in by_drug.items():
    def drms(d50):
        r=[]; [r.extend(sqres(st,d50)) for st in arms]; return np.sqrt(np.mean(np.array(r)**2))
    res=minimize_scalar(drms,bounds=BND[drug],method='bounded',options={'xatol':1e-3})
    OPT[drug]=float(res.x)
    print(f"{drug}: lit={LIT[drug]:<5} opt={res.x:6.2f}  (x{res.x/LIT[drug]:.2f} lit)   arm-RMS {drms(LIT[drug]):.3f} -> {res.fun:.3f}",flush=True)
def pooled(dm):
    r=[]; 
    for st in STUD: r.extend(sqres(st,dm[st['drug']]))
    return np.sqrt(np.mean(np.array(r)**2))
rl=pooled(LIT); ro=pooled(OPT)
print(f"\nPOOLED RMS:  literature D50 = {rl:.4f}   ->   optimized D50 = {ro:.4f}   ({100*(rl-ro)/rl:.1f}% better)",flush=True)
json.dump(dict(b=B,kdes=KD,theta_M=THM,tau_G=TAUG,lit=LIT,opt=OPT,rms_lit=rl,rms_opt=ro),
          open('opt_d50_fit.json','w'),indent=2)
print("wrote opt_d50_fit.json",flush=True)
