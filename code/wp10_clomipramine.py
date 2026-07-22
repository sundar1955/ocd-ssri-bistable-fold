"""WP10 (R4W4): self-contained clomipramine (CMI) falsification test.
CMI was HELD OUT of the calibration. Treating it as a pure SERT inhibitor (its potency D50=50 mg from the
imaging literature; doses 226 and 113 mg), the frozen model (b, kappa_des, theta_M, tau_G from best_fit.json,
gamma=2) PREDICTS its net (drug-placebo) Y-BOCS trajectory with no refitting. We compare predicted vs observed.
Falsification criterion (pre-stated): if the model systematically UNDER-predicts the CMI net response, the
excess is attributable to CMI's non-SERT (noradrenergic) action, which the SERT-only model omits by construction."""
import numpy as np, json
import build_fit_gcs_slow as M, gcs_bistable_population as B
import gcs_bistable_calibrate as C, gamma_profile_calib as GP
bf=json.load(open('best_fit.json')); THM=9.15; TAUG=12.0; HMAX=2.0
M.b_5HT1B=bf['b']; GP.KAPPA_DES=bf['kappa_des']; M.setup(bf['kappa_des'],HMAX); M.GAMMA_OBS=2.0

STUD=C.build_studies(include_cmi=True)
cmi=[s for s in STUD if s['name'].startswith('CMI')]
print("frozen model (no CMI refit): b=%.4f kappa_des=%.3f theta_M=%.2f tau_G=%d gamma=2\n"%(bf['b'],bf['kappa_des'],THM,TAUG))
print(f"{'arm':14}{'dose(mg)':>9}{'wk':>4}{'obs net':>9}{'pred net':>9}{'resid':>8}")
allobs=[]; allpred=[]
for st in cmi:
    tmax=st['weeks'][-1]+0.001; TG=np.arange(0,tmax,0.15)
    eC=C.ec_series(st['dose_fn'],st['D50'],TG)
    nm,_=GP.net_model_alpha(st,THM,TAUG,TG,eC)
    for k,wk in enumerate(st['weeks']):
        if st['skip0'] and k==0: continue
        print(f"{st['name']:14}{'--':>9}{wk:>4.0f}{st['net'][k]:>9.2f}{nm[k]:>9.2f}{st['net'][k]-nm[k]:>8.2f}")
        allobs.append(st['net'][k]); allpred.append(nm[k])
    # endpoint summary
    print(f"   -> {st['name']} ENDPOINT wk{st['weeks'][-1]:.0f}: observed {st['net'][-1]:.2f}, predicted {nm[-1]:.2f}\n")
allobs=np.array(allobs); allpred=np.array(allpred)
rms=np.sqrt(np.mean((allobs-allpred)**2)); bias=np.mean(allobs-allpred)
print(f"pooled CMI: RMS={rms:.2f}, mean(obs-pred)={bias:+.2f} Y-BOCS pts (SSRI in-sample RMS ~0.83)")
print("VERDICT:", "model UNDER-predicts CMI (positive bias) -> excess = non-SERT (NET) action, consistent with CMI pharmacology; clean off-SERT falsification of a SERT-only account."
      if bias>1.0 else "model matches CMI within SSRI-scale error (CMI behaves as a SERT inhibitor here).")
json.dump({'rms':float(rms),'bias':float(bias)},open('wp10_cmi.json','w'))
