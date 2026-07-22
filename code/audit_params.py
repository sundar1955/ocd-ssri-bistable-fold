"""Reproducibility GATE for Paper A figures/tables.

Single source of truth = best_fit.json. This script verifies that every figure/table
generating script uses the canonical parameter values (no silently-diverged hardcoded copy,
the failure mode that let b drift to 0.0135 before). Run it as the LAST step before
regenerating the final figure set:  `python3 audit_params.py`  -> exit 0 = PASS.

Decision (2026-07-13): we rely on this audit as a gate rather than refactoring ~20 scripts,
because theta_M/h_max/gamma are frozen criterion-set values and the calibrated b is already
read-from-source in every paper figure.
"""
import json, re, glob, os, sys

bf = json.load(open('best_fit.json'))
CANON = {  # name : canonical value -- ALL sourced from best_fit.json (single source of truth)
  'THM':   bf['theta_M'],   'BB':    bf['b'],
  'HMAX':  bf['h_max'],     'KDES': bf['kappa_des'], 'eC_h': bf['eC_h'],
  'GAMMA': bf['gamma'],
}
pats = {
  'THM':  r'\bTHM\s*=\s*([^\s;]+)',
  'BB':   r'\bBB\s*=\s*([^\s;]+)',  'HMAX': r'\bHMAX\s*=\s*([^\s;]+)',
  'KDES': r'\b(?:KDES|KD)\s*=\s*([^\s;]+)', 'eC_h': r'eC_h\s*=\s*([^\s;]+)',
  'GAMMA':r'GAMMA_OBS\s*=\s*([^\s;,]+)|GAMMA\s*=\s*([^\s;,]+)',
}
FROMSRC = re.compile(r"bf\[|\.b\b|\.kappa_des|GP\.|M\.|Mod\.|\.eC_h|\.PHI|\.phid1_h|\.GAMMA|\.theta_M|\.h_max")
IDENT   = re.compile(r'^[A-Za-z_]\w*$')   # bare variable / function arg -> resolved at runtime, not a literal

def classify(name, rhs):
    v = CANON[name]
    if FROMSRC.search(rhs): return 'OK-src'
    if IDENT.match(rhs):    return 'OK-var'          # e.g. GAMMA_OBS=gam (function argument)
    try:
        val = float(eval(rhs, {'__builtins__':{}}, {}))
        return 'OK-lit' if abs(val-v) <= max(1e-9, abs(v)*1e-6) else 'MISMATCH'
    except Exception:
        return 'UNK'

scripts = sorted(set(glob.glob('fig*.py') + ['combo_fold.py','ap_bloch_compare.py','ap_foldreq.py',
           'build_calibration_figs.py','gamma_profile_calib.py']))
print("CANON (from best_fit.json): " + "  ".join(f"{k}={v}" for k,v in CANON.items()) + "\n")

from collections import defaultdict
tally=defaultdict(lambda: defaultdict(list)); bad=[]
for s in scripts:
    txt=open(s).read()
    for name,pat in pats.items():
        m=re.search(pat,txt)
        if not m: continue
        rhs=(next(g for g in m.groups() if g) if any(m.groups()) else m.group(1)).strip()
        st=classify(name,rhs); tally[name][st].append(os.path.basename(s))
        if st in ('MISMATCH','UNK'): bad.append(f"  [{st}] {s}: {name} = {rhs}")
for name in pats:
    d=tally[name]
    print(f"  {name:6s}: {len(d.get('OK-src',[]))} from-source, {len(d.get('OK-lit',[]))} hardcoded-correct, "
          f"{len(d.get('OK-var',[]))} runtime-var, {len(d.get('MISMATCH',[]))} MISMATCH, {len(d.get('UNK',[]))} unknown")
if bad:
    print("\nFLAGS:\n" + "\n".join(bad))
    print("\n*** FAIL: divergence from best_fit.json ***"); sys.exit(1)
print("\n*** PASS: every figure/table script matches best_fit.json ***"); sys.exit(0)
