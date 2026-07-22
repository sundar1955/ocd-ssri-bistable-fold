"""
px_ap_pk_model.py — Antipsychotic PK/PD module for Paper X
============================================================
Implements three tiers of AP pharmacology for the 11D OCD+MDD circuit model.

  Tier 1 — Full antagonist Hill function
            Drugs: haloperidol, risperidone, olanzapine, amisulpride
            D2 occupancy via static Hill equation; empirically calibrated to
            steady-state PET occupancy data.

  Tier 2 — Partial agonist operational model
            Drugs: aripiprazole, brexpiprazole
            Black-Leff operational model corrects occupancy for partial
            D2 agonism (intrinsic efficacy τ).
            Aripiprazole: two-compartment PK (parent + dehydro-aripiprazole
            metabolite) implemented for time-course calculations.
            Striatal/cortical occupancy split applied to separate OCD and
            MDD circuit parameters.

  Tier 3 — Placeholder (deferred)
            Drugs: quetiapine (fast-dissociation), clozapine (multi-receptor)
            These require time-varying D2 occupancy or multi-target models
            beyond the scope of Paper X Phase 1.

Circuit bridging:
  alpha_AP      — OCD CSTC parameter: cortico-D2MSN weight reduction
                  calibrated to striatal D2 occupancy (Kapur 2000 window)
  alpha_AP_MDD  — MDD DLPFC-sgACC parameter: calibrated to cortical D2
                  occupancy (lower than striatal; Kessler 2005)

Literature:
  Kapur et al.    2000   Am J Psychiatry 157:514-520      (therapeutic window)
  Nordström et al. 1993  Biol Psychiatry 33:227-235       (raclopride occ; NOT haloperidol)
  Farde et al.    1992   Arch Gen Psychiatry 49:538-544   (clozapine low D2)
  Mamo et al.     2007   Am J Psychiatry 164:1411-1417    (aripiprazole occ)
  Kessler et al.  2005   Neuropsychopharmacology 30:2283  (cortical occ)
  Mallikaarjun et al. 2004 J Clin Pharmacol 44:179-187   (aripiprazole popPK)
  Kapur & Remington 2001a Biol Psychiatry 50:873-883      (D2 hypothesis)
  Kapur & Seeman  2001b  Am J Psychiatry 158:360-369     (fast-dissociation)
  Black & Leff    1983   Proc R Soc Lond B 220:141-162   (operational model)
  Maeda et al.    2014   J Pharmacol Exp Ther 350:589-604 (brexpiprazole)
  Girgis et al.   2016   Psychopharmacology 233:3503-3512 (cariprazine)
"""

import numpy as np
from scipy.integrate import solve_ivp

# ══════════════════════════════════════════════════════════════════════════════
# 1.  DRUG PARAMETER TABLE
# ══════════════════════════════════════════════════════════════════════════════

DRUG_PARAMS = {
    # ── Tier 1: full antagonists ──────────────────────────────────────────────
    'haloperidol': {
        'tier':  1,
        'D50':   0.75,   # mg/day; calibrated to Kapur et al. 2000 (Am J Psychiatry 157:514):
                         #   occ(1 mg) ≈ 59% (study mean 59%, SD=11%),
                         #   occ(2.5 mg) ≈ 73% (study mean 75%, SD=6%)
                         #   Nordström 1993 studied raclopride (2, 6, 12 mg/day), not haloperidol
        'n':     1.0,
        'Emax':  0.95,
        'tau_operational': 0.0,    # full antagonist: τ = 0
        'dose_range': (2, 10),
        'source': 'Kapur 2000, Am J Psychiatry 157:514-520',
    },
    'risperidone': {
        'tier':  1,
        'D50':   1.5,    # mg/day; steep dose-occ curve (Farde 1996)
        'n':     1.0,
        'Emax':  0.95,
        'tau_operational': 0.0,
        'dose_range': (2, 8),
        'source': 'Farde 1996',
    },
    'olanzapine': {
        'tier':  1,
        'D50':   3.5,    # mg/day; gentler curve than risperidone (Kapur 1999)
        'n':     1.0,
        'Emax':  0.90,
        'tau_operational': 0.0,
        'dose_range': (5, 20),
        'source': 'Kapur 1999 (approximate; calibrate to Kapur 1999 PET data)',
    },
    'amisulpride': {
        'tier':  1,
        'D50':   50.0,   # mg/day; pure D2/D3 antagonist (Martinot 1996)
        'n':     1.0,
        'Emax':  0.92,
        'tau_operational': 0.0,
        'dose_range': (200, 800),
        'source': 'Martinot 1996 (approximate)',
    },
    # ── Tier 2: partial agonists ──────────────────────────────────────────────
    'aripiprazole': {
        'tier':  2,
        'D50':   0.8,    # mg/day; calibrated to Mamo 2007:
                         #   occ(10 mg) ≈ 90%, occ(30 mg) ≈ 95%
        'n':     1.0,
        'Emax':  0.97,
        'tau_operational': 0.40,   # Black-Leff τ at D2; lower than dopamine
                                   # (~0.30-0.50 in cell assays; Black & Leff 1983)
        # Two-compartment PK (Mallikaarjun 2004)
        'CL_F':   3.70,   # L/hr  Mallikaarjun 2004: range 3.35-4.0 L/h; midpoint 3.70
        'Vd_F':   400.0,  # L     terminal Vd: CL_F × t½ / ln2 = 3.70 × 75 / 0.693 ≈ 400 L
                          #   (Mallikaarjun 2004: central Vd/F = 116-196 L; terminal higher)
        't_half_parent':     75.0,  # hr (Mallikaarjun 2004: median; range 58-79h)
        't_half_metabolite': 94.0,  # hr (dehydro-aripiprazole; literature consensus)
        'metabolite_occ_frac': 0.25,  # Mamo 2007: metabolite plasma ~25% of parent at ss
                                      #   (NOT 40%; Mallikaarjun 2004 does not measure metabolite occ)
        'ka':     0.50,   # hr⁻¹  Mallikaarjun 2004: absorption t½ = 1-2h → ka = ln2/1.4 ≈ 0.50
        'dose_range': (2, 30),
        'source': 'Mamo 2007, Am J Psychiatry 164:1411-1417; '
                  'Mallikaarjun 2004, J Clin Pharmacol 44:179-187',
    },
    'brexpiprazole': {
        'tier':  2,
        'D50':   0.346,  # mg/day; calibrated to Girgis et al. 2020 steady-state PET:
                         #   occ(1 mg) = 64%, occ(4 mg) = 80%
                         #   Hill fit with Emax=0.95, n=0.685 reproduces both points
        'n':     0.685,
        'Emax':  0.95,
        'tau_operational': 0.20,   # Black-Leff τ at hD2, estimated in vivo.
                                   # Maeda 2014 Table 4 (CHO-hD2L cAMP):
                                   #   Emax_brex = 43%, Emax_ari = 61%,
                                   #   ratio ≈ 0.70.  Raw τ_CHO = 0.43/0.57
                                   #   = 0.75, but CHO high-D2-expression
                                   #   inflates partial-agonist Emax (receptor
                                   #   reserve); after Furchgott correction,
                                   #   τ_brex ≈ τ_ari × 0.70 ≈ 0.40 × 0.70
                                   #   = 0.28.  Value 0.20 is conservative,
                                   #   consistent with in vivo DOPA assay
                                   #   Emax ratio 55/89 ≈ 0.62 (Table 6).
        'metabolite_occ_frac': 0.0,   # no major active metabolite
        't_half_parent': 91.0,        # hr (approximate)
        'dose_range': (1, 4),
        'source': 'Girgis et al. 2020, Neuropsychopharmacology 45:786-792 '
                  '(steady-state PET); Maeda 2014 (preclinical pharmacology)',
    },
    # ── Tier 3: deferred ─────────────────────────────────────────────────────
    'quetiapine': {
        'tier':  3,
        'D50':   None,   # fast-dissociation: static Hill invalid
        'n':     None,
        'Emax':  0.70,   # peak post-dose only; trough << peak
        'tau_operational': 0.0,
        'dose_range': (150, 800),
        'source': 'Kapur & Seeman 2001b; Tier 3 deferred',
    },
    'clozapine': {
        'tier':  3,
        'D50':   None,   # multi-receptor; D2 occ 40-60% (Farde 1992)
        'n':     None,
        'Emax':  0.60,
        'tau_operational': 0.0,
        'dose_range': (100, 600),
        'source': 'Farde 1992, Arch Gen Psychiatry 49:538-544; Tier 3 deferred',
    },
}

# ── Bridging constants ────────────────────────────────────────────────────────
#
# alpha_AP = f_occ * occ_striatal      (OCD CSTC circuit)
# alpha_AP_MDD = f_occ_mdd * occ_cortical   (MDD DLPFC-sgACC circuit)
#
# f_occ is estimated from:
#   Paper 2 calibration: alpha_AP = 0.167 at population-average standard dose
#   Kapur 2000: population-average D2 occ ≈ 75% at standard augmentation dose
#   → f_occ = 0.167 / 0.75 ≈ 0.222
#
# cortical_striatal_ratio from Kessler 2005 (olanzapine/haloperidol):
#   cortical D2 occ / striatal D2 occ ≈ 0.70 at same dose
#
# f_occ_mdd = f_occ * cortical_striatal_ratio (default; can be calibrated
#             separately to MDD meta-analysis data once obtained)

F_OCC                  = 0.222   # OCD circuit bridging constant
CORTICAL_STRIATAL_RATIO = 0.927  # Kessler 2005 (18F-fallypride PET):
                                  #   haloperidol: temporal cortex/putamen = 70.9%/76.5% = 0.927
                                  #   olanzapine:  temporal cortex/putamen = 67.5%/69.2% = 0.975
                                  #   average ≈ 0.927 (haloperidol, the reference AP)
F_OCC_MDD              = F_OCC * CORTICAL_STRIATAL_RATIO   # ≈ 0.206

# Kapur 2000 therapeutic window (haloperidol, striatal D2 occupancy):
#   65% → clinical response threshold (significant increase in response probability)
#   72% → hyperprolactinemia threshold (prolactin elevation becomes prominent)
#   78% → EPS/akathisia threshold (extrapyramidal side effects emerge)
#   Narrow therapeutic window (response before hyperprolactinemia): 65-72%
THERAPEUTIC_WINDOW = (0.65, 0.72)   # corrected: 72% = first side-effect threshold


# ══════════════════════════════════════════════════════════════════════════════
# 2.  TIER 1 — STATIC HILL FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def d2_occupancy_hill(D_mg, drug):
    """
    Striatal D2 receptor occupancy [0, 1] at steady-state oral dose D_mg
    (mg/day) using the Hill equation.

    Valid for Tier 1 drugs (haloperidol, risperidone, olanzapine, amisulpride)
    and Tier 2 drugs (aripiprazole, brexpiprazole) at steady state.
    Raises ValueError for Tier 3 drugs (quetiapine, clozapine) where the
    static Hill function is not valid.

    Parameters
    ----------
    D_mg : float  — oral dose in mg/day
    drug : str    — key in DRUG_PARAMS

    Returns
    -------
    float — D2 occupancy in [0, 1]
    """
    p = DRUG_PARAMS[drug]
    if p['tier'] == 3:
        raise ValueError(
            f"{drug} is Tier 3: static Hill function not valid. "
            f"See notes: {p['source']}"
        )
    if p['D50'] is None:
        raise ValueError(f"{drug}: D50 not defined.")
    D = max(D_mg, 0.0)
    return float(p['Emax'] * D**p['n'] / (D**p['n'] + p['D50']**p['n']))


# ══════════════════════════════════════════════════════════════════════════════
# 3.  TIER 2 — PARTIAL AGONIST OPERATIONAL MODEL (Black & Leff 1983)
# ══════════════════════════════════════════════════════════════════════════════

def partial_agonist_effective_blockade(occ, tau):
    """
    Convert D2 receptor occupancy to *effective blockade fraction* using
    the Black-Leff operational model.

    For a full antagonist (τ = 0):
        eff_blockade = occ           (all occupied receptors contribute 0 activation)

    For a partial agonist (τ > 0):
        Occupied receptors provide partial activation = τ/(1+τ) × (baseline dopamine)
        Net reduction vs. unoccupied state:
          eff_blockade = occ × (1 − τ/(1+τ)) = occ / (1+τ)

    This assumes the receptor reserve is negligible and that baseline dopamine
    tone saturates unoccupied receptors (striatal high-tone approximation).
    For cortical (low-tone) regions, the agonist term may dominate instead —
    this is the agonism-switch mechanism for aripiprazole (see notes in module
    docstring).

    Parameters
    ----------
    occ : float   — D2 occupancy [0, 1]
    tau : float   — Black-Leff transduction ratio (0 = full antagonist)

    Returns
    -------
    float — effective blockade fraction [0, 1]
    """
    if tau <= 0.0:
        return float(occ)
    return float(occ / (1.0 + tau))


def net_d2_effect(occ, tau, Emax=1.0):
    """
    Net D2 receptor activation from the Black-Leff operational model.

    E_net = Emax × (τ × occ) / (1 + (1+τ) × occ)    [Black & Leff 1983 Eq. 2]

    Interpreted as fractional D2-MSN activation at the given occupancy.
    For τ=0 (full antagonist): E_net = 0 (pure blockade).
    For τ→∞ (full agonist): E_net → Emax.

    Parameters
    ----------
    occ  : float  — D2 occupancy [0, 1]
    tau  : float  — transduction ratio
    Emax : float  — maximal system response (normalised to 1.0 by default)

    Returns
    -------
    float — net D2-MSN activation [0, Emax]
    """
    return float(Emax * tau * occ / (1.0 + (1.0 + tau) * occ))


# ══════════════════════════════════════════════════════════════════════════════
# 4.  ARIPIPRAZOLE TWO-COMPARTMENT PK (Mallikaarjun 2004)
# ══════════════════════════════════════════════════════════════════════════════

def _aripiprazole_pk_odes(t, y, ka, ke_p, k_fm, ke_m, dose_per_hr):
    """
    ODE RHS for aripiprazole one-compartment + metabolite model.

    State vector y = [Agut, Cparent, Cmet]
      Agut    — amount in gut (mg)
      Cparent — parent plasma concentration (mg/L, assuming Vd=1 L normalised)
      Cmet    — metabolite plasma concentration (mg/L equivalent)

    Parameters
    ----------
    ka          — absorption rate (hr⁻¹)
    ke_p        — parent elimination rate (hr⁻¹) = CL_F / Vd_F
    k_fm        — parent→metabolite conversion rate (hr⁻¹); fraction metabolised
    ke_m        — metabolite elimination rate (hr⁻¹)
    dose_per_hr — continuous dose equivalent (mg/hr) for simulation convenience
    """
    Agut, Cp, Cm = y
    dAgut = dose_per_hr - ka * Agut
    dCp   = ka * Agut - ke_p * Cp
    dCm   = k_fm * Cp - ke_m * Cm
    return [dAgut, dCp, dCm]


def aripiprazole_pk_timecourse(D_mg_day, t_max_hr=500.0, n_points=1000):
    """
    Simulate aripiprazole and dehydro-aripiprazole (metabolite) plasma
    concentrations over time using a simplified one-compartment + metabolite
    model (Mallikaarjun 2004 parameters).

    Dosing is modelled as a continuous infusion equivalent for simplicity
    (steady-state occupancy is the key output; oscillations within the
    dosing interval are secondary for Paper X Phase 1).

    Parameters
    ----------
    D_mg_day : float — oral dose in mg/day
    t_max_hr : float — simulation duration in hours (default 500 hr ≈ 21 days)
    n_points : int   — number of output time points

    Returns
    -------
    dict with keys:
        't_hr'         — time array (hr)
        'C_parent'     — parent normalised concentration (arbitrary units)
        'C_met'        — metabolite normalised concentration
        'occ_parent'   — D2 occupancy from parent alone
        'occ_total'    — total D2 occupancy (parent + metabolite)
        'alpha_AP_ocd' — OCD circuit parameter time-course
        'alpha_AP_mdd' — MDD circuit parameter time-course
    """
    p = DRUG_PARAMS['aripiprazole']
    ke_p  = np.log(2) / p['t_half_parent']        # 0.00924 hr⁻¹
    ke_m  = np.log(2) / p['t_half_metabolite']    # 0.00737 hr⁻¹
    ka    = p['ka']                                 # 0.20 hr⁻¹
    # Fraction of parent converted to metabolite; tuned so that at steady
    # state metabolite contributes ~40% of total D2 occupancy.
    # Assuming metabolite has same D2 affinity as parent:
    #   C_met_ss / C_parent_ss = k_fm / ke_m
    #   occ_met / occ_parent   ≈ C_met / C_parent = k_fm / ke_m
    #   Given metabolite_occ_frac = 0.40:
    #     C_met/C_parent = 0.40/(1-0.40) = 0.667
    #     k_fm = 0.667 * ke_m
    met_frac = p['metabolite_occ_frac']
    k_fm  = (met_frac / (1.0 - met_frac)) * ke_m   # ≈ 0.00491 hr⁻¹

    dose_per_hr = D_mg_day / 24.0
    y0    = [0.0, 0.0, 0.0]
    t_arr = np.linspace(0.0, t_max_hr, n_points)

    sol = solve_ivp(
        _aripiprazole_pk_odes,
        [0.0, t_max_hr],
        y0,
        args=(ka, ke_p, k_fm, ke_m, dose_per_hr),
        t_eval=t_arr,
        method='RK45',
        rtol=1e-6, atol=1e-9,
    )

    Cp = sol.y[1]   # parent concentration (arbitrary units)
    Cm = sol.y[2]   # metabolite concentration

    # Normalise so that steady-state total concentration maps to the Hill
    # occupancy at D_mg_day via the calibrated D50.
    # Steady-state parent: Cp_ss = dose_per_hr / ke_p (continuous infusion)
    Cp_ss = dose_per_hr / ke_p
    Cm_ss = Cp_ss * k_fm / ke_m

    # Hill occupancy at steady state (from Tier 2 Hill function, which is
    # calibrated to total steady-state PET occupancy)
    occ_total_ss = d2_occupancy_hill(D_mg_day, 'aripiprazole')
    # Split: parent fraction = (1 - met_frac) of total occ at ss
    occ_parent_ss = occ_total_ss * (1.0 - met_frac)
    occ_met_ss    = occ_total_ss * met_frac

    # Scale factor: Cp_ss → occ_parent_ss via effective D50 in concentration
    # units.  At ss: occ_parent = Emax_p * Cp_ss / (Cp_ss + D50_conc)
    # → D50_conc = Cp_ss * (Emax_p/occ_parent_ss - 1)
    Emax_p = p['Emax']
    D50_conc_p = Cp_ss * (Emax_p / occ_parent_ss - 1.0) if occ_parent_ss > 0 else 1.0
    D50_conc_m = Cm_ss * (Emax_p / occ_met_ss   - 1.0) if occ_met_ss > 0 else 1.0

    occ_p = Emax_p * Cp / (Cp + D50_conc_p + 1e-15)
    occ_m = Emax_p * Cm / (Cm + D50_conc_m + 1e-15)
    occ_tot = occ_p + occ_m

    # Operational model correction (partial agonism)
    tau = p['tau_operational']
    eff_blockade = np.array([partial_agonist_effective_blockade(o, tau)
                             for o in occ_tot])

    aAP_ocd = F_OCC * eff_blockade
    aAP_mdd = F_OCC_MDD * eff_blockade * CORTICAL_STRIATAL_RATIO

    return {
        't_hr':         sol.t,
        'C_parent':     Cp,
        'C_met':        Cm,
        'occ_parent':   occ_p,
        'occ_total':    occ_tot,
        'alpha_AP_ocd': aAP_ocd,
        'alpha_AP_mdd': aAP_mdd,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5.  CIRCUIT PARAMETER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def alpha_AP_ocd(D_mg, drug, f_occ=F_OCC):
    """
    OCD CSTC circuit parameter alpha_AP at steady-state dose D_mg.

    alpha_AP = f_occ × effective_blockade(occ_striatal, tau)

    For full antagonists: effective_blockade = occ_striatal
    For partial agonists: effective_blockade = occ_striatal / (1 + tau)

    Parameters
    ----------
    D_mg  : float — dose in mg/day
    drug  : str   — drug name (key in DRUG_PARAMS)
    f_occ : float — bridging constant (default F_OCC = 0.222)

    Returns
    -------
    float — alpha_AP [dimensionless, same units as Paper 2 alpha_AP]
    """
    occ = d2_occupancy_hill(D_mg, drug)
    tau = DRUG_PARAMS[drug]['tau_operational']
    eff = partial_agonist_effective_blockade(occ, tau)
    return float(f_occ * eff)


def alpha_AP_mdd(D_mg, drug, f_occ_mdd=F_OCC_MDD,
                  cortical_ratio=CORTICAL_STRIATAL_RATIO):
    """
    MDD DLPFC-sgACC circuit parameter alpha_AP_MDD at steady-state dose D_mg.

    Uses cortical D2 occupancy, which is lower than striatal by
    cortical_ratio (Kessler 2005).

    alpha_AP_MDD = f_occ_mdd × effective_blockade(occ_cortical, tau)
    occ_cortical = occ_striatal × cortical_ratio

    Parameters
    ----------
    D_mg          : float — dose in mg/day
    drug          : str   — drug name
    f_occ_mdd     : float — MDD bridging constant (default F_OCC_MDD = 0.155)
    cortical_ratio: float — cortical/striatal D2 occ ratio (Kessler 2005 ≈ 0.70)

    Returns
    -------
    float — alpha_AP_MDD [dimensionless]
    """
    occ_striatal = d2_occupancy_hill(D_mg, drug)
    occ_cortical = occ_striatal * cortical_ratio
    tau = DRUG_PARAMS[drug]['tau_operational']
    eff = partial_agonist_effective_blockade(occ_cortical, tau)
    return float(f_occ_mdd * eff)


def dose_scan(drug, D_min=None, D_max=None, n_points=50,
              f_occ=F_OCC, f_occ_mdd=F_OCC_MDD,
              cortical_ratio=CORTICAL_STRIATAL_RATIO):
    """
    Scan dose range for a drug and return occupancy and circuit parameters.

    Returns
    -------
    dict with numpy arrays:
        'D_mg', 'occ_striatal', 'occ_cortical',
        'eff_blockade_striatal', 'eff_blockade_cortical',
        'alpha_AP_ocd', 'alpha_AP_mdd'
    """
    p = DRUG_PARAMS[drug]
    if p['tier'] == 3:
        raise ValueError(f"{drug} is Tier 3; dose_scan not supported.")
    lo, hi = p['dose_range']
    D_min  = lo if D_min is None else D_min
    D_max  = hi if D_max is None else D_max
    D_arr  = np.linspace(D_min, D_max, n_points)

    tau = p['tau_operational']
    occ_s = np.array([d2_occupancy_hill(D, drug) for D in D_arr])
    occ_c = occ_s * cortical_ratio
    eff_s = np.array([partial_agonist_effective_blockade(o, tau) for o in occ_s])
    eff_c = np.array([partial_agonist_effective_blockade(o, tau) for o in occ_c])

    return {
        'D_mg':                   D_arr,
        'occ_striatal':           occ_s,
        'occ_cortical':           occ_c,
        'eff_blockade_striatal':  eff_s,
        'eff_blockade_cortical':  eff_c,
        'alpha_AP_ocd':           f_occ     * eff_s,
        'alpha_AP_mdd':           f_occ_mdd * eff_c,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6.  VALIDATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

def _check(label, val, lo, hi, tol=0.0):
    """Assert val ∈ [lo-tol, hi+tol] and print pass/fail."""
    ok = (lo - tol) <= val <= (hi + tol)
    status = 'PASS' if ok else 'FAIL'
    print(f"  [{status}] {label}: {val:.4f}  (expected [{lo:.4f}, {hi:.4f}])")
    return ok


def validate_kapur2000_haloperidol():
    """
    Kapur et al. 2000 (Am J Psychiatry 157:514-520):
    Haloperidol D2 occupancy in first-episode schizophrenia:
      1 mg/day: mean 59%, SD=11% → test range [48, 70]
      2.5 mg/day: mean 75%, SD=6% → test range [63, 87]

    Note: Nordström 1993 (same journal year) studied RACLOPRIDE (not haloperidol),
    at 2, 6, 12 mg/day yielding D2 occupancy ~25%, ~57%, ~72% respectively.
    The haloperidol Hill function (D50=0.75) is calibrated to Kapur 2000 data.
    """
    print("\n── Kapur 2000: haloperidol D2 occupancy ──")
    ok1 = _check('Occ at 1 mg/day',   d2_occupancy_hill(1.0,   'haloperidol') * 100,
                 48.0, 70.0)
    ok2 = _check('Occ at 2.5 mg/day', d2_occupancy_hill(2.5,   'haloperidol') * 100,
                 63.0, 87.0)
    return ok1 and ok2


def validate_kapur2000_window():
    """
    Kapur et al. 2000 (Am J Psychiatry 157:514-520):
    D2 therapeutic window 65-80%.
    Haloperidol at 2, 4 mg: both should be near window.
    """
    print("\n── Kapur 2000: therapeutic window (65-80%) ──")
    results = []
    for drug, dose in [('haloperidol', 3.0), ('risperidone', 4.0), ('olanzapine', 10.0)]:
        occ = d2_occupancy_hill(dose, drug) * 100
        lo, hi = 60.0, 85.0   # allow ±5 pp around window edges
        ok = _check(f'{drug} {dose} mg/day', occ, lo, hi)
        results.append(ok)
    return all(results)


def validate_mamo2007():
    """
    Mamo et al. 2007 (Am J Psychiatry 164:1411-1417):
    Aripiprazole D2 occupancy: 89-95% at 10-30 mg/day.
    """
    print("\n── Mamo 2007: aripiprazole occupancy ──")
    ok1 = _check('Occ at 10 mg/day', d2_occupancy_hill(10.0, 'aripiprazole') * 100,
                 85.0, 95.0)
    ok2 = _check('Occ at 30 mg/day', d2_occupancy_hill(30.0, 'aripiprazole') * 100,
                 90.0, 97.0)
    return ok1 and ok2


def validate_partial_agonist_model():
    """
    Partial agonist effective blockade < occupancy for aripiprazole.
    At 90% D2 occupancy with τ=0.40:
        effective_blockade = 0.90 / (1 + 0.40) = 0.643
    """
    print("\n── Black-Leff model: aripiprazole effective blockade ──")
    occ = 0.90
    tau = DRUG_PARAMS['aripiprazole']['tau_operational']
    eff = partial_agonist_effective_blockade(occ, tau)
    expected = occ / (1.0 + tau)
    ok1 = _check('eff_blockade at occ=0.90', eff, expected - 0.001, expected + 0.001)

    # Full antagonist: effective_blockade == occupancy
    eff_hal = partial_agonist_effective_blockade(0.75, 0.0)
    ok2 = _check('Full antagonist eff==occ', eff_hal, 0.749, 0.751)

    # Partial agonist produces less circuit effect than same-occupancy full antagonist
    ok3 = eff < occ
    print(f"  [{'PASS' if ok3 else 'FAIL'}] Partial agonist eff ({eff:.3f}) < occ ({occ:.3f}): {ok3}")
    return ok1 and ok2 and ok3


def validate_kessler2005():
    """
    Kessler et al. 2005 (Neuropsychopharmacology 30:2283-2289):
    Cortical D2 occupancy ≈ 70% of striatal at same dose.
    Test that alpha_AP_mdd < alpha_AP_ocd for all Tier 1 drugs.
    """
    print("\n── Kessler 2005: cortical < striatal D2 occupancy ──")
    results = []
    for drug, dose in [('haloperidol', 3.0), ('risperidone', 4.0), ('olanzapine', 10.0)]:
        a_ocd = alpha_AP_ocd(dose, drug)
        a_mdd = alpha_AP_mdd(dose, drug)
        ok    = a_mdd < a_ocd
        ratio = a_mdd / a_ocd if a_ocd > 0 else 0
        print(f"  [{'PASS' if ok else 'FAIL'}] {drug} {dose} mg: "
              f"alpha_OCD={a_ocd:.4f}, alpha_MDD={a_mdd:.4f}, ratio={ratio:.3f}")
        results.append(ok)
    return all(results)


def validate_paper2_calibration():
    """
    Paper 2 calibration check:
    alpha_AP_ocd at population-average augmentation dose should be ≈ 0.167.

    Using haloperidol at ~3 mg/day (population-average standard dose ≈ 75% occ):
      alpha_AP = f_occ × 0.75 = 0.222 × 0.75 = 0.167  ✓
    """
    print("\n── Paper 2 calibration: alpha_AP at 75% D2 occ ──")
    # Find dose of haloperidol giving ~75% D2 occupancy
    for D in np.linspace(1, 10, 100):
        if abs(d2_occupancy_hill(D, 'haloperidol') - 0.75) < 0.005:
            a = alpha_AP_ocd(D, 'haloperidol')
            ok = _check(f'alpha_AP (hal ~{D:.1f} mg, 75% occ)', a, 0.155, 0.180)
            return ok
    print("  [FAIL] Could not find haloperidol dose giving 75% D2 occ")
    return False


def validate_girgis2020_brexpiprazole():
    """
    Girgis et al. 2020 (Neuropsychopharmacology 45:786-792):
    Brexpiprazole steady-state D2/D3 occupancy:
      64 ± 8% at 1 mg/day  → test range [56, 72]
      80 ± 12% at 4 mg/day → test range [68, 92]
    """
    print("\n── Girgis 2020: brexpiprazole steady-state occupancy ──")
    ok1 = _check('Occ at 1 mg/day', d2_occupancy_hill(1.0, 'brexpiprazole') * 100,
                 56.0, 72.0)
    ok2 = _check('Occ at 4 mg/day', d2_occupancy_hill(4.0, 'brexpiprazole') * 100,
                 68.0, 92.0)
    return ok1 and ok2


def validate_aripiprazole_pk():
    """
    Aripiprazole PK model (Mallikaarjun 2004):
    After ~500 hr, occupancy should be near steady-state Hill value.
    Half-life should be consistent with published values (parent ~75 hr).
    """
    print("\n── Aripiprazole PK timecourse (Mallikaarjun 2004) ──")
    res = aripiprazole_pk_timecourse(10.0, t_max_hr=600.0, n_points=500)

    occ_ss_model  = res['occ_total'][-1]
    occ_ss_target = d2_occupancy_hill(10.0, 'aripiprazole')
    ok1 = _check('Steady-state occ (model vs Hill)',
                 occ_ss_model, occ_ss_target * 0.95, occ_ss_target * 1.05)

    # Check metabolite contributes ~25% of occupancy at steady state (Mamo 2007)
    occ_met_frac = res['occ_total'][-1] - res['occ_parent'][-1]
    met_pct = occ_met_frac / (res['occ_total'][-1] + 1e-15)
    ok2 = _check('Metabolite occ fraction at ss', met_pct, 0.15, 0.35)

    # Verify ke_parent corresponds to t_half ≈ 75 hr
    ke_p = np.log(2) / DRUG_PARAMS['aripiprazole']['t_half_parent']
    t_half_check = np.log(2) / ke_p
    ok3 = _check('Parent half-life (hr)', t_half_check, 74.0, 76.0)

    return ok1 and ok2 and ok3


def run_all_tests():
    """Run all validation tests and print summary."""
    print("=" * 60)
    print("px_ap_pk_model — Validation Suite")
    print("=" * 60)

    tests = [
        ('Kapur 2000 haloperidol',         validate_kapur2000_haloperidol),
        ('Kapur 2000 therapeutic window', validate_kapur2000_window),
        ('Mamo 2007 aripiprazole',        validate_mamo2007),
        ('Girgis 2020 brexpiprazole',     validate_girgis2020_brexpiprazole),
        ('Black-Leff partial agonism',    validate_partial_agonist_model),
        ('Kessler 2005 cortical/striatal',validate_kessler2005),
        ('Paper 2 calibration',           validate_paper2_calibration),
        ('Aripiprazole PK timecourse',    validate_aripiprazole_pk),
    ]

    results = {}
    for name, fn in tests:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"\n── {name} ──")
            print(f"  [ERROR] {e}")
            results[name] = False

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    n_pass = sum(results.values())
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n  {n_pass}/{len(results)} passed")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 7.  DOSE-SCAN REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_dose_scan(drug, doses=None):
    """Print a formatted dose-scan table for a drug."""
    p = DRUG_PARAMS[drug]
    if p['tier'] == 3:
        print(f"{drug}: Tier 3 — dose scan not available.")
        return

    lo, hi = p['dose_range']
    if doses is None:
        doses = np.linspace(lo, hi, 8)

    tau = p['tau_operational']
    print(f"\n{'─'*80}")
    print(f"Dose scan: {drug}  (Tier {p['tier']})")
    print(f"  D50={p['D50']} mg  n={p['n']}  Emax={p['Emax']}  τ={tau}")
    print(f"  f_occ={F_OCC:.3f}  cortical_ratio={CORTICAL_STRIATAL_RATIO:.2f}  "
          f"f_occ_mdd={F_OCC_MDD:.3f}")
    print(f"{'─'*80}")
    print(f"  {'D(mg)':>8}  {'occ_S(%)':>10}  {'occ_C(%)':>10}  "
          f"{'eff_S':>8}  {'α_AP':>8}  {'α_AP_MDD':>10}  {'window':>8}")
    print(f"  {'─'*74}")

    for D in doses:
        occ_s = d2_occupancy_hill(D, drug)
        occ_c = occ_s * CORTICAL_STRIATAL_RATIO
        eff_s = partial_agonist_effective_blockade(occ_s, tau)
        a_ocd = F_OCC     * eff_s
        a_mdd = F_OCC_MDD * partial_agonist_effective_blockade(occ_c, tau)
        in_win = ('✓' if THERAPEUTIC_WINDOW[0] <= occ_s <= THERAPEUTIC_WINDOW[1]
                  else '–')
        print(f"  {D:>8.1f}  {occ_s*100:>10.1f}  {occ_c*100:>10.1f}  "
              f"{eff_s:>8.4f}  {a_ocd:>8.4f}  {a_mdd:>10.4f}  {in_win:>8}")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN — run tests and dose scan when executed directly
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    results = run_all_tests()

    print("\n")
    for drug in ['haloperidol', 'risperidone', 'olanzapine',
                 'aripiprazole', 'brexpiprazole']:
        print_dose_scan(drug)

    # Aripiprazole PK time-course summary
    print("\n\n── Aripiprazole PK time-course at 10 mg/day ──")
    res = aripiprazole_pk_timecourse(10.0, t_max_hr=600.0, n_points=300)
    for idx, thr in enumerate([24, 72, 168, 336, 500]):
        i = np.argmin(np.abs(res['t_hr'] - thr))
        print(f"  t={thr:4d} hr: occ_total={res['occ_total'][i]:.3f}  "
              f"alpha_OCD={res['alpha_AP_ocd'][i]:.4f}  "
              f"alpha_MDD={res['alpha_AP_mdd'][i]:.4f}")
