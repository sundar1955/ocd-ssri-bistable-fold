# Reproducibility bundle — "Does SSRI treatment of OCD cross a fold?"

This bundle regenerates the calibration, analyses, and figures of the manuscript. All code is Python.

## Environment
See `ENVIRONMENT.txt`. Tested with Python 3.13, NumPy 2.2, SciPy 1.17, Matplotlib 3.11 (any recent versions
should work). No non-standard dependencies.

```
pip install numpy scipy matplotlib
```

## Data
- `data/digitized_trajectories.csv` — the calibration dataset: net (drug − placebo) Y-BOCS improvement per
  week, 106 points across 15 arms (13 SSRI arms fitted; 2 clomipramine + 1 sertraline held out). Columns:
  drug, trial, dose_mg, Y0_mean, Y0_sd, week, net_dYBOCS, SE, role. Regenerate with `data/make_dataset.py`.
  Provenance: digitized from the cited trial figures/tables (Supplement §"Digitized trajectory data").
- `data/best_fit.json` — the locked calibrated constants (theta_M, b, tau_G, kappa_des, gamma, h_max, eC_h)
  with the b confidence band. This is the canonical parameter set consumed by every downstream script.

## Model modules (imported by the scripts below)
- `code/var_circuit_validation.py` — van Albada–Robinson 9-population mean-field circuit + validation.
- `code/build_fit_ssri_panel.py` — circuit fixed-point solver as a function of G_CS.
- `code/gcs_bistable_population.py` — baseline-referenced BCM plasticity source S(G).
- `code/gamma_profile_calib.py` — readout Y = 40 p^gamma and per-patient alpha placement.
- `code/gcs_bistable_calibrate.py`, `code/build_fit_gcs_slow.py` — serotonin engine + calibration objective.

## Regenerating the results (map of manuscript item → script)
| Manuscript item | Script (in `code/`) | Output |
|---|---|---|
| Calibration / best_fit.json | `build_fit_gcs_slow.py` (fit); constants in `data/best_fit.json` | fitted b, kappa_des |
| Fig. 1 model chain (schematic) | hand-drawn in LaTeX/TikZ — no script | fig_model_chain.pdf |
| Fig. 2 readout + landscape | `fig_phi_vs_gcs.py` | fig_phi_vs_gcs.pdf |
| Fig. 3 plasticity gain α (vulnerability) | `fig_alpha_role.py` | fig_alpha_role.pdf |
| Fig. 4 calibration (six SSRIs) | `fig3_calibration.py` | fig3_calibration.pdf |
| Fig. 5 phase lines (treatment across the fold) | `fig4_phaseline.py` | fig4_phaseline.pdf |
| Fig. 6 graded vs fold signatures | `graded_vs_fold_signatures.py` | fig_graded_vs_fold.pdf, signatures_summary.json |
| Fig. 7 combination (routes to the fold) | `combo_fold.py` | fig_combo_fold.pdf, fig_combo_mono.pdf |
| SI bifurcation diagram | `fig_bifurcation.py` | fig_bifurcation.pdf |
| SI dose-sweep BC/VR | `graded_vs_fold_dosesweep.py` | fig_dosesweep_BC_VR.pdf |
| SI serotonin ceiling | `fig_disc_serotonin_ceiling.py` | fig_disc_serotonin_ceiling.pdf |
| Graded-twin fit (chi2 6.3 vs 7.6, AIC) | `graded_twin_fit.py` | graded_twin_fit.json |
| Discriminating signatures (BC/VR, jump, hysteresis) | `graded_vs_fold_signatures.py` | signatures_summary.json |
| Detectability / sample size (N≈30–50) | `graded_vs_fold_detectability.py` | graded_vs_fold_detectability.json |
| Prediction intervals (u_fold, remission) | `wp3_prediction_intervals.py` | wp3_intervals.json |
| Leave-one-arm-out CV + bootstrap | `wp3_cv.py` | wp3_cv.json |
| Source-form + gamma×theta_M robustness | `wp567_sensitivity.py`, `wp5_altsource.py` | wp567_out.txt, wp5_altsource.json |
| Clomipramine specificity check | `wp10_clomipramine.py` | wp10_cmi.json |
| Cohen severity head-to-head | `cohen_response_vs_severity.py` | (stdout / figure) |

Random seeds are set inside the stochastic scripts (e.g. `graded_vs_fold_detectability.py` uses
`np.random.default_rng(11)`; the cohort simulations use fixed seeds) so results are deterministic.

## Notes / caveats (as stated in the manuscript)
- Some digitized points were pending source-PDF verification at submission; see the manuscript's data-availability
  statement. The `role` column marks fitted vs held-out arms.
- The bimodality/variability-ratio discriminator is demonstrated here on **model-simulated cohorts**
  (`graded_vs_fold_detectability.py`); it is a prospective prediction, not run on patient data.

## Status
Staged for deposit. Before public release: (1) finalize source-verification of the digitized points;
(2) choose hosting (a private reviewer link during review + a Zenodo DOI on acceptance is standard for NC).
