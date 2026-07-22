# POLARIS: paper reproduction package

Code and result artifacts for "POLARIS: Patterns over Local Cohorts for
Auditable Recommendation in Strict Cold Start".

```
conf/params.json   frozen validation-selected configurations (Table II)
src/polaris/       method, baselines, protocol, metrics
scripts/           one entry point per paper artifact
results/           archived JSON results backing every reported number
resources/         dataset directory (not distributed; see resources/README.md)
outputs/           created by the scripts
```

## Installation

Python 3.11+.

```
python -m venv .venv && source .venv/bin/activate
pip install -e ".[full]"
bash scripts/setup_lightfm_env.sh python3.11   # separate env for LightFM
```

Reference environment: Python 3.14.0, numpy 2.5.0, pandas 3.0.3,
scikit-learn 1.9.0, scipy 1.18.0, torch 2.12.1, lightgbm 4.6.0. The pipeline is
seeded end to end; identical versions reproduce the reported numbers exactly.

Download the datasets as described in [resources/README.md](resources/README.md).

## Reproducing the paper

All scripts run from the repository root and write to `outputs/`.
`POLARIS_QUIET=1` silences progress logs.

Main results (Table III) and paired statistics:

```
python scripts/run_reranker.py ml-1m 20 lambdamart   # deployed ML-1M set selector (run first)
python scripts/run_main.py all 20                    # 20-seed runs, all methods, K in {10,20,50}
python scripts/make_significance.py                  # paired Wilcoxon + Holm (bold markers)
python scripts/run_validation.py ml-100k POLARIS     # audit the nested selection (per method/dataset)
```

`run_main.py <ds> 20 fast` skips the learned baselines and reproduces the
POLARIS rows in minutes.

Reliance probes (Table IV):

```
python scripts/probes_lolo_lsuo.py all 20 10     # LOLO dropout + LSUO excess
python scripts/probes_sufficiency.py all 20 10   # sufficiency excess
python scripts/probes_anchoring.py all 20 10     # anchoring@10 excess
python scripts/run_matched_placebo.py ml-100k 20 # activity-matched LSUO placebo
```

`POLARIS_ANCHOR_RHO=0` reruns the anchoring probe on raw participation instead
of the deployed popularity-adjusted contribution (Sec. VI.D contrast).

Robustness, controls and ablations (Sec. VI.B-VI.C):

```
python scripts/run_cap20.py all 20               # candidate cap N=20
python scripts/run_robust_splits.py all 20       # random and low-activity regimes
python scripts/run_controls.py ml-100k 20        # SoftClusterPop + no-pattern ensemble (also ml-1m)
python scripts/run_ablations.py 20               # alpha=0 / eta=0 ablations
python scripts/run_patternonly_tuned.py ml-100k 20   # separately tuned PatternOnly
```

Additional reported quantities (Sec. V.A, VI.A, VI.D):

```
python scripts/run_ldos_context_diagnostic.py 20     # LDOS context-profile diagnostic
python scripts/run_pattern_share.py ml-100k 20       # pattern share of the top score
python scripts/run_stability_bootstrap.py ml-100k 20 # LFI stability (Jaccard)
python scripts/explain_user.py --dataset ml-100k --seed 1 --user 889 --why-not Titanic  # Fig. 1
```
