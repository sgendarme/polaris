# Archived result files

JSON artifacts backing every number reported in the paper. All files cover the
20 reported seeds unless stated otherwise. Older artifacts label POLARIS by its
internal codename `CLIRE`.

```
main_test20_ml-100k.json        Table III ML-100K rows (incl. MeLU, broadened LightFM), K in {10,20,50}
main_test20_ml-1m.json          Table III ML-1M rows (POLARIS row = reranker_ml-1m.json;
                                LightFM row superseded by lightfm_broadened_ml-1m.json)
main_test20_ml-1m_fast.json     ML-1M without learned baselines, K in {10,20,50}
main_test20_comoda_strict.json  Table III LDOS rows (incl. MeLU), K in {10,20,50}
main_ldos_strict_suite.json     LDOS controls, ablations and regime splits
melu_test20_ml-1m.json          MeLU rows (ML-1M)
reranker_ml-1m.json             deployed ML-1M POLARIS+RR metrics
paired_significance.json        paired statistics behind Table III's bold markers
                                (mean, 95% CI, win/loss, Holm; 63 comparisons)
lightfm_broadened_*.json        LightFM at the broadened (d, lr, epochs) selection
controls_softpop_ensemble_*.json  SoftClusterPopularity and no-pattern ensemble (VI.C)
patternonly_tuned_*.json        separately tuned PatternOnly (VI.C)
ablations_*.json                alpha=0 / eta=0 / ensemble ablations + split stats
regime_random_*.json            random cold-user splits (VI.B)
regime_low_activity_*.json      low-activity cold-user splits (VI.B)
alpha_regime_sweep_ml-100k.json alpha revalidation under light-rater regimes (VI.B)
cap20_ml-1m.json, cap20_ml-100k.json, cap20_comoda_strict.json  candidate cap N=20
popularity_concentration_ml-100k.json  head concentration and tail-hit shares
probes_*.json                   Table IV reliance probes; *_matched = activity-matched placebo
stability_bootstrap_*.json      bootstrap LFI stability (positive-token Jaccard, VI.D)
pattern_share_ml-100k.json      pattern share of the top recommendation (median, VI.A)
context_diagnostic_comoda.json  LDOS context-profile diagnostic (V.A)
pattern_popularity_redundancy.json  LFI token popularity percentiles (VI.D)
```

LDOS files use the strict demographics-only setting except
context_diagnostic_comoda.json.
