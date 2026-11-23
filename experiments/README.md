# Experiments

Scripts for the results of the revised paper. Every method uses the same contracts and the same
five stratified folds (`folds/`). Contracts with identical path databases are in the same fold.

## Setup

```
pip install -r requirements.txt
export SMARTTAINTRL_DATA=/path/to/data        # default: <repo>/data
```

The expected layout of the data folder is described at the top of `config.py`. Results are written to
`<repo>/results` (change it with `SMARTTAINTRL_RESULTS`). Run every script from the repository root.

`folds/` holds the fold of every contract (`folds_<set>.json`, and the same as a table in
`contracts_<set>.csv`). `models/` holds the ten trained agents (`dqn_<set>_fold<k>.pth`, trained on all
folds except k) and the model used for localization (`trained_dqn_model_2500.pth`).

## Paper results

| Result | Script | Needs |
|---|---|---|
| Training (one model per fold) | `cv/kfold_eval.py <set>` | data |
| Fig. 9, pruning during training | `cv/train_with_pruning_log.py <set>` | data |
| Table 7, path level | `cv/score_test_folds.py <set>` | models |
| Agent decisions for the contract level | `contract_level/dump_agent.py <set> <k>` (k = 1..5) | models |
| Table 8, SmartTaintRL | `contract_level/contract_head.py <set>` | agent dumps |
| Keyword-count baseline | `contract_level/keyword_baseline.py <set>` | sources |
| Slither | `tools/run_slither.py <set>` (`--solc 0.4.24` for a fixed version) | Slither 0.11.6 |
| Mythril (balanced) | `tools/run_mythril.py balanced` | Mythril 0.24.8 |
| RNVulDet | `tools/rnvuldet_compile.py`, then `tools/run_rnvuldet.py <set>` | RNVulDet, `RNVULDET_DIR` |
| TaintSentinel | `tools/run_taintsentinel_folds.py <set>` | TaintSentinel, `TAINTSENTINEL_DIR` |
| Table 8 and appendix table, with p-values | `analysis/table8_stats.py` | the outputs above |
| Table 9, context test | `analysis/context_test.py` | contract head + Slither (imbalanced) |
| Mutation test | `analysis/mutation_test.py` | Slither |
| Table 10, path-level baselines | `baselines/path_baselines.py <set>` | `cv/score_test_folds.py` output |
| Fig. 2, source shares on the training folds | `analysis/source_prevalence.py` | data |
| Source priorities, uniform / perturbed | `analysis/priority_sensitivity.py uniform\|perturbed` | data (retrains 5 models) |
| Random part of the pool | `analysis/refill_seed_test.py` | models |
| Pruning-recall trade-off (appendix) | `analysis/pruning_recall.py` | models |
| Attention deletion test | `analysis/attention_deletion.py` | models |
| Gradient vs Integrated Gradients | `analysis/ig_vs_gradient.py` | models |
| Localization, Tables 12 and 13 | `src/localization/test_localization.py [--ablation]` | localization data |
| Propagation decay | `analysis/decay_sweep.py` | localization data |
| Run time | `analysis/time_agent.py`, `analysis/time_pipeline.py` | models, TaintSentinel preprocessing |

`<set>` is `balanced` or `imbalanced`. A failed or timed-out tool run counts as "safe".

The contract-level decision follows `contract_level/PROTOCOL.md`, which was fixed before any test fold
was scored.

## Reproducibility notes

- The initial pool of an episode is drawn at random. The scripts fix this draw, but the runs in the paper
  did not always do so; a rerun of `cv/score_test_folds.py` can therefore differ in the third decimal
  (for the balanced set only the accuracy changes, 0.942 instead of 0.941).
- `contract_level/dump_agent.py` was seeded in the paper as well, so it gives the same agent decisions,
  and `contract_head.py` then gives the same predictions as in Table 8.
- In Table 13, the centrality-only and propagation-only rows depend on the order of tied scores, which
  follows `PYTHONHASHSEED`. `PYTHONHASHSEED=14` gives the values of the paper; the other rows do not change.
- On the laptop CPU of the paper, training took about 30 minutes on the balanced set and about 3 hours on the imbalanced set (Table 5).
