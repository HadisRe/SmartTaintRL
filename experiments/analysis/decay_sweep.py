"""
Node-level localization against the propagation decay (paper value 0.7), on the 14 ground-truth
contracts, full configuration (all four weights = 1). (Section 6, Reviewer 1 comment 12.)

Usage:  python experiments/analysis/decay_sweep.py
Output: results/analysis/decay_sweep.json
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
sys.path.insert(0, str(config.REPO / "src" / "localization"))
from test_localization import evaluate_configuration, load_ground_truth, DEFAULT_GT_PATH
from localizer import VulnerabilityLocalizer

gt = load_ground_truth(DEFAULT_GT_PATH)
res = {}
for d in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    loc = VulnerabilityLocalizer(str(config.LOC_MODEL), propagation_decay=d)
    s = evaluate_configuration(loc, gt, config.NODE_DB, config.SEMANTIC_GRAPHS, verbose=False)['summary']
    res[d] = s; print(d, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}, flush=True)
json.dump({str(k): v for k, v in res.items()}, open(config.out("analysis", "decay_sweep.json"), "w"), indent=1, default=str)
