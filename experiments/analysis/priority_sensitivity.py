"""
Sensitivity to the source priorities of Eq. 6 (Section 4.2.3), balanced set.

The agent is retrained on the same folds, seeds and loop as kfold_eval.py, with the source values
of the importance score replaced:
  uniform    every source 0.64
  perturbed  timestamp 1.0, blocknumber 0.883, blockhash 0.678, sender 0.452, other 0.301
             (the paper values 0.9 / 0.8 / 0.7 / 0.5 / 0.3, each moved at random by at most 20%)
The change is applied to a copy of env4_environment loaded in memory; the file in src/ is not touched.
Each setting trains five models (2,500 episodes each), so this takes several hours on a CPU.

Usage:  python experiments/analysis/priority_sensitivity.py uniform|perturbed
Output: results/analysis/priority_<setting>/fold<k>_eval.json
"""
import os, re, sys, json, random, types, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cv"))
SETTING = sys.argv[1]; sys.argv = [sys.argv[0]]
import config

PAPER = {"0.9": 0, "0.8": 1, "0.7": 2, "0.5": 3, "0.3": 4}
NEW = {"uniform": ["0.64"] * 5, "perturbed": ["1.0", "0.883", "0.678", "0.452", "0.301"]}[SETTING]

# load env4 with the new source values, before anything imports the original
src = open(config.REPO / "src" / "rl_agent" / "env4_environment.py").read()
i = src.index("        source_risk = 0.0\n") + len("        source_risk = 0.0\n"); j = src.index("# Sink scoring", i)
block = re.sub(r"source_risk = (0\.\d)\b", lambda m: "source_risk = " + NEW[PAPER[m.group(1)]], src[i:j])
assert block.count("source_risk = ") == 9
mod = types.ModuleType("env4_environment"); mod.__file__ = "env4_environment.py"
exec(compile(src[:i] + block + src[j:], "env4_environment.py", "exec"), mod.__dict__)
sys.modules["env4_environment"] = mod

import kfold_eval as v2

cfg = v2.DATASETS["balanced"]; base = v2.ContractDataLoader(path_db_dir=cfg['path_db_dir'], profile_dir=cfg['profile_dir'])
allc = {c[0]: c for c in base.get_valid_contracts(min_paths=5)}; base.contracts = {}
folds = [[allc[a] for a in f] for f in config.folds("balanced")]
f1 = []
for k in range(5):
    out = config.out("analysis", f"priority_{SETTING}", f"fold{k+1}_eval.json")
    if not out.exists():
        train = [c for i, f in enumerate(folds) if i != k for c in f]
        torch.manual_seed(v2.SEED + k); random.seed(v2.SEED + k); np.random.seed(v2.SEED + k)
        agent = v2.train_on_fold(base, train, cfg['episodes'])
        s = v2.summarize(v2.evaluate_on_fold(base, agent, folds[k]))
        json.dump(dict(summary=s), open(out, "w"))
    s = json.load(open(out))["summary"]; f1.append(s['path_level']['f1'])
    print(k + 1, round(s['path_level']['f1'], 4), round(s['pruning_of_all_paths'], 4), flush=True)
print(f"{SETTING}: path F1 {np.mean(f1):.3f} +- {np.std(f1, ddof=1):.3f}")
