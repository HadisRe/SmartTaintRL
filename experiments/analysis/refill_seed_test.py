"""
Does the random part of the pool change the results? (Section 4.2.4, Reviewer 1 comment 2)
The balanced models are evaluated greedily on their test folds with five seeds. No training.

Usage:  python experiments/analysis/refill_seed_test.py
Output: results/analysis/refill_seed_test_balanced.json
"""
import os, sys, json, random, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cv"))
sys.argv = [sys.argv[0]]
import config
import kfold_eval as v2
cfg = v2.DATASETS["balanced"]; base = v2.ContractDataLoader(path_db_dir=cfg['path_db_dir'], profile_dir=cfg['profile_dir'])
allc = {c[0]: c for c in base.get_valid_contracts(min_paths=5)}; base.contracts = {}
folds = [[allc[a] for a in f] for f in config.folds("balanced")]
res = {}
for seed in [0, 1, 2, 3, 4]:
    f1, pr = [], []
    for k in range(5):
        agent = v2.DQNAgent(learning_rate=v2.LR, gamma=v2.GAMMA, epsilon_decay=v2.EPS_DECAY, epsilon_end=v2.EPS_END)
        agent.q_network.load_state_dict(torch.load(config.model("balanced", k + 1), map_location="cpu")); agent.q_network.eval()
        random.seed(1000 + seed); np.random.seed(1000 + seed); torch.manual_seed(1000 + seed)
        s = v2.summarize(v2.evaluate_on_fold(base, agent, folds[k])); f1.append(s['path_level']['f1']); pr.append(s['pruning_of_all_paths'])
    res[seed] = dict(f1=float(np.mean(f1)), pruning=float(np.mean(pr)), f1_folds=f1, pruning_folds=pr)
    print(seed, round(res[seed]['f1'], 4), round(res[seed]['pruning'], 4), flush=True)
F = [r['f1'] for r in res.values()]; P = [r['pruning'] for r in res.values()]
print("F1 across seeds: mean %.4f sd %.4f range %.4f-%.4f" % (np.mean(F), np.std(F), min(F), max(F)))
print("pruning across seeds: mean %.4f sd %.4f range %.4f-%.4f" % (np.mean(P), np.std(P), min(P), max(P)))
json.dump(res, open(config.out("analysis", "refill_seed_test_balanced.json"), "w"), indent=1)
