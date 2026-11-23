"""
Path-level results of Table 7 (and the input of the path-level baselines, Table 10).

For each fold k the saved model dqn_<set>_fold<k>.pth decides, greedily, on every path
of every contract (no step limit). Test contracts of fold k give the reported scores;
the other contracts are kept because the contract-level decision is trained on them.

The initial pool of each contract is drawn at random. The runs in the paper did not fix this
draw, so a rerun can differ in the third decimal (balanced, checked: accuracy 0.942 vs 0.941,
all other Table 7 values equal). The seed below makes reruns of this script identical.

Usage:  python experiments/cv/score_test_folds.py balanced|imbalanced
Output: results/agent_rows/<set>_fold<k>.json
"""
import os, sys, json, random, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NAME = sys.argv[1] if len(sys.argv) > 1 else "balanced"; sys.argv = [sys.argv[0]]
import config
import kfold_eval as v2
from env2_pool_manager import PoolManager

v2.MAX_STEPS = 10**7; cfg = v2.DATASETS[NAME]
base = v2.ContractDataLoader(path_db_dir=cfg['path_db_dir'], profile_dir=cfg['profile_dir'])
allc = {c[0]: c for c in base.get_valid_contracts(min_paths=5)}; base.contracts = {}
folds = config.folds(NAME)
for k in range(5):
    of = config.out("agent_rows", f"{NAME}_fold{k+1}.json")
    if of.exists(): print("fold", k + 1, "done"); continue
    random.seed(v2.SEED + k); np.random.seed(v2.SEED + k)
    agent = v2.DQNAgent(learning_rate=v2.LR, gamma=v2.GAMMA, epsilon_decay=v2.EPS_DECAY, epsilon_end=v2.EPS_END)
    agent.q_network.load_state_dict(torch.load(config.model(NAME, k + 1), map_location="cpu")); agent.q_network.eval()
    cl = [allc[a] for f in folds for a in f if a in allc]
    loader = v2.FoldDataLoader(base, cl)
    env = v2.BadRandomnessEnv(data_loader=loader, pool_manager=PoolManager(), state_builder=v2.StateBuilder(), debug_mode=False)
    env.max_steps = 10**7   # decide every path
    agent.steps_done = 10_000_000; agent.epsilon_start = 0.0; agent.epsilon_end = 0.0; rows = []
    for j, (addr, n_paths, label) in enumerate(cl):
        loader.only = addr; s, info = env.reset()
        r = dict(address=addr, label=label, n_paths=n_paths, analyzed=0, skipped=0, tp=0, fp=0, tn=0, fn=0, qgap_max=-1e9, qgap_sum_pos=0.0)
        done = False
        while not done:
            v = len(env.pool_state.current_pool)
            if v == 0: break
            with torch.no_grad(): q = agent.q_network(torch.FloatTensor(s).unsqueeze(0))[0][:min(v, 20)]
            gap = (q[:, 0] - q[:, 1]).numpy(); r['qgap_max'] = max(r['qgap_max'], float(gap.max()))
            pi, at = agent.select_action(s, valid_paths=v)
            risk = env.pool_state.current_pool[pi]['basic_info'].get('risk_level') if pi < v else None
            if at == 'ANALYZE' and pi < len(gap): r['qgap_sum_pos'] += float(max(gap[pi], 0))
            s, _, done, info = env.step((pi, at))
            if at == 'ANALYZE': r['analyzed'] += 1
            else: r['skipped'] += 1
            # path-level counts: HIGH analyzed = TP, HIGH skipped = FN, LOW analyzed = FP, LOW skipped = TN
            if risk == 'HIGH': r['tp' if at == 'ANALYZE' else 'fn'] += 1
            elif risk == 'LOW': r['tn' if at == 'SKIP' else 'fp'] += 1
        base.loaded_contracts.pop(addr, None); rows.append(r)
        if j % 500 == 0: print(f"fold {k+1}: {j}/{len(cl)}", flush=True)
    json.dump(rows, open(of, "w"))
    te = [r for r in rows if r['address'] in set(folds[k])]
    tp, fp, fn = (sum(r[x] for r in te) for x in ('tp', 'fp', 'fn')); P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    dec = sum(r['analyzed'] + r['skipped'] for r in te)
    print(f"fold {k+1}: test path F1 {2*P*R/max(P+R,1e-9):.3f}  pruning {sum(r['skipped'] for r in te)/max(dec,1):.3f}", flush=True)
# Table 7: mean and standard deviation over the five test folds
M = []
for k in range(5):
    te = [r for r in json.load(open(config.out("agent_rows", f"{NAME}_fold{k+1}.json"))) if r['address'] in set(folds[k])]
    tp, fp, fn, tn = (sum(r[x] for r in te) for x in ('tp', 'fp', 'fn', 'tn'))
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    M.append([(tp + tn) / max(tp + tn + fp + fn, 1), P, R, 2 * P * R / max(P + R, 1e-9),
              sum(r['skipped'] for r in te) / max(sum(r['analyzed'] + r['skipped'] for r in te), 1)])
M = np.array(M)
for i, m in enumerate(["accuracy", "precision", "recall", "F1", "pruning"]):
    print(f"{m:10s} {M[:, i].mean():.3f} +- {M[:, i].std(ddof=1):.3f}")
