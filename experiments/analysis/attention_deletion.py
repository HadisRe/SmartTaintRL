"""
Are the attention weights faithful to the agent's decisions? (Section 6, Reviewer 1 comment 9)
Deletion test on the balanced models, test folds only, greedy run, no training.

In every visited state with at least 6 real paths, the 3 paths ranked highest by a rule are deleted
(row set to zero, which is how the network sees an empty pool slot). On the remaining real paths we
measure (a) the mean absolute change of the margin Q(ANALYZE) - Q(SKIP) and (b) the share of decisions
that flip. Rules: attention received (column mean over real query paths), attention x gradient
(the localization signal), and random (5 draws).

Usage:  python experiments/analysis/attention_deletion.py
Output: results/analysis/attention_deletion_test_balanced.json
"""
import os, sys, json, random, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cv"))
sys.argv = [sys.argv[0]]
import config
import kfold_eval as v2
from env2_pool_manager import PoolManager
# the initial pool of each contract is random; the paper runs did not fix it, so reruns can
# differ slightly in the last digits. The seed makes reruns of this script identical.
random.seed(0); np.random.seed(0); torch.manual_seed(0)
cfg = v2.DATASETS["balanced"]; base = v2.ContractDataLoader(path_db_dir=cfg['path_db_dir'], profile_dir=cfg['profile_dir'])
allc = {c[0]: c for c in base.get_valid_contracts(min_paths=5)}; base.contracts = {}
folds = [[allc[a] for a in f] for f in config.folds("balanced")]
K = 3; rng = random.Random(0); out = {"attention": [], "attention_x_gradient": [], "random": []}
# margin change and flip rate on the paths that were not deleted
def effect(net, s, v, drop):
    x = torch.FloatTensor(s).unsqueeze(0)
    with torch.no_grad():
        q0 = net(x)[0][:v]; x2 = x.clone(); x2[0, drop] = 0; q1 = net(x2)[0][:v]
    keep = [i for i in range(v) if i not in drop]
    g0 = (q0[keep, 0] - q0[keep, 1]); g1 = (q1[keep, 0] - q1[keep, 1])
    return float((g1 - g0).abs().mean()), float(((g0 > 0) != (g1 > 0)).float().mean())
n_states = 0; paired = []
for k in range(5):
    agent = v2.DQNAgent(learning_rate=v2.LR, gamma=v2.GAMMA, epsilon_decay=v2.EPS_DECAY, epsilon_end=v2.EPS_END)
    agent.q_network.load_state_dict(torch.load(config.model("balanced", k + 1), map_location="cpu")); net = agent.q_network.eval()
    loader = v2.FoldDataLoader(base, folds[k]); env = v2.BadRandomnessEnv(data_loader=loader, pool_manager=PoolManager(), state_builder=v2.StateBuilder(), debug_mode=False)
    agent.steps_done = 10_000_000; agent.epsilon_start = 0.0; agent.epsilon_end = 0.0
    for addr, n_paths, label in folds[k]:
        loader.only = addr; s, info = env.reset(); done = False
        while not done:
            v = min(len(env.pool_state.current_pool), 20)
            if v == 0: break
            if v >= 6:
                x = torch.FloatTensor(s).unsqueeze(0).requires_grad_(True)
                q, att = net(x, return_attention=True)
                a_recv = att[0, :v, :v].mean(0)                                   # attention received by each real path
                gap = (q[0, :v, 0] - q[0, :v, 1]).sum(); gap.backward()
                grad = x.grad[0, :v].abs().sum(1)
                ax = (a_recv.detach() * grad)
                for name, score in [("attention", a_recv.detach()), ("attention_x_gradient", ax)]:
                    out[name].append(effect(net, s, v, [int(i) for i in torch.argsort(score, descending=True)[:K]]))
                rr = [effect(net, s, v, rng.sample(range(v), K)) for _ in range(5)]; out["random"] += rr
                paired.append((out["attention"][-1][0], float(np.mean([a for a, _ in rr]))))
                n_states += 1
            pi, at = agent.select_action(s, valid_paths=len(env.pool_state.current_pool))
            s, _, done, info = env.step((pi, at))
        base.loaded_contracts.pop(addr, None)
    print("fold", k + 1, "states so far", n_states, flush=True)
summ = {m: dict(mean_abs_margin_change=float(np.mean([a for a, _ in v])), decision_flip_rate=float(np.mean([b for _, b in v])), n=len(v)) for m, v in out.items()}
from scipy import stats
summ["n_states"] = n_states; summ["k_deleted"] = K
summ["wilcoxon_attention_vs_random_p"] = float(stats.wilcoxon([a for a, _ in paired], [b for _, b in paired], alternative="greater").pvalue)
summ["share_states_attention_gt_random"] = float(np.mean([a > b for a, b in paired]))
print(json.dumps(summ, indent=1)); json.dump(summ, open(config.out("analysis", "attention_deletion_test_balanced.json"), "w"), indent=1)
