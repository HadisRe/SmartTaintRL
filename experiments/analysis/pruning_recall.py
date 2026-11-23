"""
Pruning-recall trade-off (appendix figure), balanced models on their test folds, greedy, no training.

For each decided path we keep its risk label and the margin Q(ANALYZE) - Q(SKIP) at decision time.
Moving a threshold t on the margin (analyze if margin > t) traces pruning against path recall.
Upper bound: with a share pi of HIGH-risk paths, pruning a share rho gives recall <= min(1, (1 - rho) / pi).

Usage:  python experiments/analysis/pruning_recall.py
Output: results/analysis/pruning_recall_tradeoff_balanced.json and results/figures/pruning_recall_tradeoff.pdf
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
rec = []   # (margin, risk, analyzed_by_agent)
for k in range(5):
    agent = v2.DQNAgent(learning_rate=v2.LR, gamma=v2.GAMMA, epsilon_decay=v2.EPS_DECAY, epsilon_end=v2.EPS_END)
    agent.q_network.load_state_dict(torch.load(config.model("balanced", k + 1), map_location="cpu")); net = agent.q_network.eval()
    loader = v2.FoldDataLoader(base, folds[k]); env = v2.BadRandomnessEnv(data_loader=loader, pool_manager=PoolManager(), state_builder=v2.StateBuilder(), debug_mode=False)
    agent.steps_done = 10_000_000; agent.epsilon_start = 0.0; agent.epsilon_end = 0.0
    for addr, n_paths, label in folds[k]:
        loader.only = addr; s, info = env.reset(); done = False
        while not done:
            v = len(env.pool_state.current_pool)
            if v == 0: break
            with torch.no_grad(): q = net(torch.FloatTensor(s).unsqueeze(0))[0]
            pi, at = agent.select_action(s, valid_paths=v)
            risk = env.pool_state.current_pool[pi]['basic_info'].get('risk_level') if pi < v else None
            rec.append((float(q[pi, 0] - q[pi, 1]), risk, at == 'ANALYZE'))
            s, _, done, info = env.step((pi, at))
        base.loaded_contracts.pop(addr, None)
    print("fold", k + 1, len(rec), flush=True)
m = np.array([r[0] for r in rec]); hi = np.array([r[1] == 'HIGH' for r in rec]); lo = np.array([r[1] == 'LOW' for r in rec]); an = np.array([r[2] for r in rec])
pi_high = hi.mean()
def point(mask):
    tp = (mask & hi).sum(); fn = (~mask & hi).sum(); fp = (mask & lo).sum()
    P = tp / max(tp + fp, 1); Rr = tp / max(tp + fn, 1)
    return dict(pruning=float(1 - mask.mean()), recall=float(Rr), precision=float(P), f1=float(2 * P * Rr / max(P + Rr, 1e-9)))
curve = [dict(t=float(t), **point(m > t)) for t in np.quantile(m, np.linspace(0, 1, 101))]
agent_pt = point(an)
bound = lambda rho: min(1.0, (1 - rho) / pi_high)
summ = dict(n_paths=len(rec), share_high=float(pi_high), max_lossless_pruning=float(1 - pi_high), agent=agent_pt,
            recall_at_pruning={f"{p:.2f}": max([c['recall'] for c in curve if c['pruning'] >= p - 1e-9] or [0]) for p in [0.2, 0.3, 0.4, 0.45, 0.5, 0.6]},
            bound_at_pruning={f"{p:.2f}": bound(p) for p in [0.2, 0.3, 0.4, 0.45, 0.5, 0.6]})
json.dump(dict(summary=summ, curve=curve), open(config.out("analysis", "pruning_recall_tradeoff_balanced.json"), "w"), indent=1)
print(json.dumps(summ, indent=1))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
f, a = plt.subplots(figsize=(6.5, 4.2))
xs = np.linspace(0, 0.9, 200); a.plot(xs * 100, [bound(x) for x in xs], '--', color='#555555', lw=1.8, label='Upper bound (perfect ranking)')
cc = sorted(curve, key=lambda c: c['pruning']); a.plot([c['pruning'] * 100 for c in cc], [c['recall'] for c in cc], '-', color='#0072B2', lw=2.2, label='SmartTaintRL, varying threshold')
a.plot(agent_pt['pruning'] * 100, agent_pt['recall'], 'o', color='#D55E00', ms=9, label='SmartTaintRL, greedy policy')
a.axvline((1 - pi_high) * 100, color='#999999', lw=0.8, ls=':'); a.set_xlabel('Pruned paths (%)'); a.set_ylabel('Path-level recall (HIGH)'); a.set_xlim(0, 90); a.set_ylim(0, 1.02)
a.grid(alpha=.3, ls='--'); a.legend(loc='lower left', fontsize=10.5); f.tight_layout(); f.savefig(config.out("figures", "pruning_recall_tradeoff.pdf")); f.savefig(config.out("figures", "pruning_recall_tradeoff.png"), dpi=130)
