"""
Plain gradient attribution (used in the paper) against Integrated Gradients (Sundararajan et al., 2017).
(Section 6, Reviewer 1 comment 10.) Balanced models, test folds, greedy run.

For each analyzed path the target is Q(ANALYZE) - Q(SKIP) of that path, attributed over its 100 features.
  grad: |d target / d x_j|  (the gradient term of the localization score; the attention factor is constant within a path)
  IG  : |x_j * mean over alpha of d target / d x_j(alpha x)|, 32 steps, zero baseline for that path's row only
Agreement: Spearman correlation and top-5 overlap. Faithfulness: zero the top-5 features of the path and
measure the change of the target; 5 random features as reference.

Usage:  python experiments/analysis/ig_vs_gradient.py
Output: results/analysis/ig_vs_gradient_balanced.json
"""
import os, sys, json, random, numpy as np, torch
from scipy import stats
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
rng = random.Random(0); STEPS = 32; TOP = 5
rho, overlap, dG, dI, dR, dGx = [], [], [], [], [], []
def target(net, x, i): q = net(x)[0, i]; return q[0] - q[1]
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
            pi, at = agent.select_action(s, valid_paths=v)
            if at == 'ANALYZE' and pi < 20:
                x0 = torch.FloatTensor(s).unsqueeze(0)
                xg = x0.clone().requires_grad_(True); t = target(net, xg, pi); t.backward(); g = xg.grad[0, pi].abs()
                tot = torch.zeros(100)
                for a in range(1, STEPS + 1):
                    xa = x0.clone(); xa[0, pi] = x0[0, pi] * a / STEPS; xa.requires_grad_(True)
                    ta = target(net, xa, pi); ta.backward(); tot += xa.grad[0, pi]
                ig = (x0[0, pi] * tot / STEPS).abs()
                nz = (x0[0, pi] != 0)
                if nz.sum() >= 10:
                    r = stats.spearmanr(g[nz].numpy(), ig[nz].numpy()).correlation
                    if not np.isnan(r): rho.append(r)
                    gt = set(torch.argsort(g * nz, descending=True)[:TOP].tolist()); it = set(torch.argsort(ig, descending=True)[:TOP].tolist())
                    overlap.append(len(gt & it) / TOP)
                    with torch.no_grad():
                        base_t = float(target(net, x0, pi))
                        def drop(idx):
                            x2 = x0.clone(); x2[0, pi, list(idx)] = 0; return abs(float(target(net, x2, pi)) - base_t)
                        gx = (g * x0[0, pi].abs())
                        dG.append(drop(gt)); dI.append(drop(it)); dGx.append(drop(torch.argsort(gx, descending=True)[:TOP].tolist()))
                        dR.append(drop(rng.sample(torch.nonzero(nz).flatten().tolist(), TOP)))
            s, _, done, info = env.step((pi, at))
        base.loaded_contracts.pop(addr, None)
    print("fold", k + 1, "explained paths", len(dG), flush=True)
summ = dict(n=len(dG), spearman_grad_vs_IG_median=float(np.median(rho)), spearman_mean=float(np.mean(rho)), top5_overlap_mean=float(np.mean(overlap)),
            deletion_change_grad=float(np.mean(dG)), deletion_change_IG=float(np.mean(dI)), deletion_change_grad_x_input=float(np.mean(dGx)), deletion_change_random=float(np.mean(dR)),
            wilcoxon_IG_vs_grad_p=float(stats.wilcoxon(dI, dG).pvalue))
print(json.dumps(summ, indent=1)); json.dump(summ, open(config.out("analysis", "ig_vs_gradient_balanced.json"), "w"), indent=1)
