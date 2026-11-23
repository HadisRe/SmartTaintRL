"""
Run a saved agent greedily over every path of every contract and store each decision.
This is the input of the contract-level decision (Section 4.2.8, Table 8).

For every decision we keep: Q-gap of the chosen path (Q_analyze - Q_skip), the action,
the source and sink type of the path, and Q_analyze. Risk labels are not stored.

Usage:  python experiments/contract_level/dump_agent.py balanced|imbalanced <fold 1-5>
Output: results/agent_dumps/<set>_fold<k>.json   (resumable through a .partial file)
"""
import os, sys, json, random, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cv"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
random.seed(0); np.random.seed(0); torch.manual_seed(0)
name, k = sys.argv[1], int(sys.argv[2]); sys.argv = [sys.argv[0]]
import config
import kfold_eval as v2
from env2_pool_manager import PoolManager

v2.MAX_STEPS = 10**7; cfg = v2.DATASETS[name]
base = v2.ContractDataLoader(path_db_dir=cfg['path_db_dir'], profile_dir=cfg['profile_dir'])
allc = {c[0]: c for c in base.get_valid_contracts(min_paths=5)}; base.contracts = {}
folds = config.folds(name)
OUT = str(config.out("agent_dumps", f"{name}_fold{k}.json")); PART = OUT + ".partial"
if os.path.exists(OUT): print("done already"); sys.exit(0)

agent = v2.DQNAgent(learning_rate=v2.LR, gamma=v2.GAMMA, epsilon_decay=v2.EPS_DECAY, epsilon_end=v2.EPS_END)
agent.q_network.load_state_dict(torch.load(config.model(name, k), map_location="cpu")); agent.q_network.eval()
cl = [allc[a] for f in folds for a in f if a in allc]
loader = v2.FoldDataLoader(base, cl)
env = v2.BadRandomnessEnv(data_loader=loader, pool_manager=PoolManager(), state_builder=v2.StateBuilder(), debug_mode=False)
env.max_steps = 10**7
agent.steps_done = 10_000_000; agent.epsilon_start = 0.0; agent.epsilon_end = 0.0

rows = json.load(open(PART)) if os.path.exists(PART) else []
done_set = {r['address'] for r in rows}
for j, (addr, n_paths, label) in enumerate(cl):
    if addr in done_set: continue
    random.seed(j); np.random.seed(j)   # fixed initial pool per contract
    loader.only = addr; s, info = env.reset()
    r = dict(address=addr, label=label, n_paths=n_paths, qgap_max=-1e9, dec=[]); done = False
    while not done:
        v = len(env.pool_state.current_pool)
        if v == 0: break
        with torch.no_grad(): q = agent.q_network(torch.FloatTensor(s).unsqueeze(0))[0][:min(v, 20)]
        gap = (q[:, 0] - q[:, 1]).numpy(); r['qgap_max'] = max(r['qgap_max'], float(gap.max()))
        pi, at = agent.select_action(s, valid_paths=v)
        bi = env.pool_state.current_pool[pi]['basic_info'] if pi < v else {}
        g = float(gap[pi]) if pi < len(gap) else 0.0
        r['dec'].append([round(g, 4), int(at == 'ANALYZE'), bi.get('source_type', ''), bi.get('sink_type', ''),
                         round(float(q[pi, 0]) if pi < len(gap) else 0.0, 4)])
        s, _, done, info = env.step((pi, at))
    base.loaded_contracts.pop(addr, None); rows.append(r)
    if j % 100 == 0:
        print(f"{name} fold {k}: {j}/{len(cl)}", flush=True); json.dump(rows, open(PART, "w"))
json.dump(rows, open(OUT, "w")); print("saved", OUT, flush=True)
