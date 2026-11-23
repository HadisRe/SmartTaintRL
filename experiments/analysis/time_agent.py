"""
Agent time per contract (Section 5): imbalanced fold-1 model, greedy, every path decided, on a random
sample of 300 contracts, one CPU thread. Reading the path database from disk is not included (it is
part of path extraction, see time_pipeline.py).

Usage:  python experiments/analysis/time_agent.py
"""
import os, sys, json, time, random, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cv"))
sys.argv = [sys.argv[0]]
import config
import kfold_eval as v2
from env2_pool_manager import PoolManager
torch.set_num_threads(1); random.seed(0); np.random.seed(0)
v2.MAX_STEPS = 10**7; cfg = v2.DATASETS["imbalanced"]
base = v2.ContractDataLoader(path_db_dir=cfg['path_db_dir'], profile_dir=cfg['profile_dir'])
allc = {c[0]: c for c in base.get_valid_contracts(min_paths=5)}; base.contracts = {}
cl = random.sample(sorted(allc.values()), 300)
agent = v2.DQNAgent(learning_rate=v2.LR, gamma=v2.GAMMA, epsilon_decay=v2.EPS_DECAY, epsilon_end=v2.EPS_END)
agent.q_network.load_state_dict(torch.load(config.model("imbalanced", 1), map_location="cpu")); agent.q_network.eval()
loader = v2.FoldDataLoader(base, cl); env = v2.BadRandomnessEnv(data_loader=loader, pool_manager=PoolManager(), state_builder=v2.StateBuilder(), debug_mode=False); env.max_steps = 10**7
agent.steps_done = 10_000_000; agent.epsilon_start = 0.0; agent.epsilon_end = 0.0; T = []
for addr, n, lab in cl:
    loader.only = addr; base.load_contract(addr) if hasattr(base, "load_contract") else None
    t0 = time.perf_counter(); s, info = env.reset(); done = False
    while not done:
        v = len(env.pool_state.current_pool)
        if v == 0: break
        pi, at = agent.select_action(s, valid_paths=v); s, _, done, info = env.step((pi, at))
    T.append(time.perf_counter() - t0); base.loaded_contracts.pop(addr, None)
print("contracts", len(T), "mean ms %.1f median ms %.1f" % (1000 * np.mean(T), 1000 * np.median(T)))
