"""
Five-fold cross-validation of the SmartTaintRL agent (Sections 5.2-5.3, Table 7).

The agent code in src/rl_agent is used unchanged: same state, actions, reward,
network and hyper-parameters (Table 5). This script only adds the protocol:
  - stratified 5-fold split; contracts with identical path databases stay in the
    same fold, so a copy of a test contract never appears in training;
  - the folds are saved (experiments/folds) and reused by every other method;
  - each test contract is evaluated exactly once, greedily, with no learning.

Training and a quick evaluation of each fold (at most 50 decisions per contract).
The path-level numbers of Table 7 come from score_test_folds.py, which runs the saved
models until every path of a test contract has been decided.

Usage:
    python experiments/cv/kfold_eval.py balanced
    python experiments/cv/kfold_eval.py imbalanced
The models used in the paper are in models/ (dqn_<set>_fold<k>.pth).
"""
import os, sys, json, random, hashlib, time, logging
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
logging.disable(logging.CRITICAL)

from env1_data_loader import ContractDataLoader
from env2_pool_manager import PoolManager
from env3_state_builder import StateBuilder
from env4_environment import BadRandomnessEnv
from dqn_agent import DQNAgent, PrioritizedReplayBuffer

DATASETS = {
    name: dict(path_db_dir=str(config.PATH_DB[name]), profile_dir=str(config.PROFILES[name]),
               episodes=int(os.environ.get("EPISODES", 2500)))
    for name in ("balanced", "imbalanced")
}

# training settings (Table 5)
N_FOLDS = 5; MAX_STEPS = 50; SEED = 42
LR = 1e-5; GAMMA = 0.95; EPS_DECAY = 500; EPS_END = 0.05
BUFFER_CAPACITY = 5000; BATCH_SIZE = 32; TARGET_UPDATE = 50
GROUP_DUPLICATES = True


class FoldDataLoader:
    def __init__(self, base, contracts):
        self._base = base; self._list = list(contracts); self.only = None
    def get_valid_contracts(self, min_paths: int = 5):
        if self.only is not None:
            return [c for c in self._list if c[0] == self.only]
        return self._list
    def __getattr__(self, name):
        return getattr(self._base, name)


def contract_fingerprint(base, addr):
    c = base.load_contract(addr)
    sig = [(p['basic_info'].get('path_nodes'), p.get('aggregate_features')) for p in c.paths]
    base.loaded_contracts.pop(addr, None)
    return hashlib.md5(json.dumps(sig, sort_keys=True, default=str).encode()).hexdigest()


def make_folds(base, contracts, n_folds, seed):
    """Stratified folds; identical (duplicate) contracts are kept in the same fold."""
    groups = {}
    for c in contracts:
        key = contract_fingerprint(base, c[0]) if GROUP_DUPLICATES else c[0]
        groups.setdefault(key, []).append(c)
    glist = list(groups.values())
    rng = random.Random(seed); rng.shuffle(glist)
    # vulnerable-containing groups first, then safe groups, assigned to the lightest fold
    glist.sort(key=lambda g: 0 if any(x[2] == 'vulnerable' for x in g) else 1)
    folds = [[] for _ in range(n_folds)]
    for g in glist:
        isv = any(x[2] == 'vulnerable' for x in g)
        k = min(range(n_folds), key=lambda i: (sum(x[2] == 'vulnerable' for x in folds[i]) if isv else len(folds[i]), len(folds[i])))
        folds[k].extend(g)
    return folds, sum(1 for g in glist if len(g) > 1)


def train_on_fold(base, train_contracts, episodes):
    env = BadRandomnessEnv(data_loader=FoldDataLoader(base, train_contracts),
                           pool_manager=PoolManager(), state_builder=StateBuilder(), debug_mode=False)
    agent = DQNAgent(learning_rate=LR, gamma=GAMMA, epsilon_decay=EPS_DECAY, epsilon_end=EPS_END)
    buffer = PrioritizedReplayBuffer(capacity=BUFFER_CAPACITY)
    t0 = time.time()
    for ep in range(episodes):
        state, info = env.reset(); done, steps = False, 0
        while not done and steps < MAX_STEPS:
            valid = info.get('valid_paths', info.get('pool_size', 10))
            action = agent.select_action(state, valid_paths=valid)
            nxt, reward, done, info = env.step(action)
            buffer.push(state, action, np.clip(reward, -10, 10), nxt, done, info)
            if len(buffer) >= BATCH_SIZE:
                agent.update(buffer.sample(BATCH_SIZE))
                if agent.steps_done % TARGET_UPDATE == 0:
                    agent.update_target_network()
            state = nxt; steps += 1
        if len(env.data_loader.loaded_contracts) > 300:
            env.data_loader.loaded_contracts.clear()
        if ep % 500 == 0:
            print(f"      [train] episode {ep}/{episodes}  ({time.time()-t0:.0f}s)", flush=True)
    return agent


def evaluate_on_fold(base, agent, test_contracts):
    """Greedy, no learning, every test contract exactly once (at most MAX_STEPS decisions)."""
    loader = FoldDataLoader(base, test_contracts)
    env = BadRandomnessEnv(data_loader=loader, pool_manager=PoolManager(),
                           state_builder=StateBuilder(), debug_mode=False)
    agent.steps_done = 10_000_000; agent.epsilon_start = 0.0; agent.epsilon_end = 0.0  # fully greedy at test time
    rows = []
    for addr, n_paths, label in test_contracts:
        loader.only = addr
        state, info = env.reset()
        has_high = any(p['basic_info'].get('risk_level') == 'HIGH' for p in env.current_contract.paths)
        r = dict(address=addr, label=label, n_paths=n_paths, has_high=has_high,
                 analyzed=0, skipped=0, analyzed_high=0, tp=0, fp=0, tn=0, fn=0)
        done, steps = False, 0
        while not done and steps < MAX_STEPS:
            valid = len(env.pool_state.current_pool)
            if valid == 0:
                break
            pidx, atype = agent.select_action(state, valid_paths=valid)
            risk = env.pool_state.current_pool[pidx]['basic_info'].get('risk_level') if pidx < valid else None
            state, _, done, info = env.step((pidx, atype))
            steps += 1
            if atype == 'ANALYZE':
                r['analyzed'] += 1; r['analyzed_high'] += (risk == 'HIGH')
            else:
                r['skipped'] += 1
            # path-level: identical to original EnhancedMetricsCollector.update
            if risk == 'HIGH':
                r['tp' if atype == 'ANALYZE' else 'fn'] += 1
            elif risk == 'LOW':
                r['tn' if atype == 'SKIP' else 'fp'] += 1
        base.loaded_contracts.pop(addr, None)
        rows.append(r)
    return rows


def prf(tp, fp, fn, tn):
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1); F = 2 * P * R / max(P + R, 1e-9)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=P, recall=R, f1=F,
                accuracy=(tp + tn) / max(tp + fp + fn + tn, 1))


def summarize(rows):
    s = {}
    s['path_level'] = prf(sum(r['tp'] for r in rows), sum(r['fp'] for r in rows),
                          sum(r['fn'] for r in rows), sum(r['tn'] for r in rows))
    dec = sum(r['analyzed'] + r['skipped'] for r in rows)
    s['pruning_of_decided_paths'] = sum(r['skipped'] for r in rows) / max(dec, 1)
    s['pruning_of_all_paths'] = 1 - sum(r['analyzed'] for r in rows) / max(sum(r['n_paths'] for r in rows), 1)
    s['coverage_decided_over_all_paths'] = dec / max(sum(r['n_paths'] for r in rows), 1)
    # (a) first-submission definition (contract counted vulnerable if it has a HIGH path), kept for reference
    tp = sum(r['has_high'] and r['analyzed_high'] > 0 for r in rows)
    fn = sum(r['has_high'] and r['analyzed_high'] == 0 for r in rows)
    fp = sum((not r['has_high']) and r['analyzed'] > 0 for r in rows)
    tn = sum((not r['has_high']) and r['analyzed'] == 0 for r in rows)
    s['contract_level_OLD_definition_has_HIGH_path'] = prf(tp, fp, fn, tn)
    # (b) true contract label; contract flagged if at least one path is analyzed.
    #     (The paper's contract-level results use the decision of Section 4.2.8 instead, see contract_level/.)
    pred = [r['analyzed'] > 0 for r in rows]; y = [r['label'] == 'vulnerable' for r in rows]
    s['contract_level_TRUE_label'] = prf(sum(p and t for p, t in zip(pred, y)), sum(p and not t for p, t in zip(pred, y)),
                                         sum(t and not p for p, t in zip(pred, y)), sum((not p) and (not t) for p, t in zip(pred, y)))
    return s


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "balanced"
    cfg = DATASETS[name]
    out_dir = str(config.out("cv", "x").parent)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    base = ContractDataLoader(path_db_dir=cfg['path_db_dir'], profile_dir=cfg['profile_dir'])
    contracts = base.get_valid_contracts(min_paths=5)
    base.contracts = {}                      # free the full cache (memory)
    nv = sum(c[2] == 'vulnerable' for c in contracts)
    print(f"[{name}] contracts={len(contracts)} vulnerable={nv} safe={len(contracts)-nv}  episodes/fold={cfg['episodes']}")

    if (config.FOLDS_DIR / f"folds_{name}.json").exists():
        # reuse the folds of the paper (same for every method)
        by_addr = {c[0]: c for c in contracts}
        folds = [[by_addr[a] for a in f if a in by_addr] for f in config.folds(name)]; n_dup = "see folds file"
    else:
        folds, n_dup = make_folds(base, contracts, N_FOLDS, SEED)
    json.dump({'dataset': name, 'seed': SEED, 'group_duplicates': GROUP_DUPLICATES,
               'folds': [[c[0] for c in f] for f in folds]},
              open(os.path.join(out_dir, f"folds_{name}.json"), "w"), indent=1)
    # should be identical to experiments/folds/folds_<name>.json
    print(f"  duplicate groups kept together: {n_dup}; fold sizes: {[len(f) for f in folds]}; "
          f"vulnerable per fold: {[sum(c[2]=='vulnerable' for c in f) for f in folds]}")

    all_rows, per_fold = [], []
    for k in range(N_FOLDS):
        test = folds[k]; train = [c for i, f in enumerate(folds) if i != k for c in f]
        print(f"\n--- FOLD {k+1}/{N_FOLDS}  train={len(train)} test={len(test)}", flush=True)
        torch.manual_seed(SEED + k); random.seed(SEED + k); np.random.seed(SEED + k)
        agent = train_on_fold(base, train, cfg['episodes'])
        rows = evaluate_on_fold(base, agent, test)
        for r in rows: r['fold'] = k
        s = summarize(rows); per_fold.append(s); all_rows += rows
        print(f"  path F1={s['path_level']['f1']:.3f} | contract(OLD def) F1={s['contract_level_OLD_definition_has_HIGH_path']['f1']:.3f} "
              f"| contract(TRUE label) P={s['contract_level_TRUE_label']['precision']:.3f} "
              f"R={s['contract_level_TRUE_label']['recall']:.3f} F1={s['contract_level_TRUE_label']['f1']:.3f}", flush=True)
        torch.save(agent.q_network.state_dict(), os.path.join(out_dir, f"dqn_{name}_fold{k+1}.pth"))

    def ms(path):
        v = [s for s in per_fold]
        for p in path: v = [x[p] for x in v]
        return float(np.mean(v)), float(np.std(v))
    keys = [('path_level', 'precision'), ('path_level', 'recall'), ('path_level', 'f1'),
            ('pruning_of_decided_paths',), ('pruning_of_all_paths',), ('coverage_decided_over_all_paths',),
            ('contract_level_OLD_definition_has_HIGH_path', 'f1'),
            ('contract_level_TRUE_label', 'precision'), ('contract_level_TRUE_label', 'recall'),
            ('contract_level_TRUE_label', 'f1')]
    summary = {'/'.join(k): ms(k) for k in keys}
    print(f"\n===== {name}: mean ± std over {N_FOLDS} folds =====")
    for k, (m, s) in summary.items():
        print(f"  {k:<55} {m:.3f} ± {s:.3f}")
    json.dump({'dataset': name, 'episodes_per_fold': cfg['episodes'], 'summary_mean_std': summary,
               'per_fold': per_fold, 'per_contract': all_rows},
              open(os.path.join(out_dir, f"kfold_{name}.json"), "w"), indent=1)
    print(f"\nsaved -> {os.path.join(out_dir, f'kfold_{name}.json')}")


if __name__ == "__main__":
    main()
