"""
Training with a pruning log (Fig. 9): the same training loop as kfold_eval.py, plus the share
of skipped paths in every training episode. Folds and seeds are those of the paper, so the
balanced models are identical to models/dqn_balanced_fold<k>.pth (the script checks this).

Usage:  python experiments/cv/train_with_pruning_log.py balanced|imbalanced
Output: results/pruning_log/
"""
import os, sys, json, time, random, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NAME = sys.argv[1] if len(sys.argv) > 1 else "imbalanced"; assert NAME in ("balanced", "imbalanced")
sys.argv = [sys.argv[0]]
import config
import kfold_eval as v2

OUT = str(config.out("pruning_log", "x").parent)
cfg = v2.DATASETS[NAME]
base = v2.ContractDataLoader(path_db_dir=cfg['path_db_dir'], profile_dir=cfg['profile_dir'])
allc = {c[0]: c for c in base.get_valid_contracts(min_paths=5)}; base.contracts = {}
folds = [[allc[a] for a in f if a in allc] for f in config.folds(NAME)]
print(f"[{NAME}] fold sizes: {[len(f) for f in folds]}  episodes per fold: {cfg['episodes']}", flush=True)

def train_logged(train_contracts, episodes):
    """Same as kfold_eval.train_on_fold, plus one pruning value per episode (skipped / decided)."""
    env = v2.BadRandomnessEnv(data_loader=v2.FoldDataLoader(base, train_contracts), pool_manager=v2.PoolManager(),
                              state_builder=v2.StateBuilder(), debug_mode=False)
    agent = v2.DQNAgent(learning_rate=v2.LR, gamma=v2.GAMMA, epsilon_decay=v2.EPS_DECAY, epsilon_end=v2.EPS_END)
    buffer = v2.PrioritizedReplayBuffer(capacity=v2.BUFFER_CAPACITY); log = []; t0 = time.time()
    for ep in range(episodes):
        state, info = env.reset(); done, steps, skips = False, 0, 0
        while not done and steps < v2.MAX_STEPS:
            valid = info.get('valid_paths', info.get('pool_size', 10))
            action = agent.select_action(state, valid_paths=valid)
            skips += (action[1] == 'SKIP')
            nxt, reward, done, info = env.step(action)
            buffer.push(state, action, np.clip(reward, -10, 10), nxt, done, info)
            if len(buffer) >= v2.BATCH_SIZE:
                agent.update(buffer.sample(v2.BATCH_SIZE))
                if agent.steps_done % v2.TARGET_UPDATE == 0:
                    agent.update_target_network()
            state = nxt; steps += 1
        log.append(skips / max(steps, 1))
        if len(env.data_loader.loaded_contracts) > 300:
            env.data_loader.loaded_contracts.clear()
        if ep % 250 == 0:
            print(f"      episode {ep}/{episodes}  pruning(last 100)={np.mean(log[-100:]):.3f}  ({time.time()-t0:.0f}s)", flush=True)
    return agent, log

for k in range(5):
    logf = os.path.join(OUT, f"pruning_log_fold{k+1}.json")
    if os.path.exists(logf): print(f"fold {k+1}: done already"); continue
    train = [c for i, f in enumerate(folds) if i != k for c in f]
    print(f"\n--- fold {k+1}/5: training on {len(train)} contracts", flush=True)
    torch.manual_seed(v2.SEED + k); random.seed(v2.SEED + k); np.random.seed(v2.SEED + k)
    agent, log = train_logged(train, cfg['episodes'])
    torch.save(agent.q_network.state_dict(), os.path.join(OUT, f"dqn_{NAME}_fold{k+1}.pth"))
    rows = v2.evaluate_on_fold(base, agent, folds[k]); s = v2.summarize(rows)
    same = None
    ref = str(config.model(NAME, k + 1))
    if NAME == "balanced" and os.path.exists(ref):
        r = torch.load(ref, map_location="cpu"); n = agent.q_network.state_dict()
        same = float(max((r[x].float() - n[x].float()).abs().max() for x in r))
    json.dump(dict(summary=s, rows=rows), open(os.path.join(OUT, f"fold{k+1}_eval.json"), "w"))
    json.dump(dict(dataset=NAME, fold=k + 1, pruning_per_episode=log, test_pruning=s['pruning_of_all_paths'],
                   test_path_f1=s['path_level']['f1'], max_weight_diff_vs_paper_model=same), open(logf, "w"))
    print(f"fold {k+1} saved. test path F1={s['path_level']['f1']:.3f}  test pruning={s['pruning_of_all_paths']:.3f}"
          + (f"  weight diff vs paper model={same:.2e}" if same is not None else ""), flush=True)
print("done")
