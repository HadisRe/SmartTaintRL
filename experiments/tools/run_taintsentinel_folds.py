"""
TaintSentinel retrained and tested on the same five folds as SmartTaintRL (Table 8).

Model TaintSentinelWithPRA with the settings of the TaintSentinel paper: batch 16, 30 epochs, lr 0.001,
class weights on the imbalanced set, F1-optimal threshold on a validation split (15% of the training
folds, stratified, seed 42). Contracts missing from the TaintSentinel dataset get no prediction and
count as safe, like failed tool runs.

Set TAINTSENTINEL_DIR to the TaintSentinel code (Sentinel_1.py, Sentinel_2.py, Sentinel_4.py and its
batch1/batch2 data).
Usage:  python experiments/tools/run_taintsentinel_folds.py balanced|imbalanced
Output: results/tools/taintsentinel/<set>_fold<k>.json   (finished folds are skipped)
"""
import sys, os, json, random, io, contextlib, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

NAME = sys.argv[1] if len(sys.argv) > 1 else "balanced"
TS = os.environ.get("TAINTSENTINEL_DIR", "TaintSentinel"); sys.path.insert(0, TS)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("device:", dev, flush=True)
with contextlib.redirect_stdout(io.StringIO()):
    from Sentinel_1 import SmartContractDataset
    from Sentinel_2 import custom_collate_fn
    import Sentinel_4 as S4
    ds = SmartContractDataset(base_path=TS, batch_names=['batch1', 'batch2'], balanced=False, random_seed=42)
from torch.utils.data import DataLoader, Subset
folds = config.folds(NAME)   # same folds as SmartTaintRL
pos = {e["contract_id"].lower(): i for i, e in enumerate(ds.entries)}
lab = lambda i: ds.entries[i]["label"]
mk = lambda idx, sh: DataLoader(Subset(ds, idx), batch_size=16, shuffle=sh, collate_fn=custom_collate_fn)
for K in range(1, 6):
    out = str(config.out("tools", "taintsentinel", f"{NAME}_fold{K}.json"))
    if os.path.exists(out): print("fold", K, "already done"); continue
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    test_addr = folds[K - 1]; train_addr = [a for i, f in enumerate(folds) if i != K - 1 for a in f]
    tr_idx = [pos[a.lower()] for a in train_addr if a.lower() in pos]; te_idx = [pos[a.lower()] for a in test_addr if a.lower() in pos]
    # stratified validation split from the training folds
    rng = random.Random(42); val_idx = []
    for c in (0, 1):
        grp = [i for i in tr_idx if lab(i) == c]; rng.shuffle(grp); val_idx += grp[:max(1, int(0.15 * len(grp)))]
    sv = set(val_idx); fit_idx = [i for i in tr_idx if i not in sv]
    cw = None
    if NAME == "imbalanced":
        ns = sum(lab(i) == 0 for i in fit_idx); nv = sum(lab(i) == 1 for i in fit_idx); tot = ns + nv; cw = [tot / (2 * ns), tot / (2 * nv)]
    print(f"fold {K}: training on {len(fit_idx)} contracts (val {len(val_idx)}), testing on {len(te_idx)} ...", flush=True)
    model = S4.TaintSentinelWithPRA()
    model, _ = S4.train_model(model, mk(fit_idx, True), mk(val_idx, False), num_epochs=30, learning_rate=0.001, class_weights=cw, device=dev)
    thr = S4.find_best_threshold(model, mk(val_idx, False), dev)
    model.eval(); pred = {}
    with torch.no_grad():
        for b in mk(te_idx, False):
            b['graph'] = b['graph'].to(dev)
            for i in range(len(b['paths'])):
                for key in ['sequences', 'lengths', 'features']: b['paths'][i][key] = b['paths'][i][key].to(dev)
            o, _ = model(b, return_path_risks=True); p = F.softmax(o, dim=1)[:, 1].cpu().numpy()
            for cid, q in zip(b["contract_ids"], p): pred[cid.lower()] = float(q)
    rows = [dict(address=a, prob=pred.get(a.lower()), pred=bool(pred.get(a.lower(), 0) > thr), in_dataset=a.lower() in pred) for a in test_addr]
    json.dump(dict(threshold=float(thr), n_fit=len(fit_idx), n_val=len(val_idx), n_test=len(test_addr), n_test_in_dataset=len(te_idx), rows=rows), open(out, "w"))
    print(f"fold {K} saved -> {out}", flush=True)
print("done:", NAME)
