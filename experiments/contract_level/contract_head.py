"""
Contract-level decision (Section 4.2.8, Table 8), following PROTOCOL.md.

Inputs: the agent dumps of dump_agent.py (all five folds) and the Solidity sources.
For each outer fold, a (feature set, classifier) pair and a threshold are chosen on the
training contracts only (4-fold inner CV); the test fold is predicted once.
"pre-registered selection" is the row reported in the paper.

Usage:  python experiments/contract_level/contract_head.py balanced|imbalanced
Output: results/contract_level/result_<set>.json and pred_<set>.json (address -> predicted vulnerable)
"""
import json, re, math, os, sys, numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

name = sys.argv[1]
folds = config.folds(name)
IDX = config.source_index()
# raw counts of block variables and hash calls in the source (comments removed)
RAND = [r"block\.timestamp", r"\bnow\b", r"blockhash", r"block\.blockhash", r"block\.number", r"block\.difficulty", r"block\.coinbase", r"block\.gaslimit", r"keccak256", r"sha3", r"sha256"]
_c = {}
def src(a):
    if a not in _c:
        t = re.sub(r"//.*?$|/\*.*?\*/", " ", open(IDX[a], errors="ignore").read(), flags=re.S | re.M); _c[a] = [math.log1p(len(re.findall(k, t))) for k in RAND]
    return _c[a]
GRP = ["timestamp", "blocknumber", "blockhash", "difficulty"]
# F0: summary of the agent's decisions + source counts
def f0(r):
    d = r["dec"]; an = sum(x[1] for x in d)
    return [r["qgap_max"], sum(max(x[0], 0) for x in d if x[1]), an / max(len(d), 1), math.log1p(r["n_paths"])] + src(r["address"])
# F1: F0 + distribution of the Q-gap, and per source type
def f1(r):
    d = r["dec"]; g = np.array([x[0] for x in d]) if d else np.zeros(1); qa = max([x[4] for x in d] or [0])
    out = [g.mean(), g.std(), np.median(g), np.quantile(g, .75), np.quantile(g, .9), np.sort(g)[-3:].mean(),
           math.log1p((g > 0).sum()), math.log1p((g > 2).sum()), math.log1p((g > 5).sum()), qa]
    for G in GRP + ["other"]:
        sel = [x for x in d if (x[2] == G if G != "other" else x[2] not in GRP)]
        out += [max([x[0] for x in sel] or [-10.0]), math.log1p(sum(x[1] for x in sel))]
    return f0(r) + out
FEATS = {"F0": f0, "F1": f1}
CLFS = {"C0": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=0, n_jobs=-1),
        "C1": lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=1, class_weight="balanced_subsample", random_state=0, n_jobs=-1),
        "C2": lambda: ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced", random_state=0, n_jobs=-1),
        "C3": lambda: HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, class_weight="balanced", random_state=0)}
def met(y, p):
    tp = int((y & p).sum()); fp = int((~y & p).sum()); fn = int((y & ~p).sum()); tn = len(y) - tp - fp - fn
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1); F = 2 * P * R / max(P + R, 1e-9)
    mcc = (tp * tn - fp * fn) / math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=round(P, 3), recall=round(R, 3), f1=round(F, 4), mcc=round(mcc, 3))
# threshold with the best F1 on out-of-fold probabilities
def best_t(y, p):
    ts = np.unique(np.quantile(p, np.linspace(0, 1, 201))); f = [met(y, p >= t)["f1"] for t in ts]; i = int(np.argmax(f)); return ts[i], f[i]
def oof(X, y, mk):
    o = np.zeros(len(y))
    for a, b in StratifiedKFold(4, shuffle=True, random_state=0).split(X, y): o[b] = mk().fit(X[a], y[a]).predict_proba(X[b])[:, 1]
    return o
res = {"current (F0+C0)": dict(y=[], p=[], pf=[]), "pre-registered selection": dict(y=[], p=[], pf=[], chosen=[])}; preds = {}
for k in range(5):
    R = [r for r in json.load(open(config.out("agent_dumps", f"{name}_fold{k+1}.json"))) if r["address"] in IDX]; te_set = set(folds[k])
    tr = [r for r in R if r["address"] not in te_set]; te = [r for r in R if r["address"] in te_set]
    ytr = np.array([r["label"] == "vulnerable" for r in tr]); yte = np.array([r["label"] == "vulnerable" for r in te])
    X = {fn: (np.array([f(r) for r in tr]), np.array([f(r) for r in te])) for fn, f in FEATS.items()}
    scores = {}
    for fn in FEATS:
        for cn, mk in CLFS.items():
            o = oof(X[fn][0], ytr, mk); t, f = best_t(ytr, o); scores[(fn, cn)] = (f, t)
    def run(fn, cn):
        f, t = scores[(fn, cn)]; return CLFS[cn]().fit(X[fn][0], ytr).predict_proba(X[fn][1])[:, 1] >= t
    order = [(fn, cn) for fn in FEATS for cn in CLFS]
    # highest inner F1; on a tie the earlier (simpler) pair wins
    ch = max(order, key=lambda p: (round(scores[p][0], 6), -order.index(p)))
    for g, (fn, cn) in (("current (F0+C0)", ("F0", "C0")), ("pre-registered selection", ch)):
        p = run(fn, cn); res[g]["y"] += list(yte); res[g]["p"] += list(p); res[g]["pf"].append(met(yte, p)["f1"])
        preds.setdefault(g, {}).update({r["address"]: bool(x) for r, x in zip(te, p)})
    res["pre-registered selection"]["chosen"].append(dict(fold=k + 1, features=ch[0], clf=ch[1], inner_f1={f"{a}+{b}": round(v[0], 4) for (a, b), v in scores.items()}))
    print(f"fold {k+1}: chosen {ch}, test F1 current {res['current (F0+C0)']['pf'][-1]:.3f} selected {res['pre-registered selection']['pf'][-1]:.3f}", flush=True)
out = {g: dict(overall=met(np.array(v["y"]), np.array(v["p"])), per_fold_f1=v["pf"], **({"chosen": v["chosen"]} if "chosen" in v else {})) for g, v in res.items()}
json.dump(out, open(config.out("contract_level", f"result_{name}.json"), "w"), indent=1)
json.dump(preds, open(config.out("contract_level", f"pred_{name}.json"), "w"))
for g, v in out.items(): print(g, v["overall"], [round(x, 3) for x in v["per_fold_f1"]])
