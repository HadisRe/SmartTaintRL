"""
Keyword-count baseline of Table 8: the same random forest and threshold rule as the
contract-level decision, but with the 11 source counts only (no agent output).

Usage:  python experiments/contract_level/keyword_baseline.py balanced|imbalanced
"""
import json, re, math, os, sys, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

name = sys.argv[1]
folds = config.folds(name)
IDX = config.source_index()
RAND = [r"block\.timestamp", r"\bnow\b", r"blockhash", r"block\.blockhash", r"block\.number", r"block\.difficulty",
        r"block\.coinbase", r"block\.gaslimit", r"keccak256", r"sha3", r"sha256"]


def src(a):
    t = re.sub(r"//.*?$|/\*.*?\*/", " ", open(IDX[a], errors="ignore").read(), flags=re.S | re.M)
    return [math.log1p(len(re.findall(k, t))) for k in RAND]


def met(y, p):
    tp = int((y & p).sum()); fp = int((~y & p).sum()); fn = int((y & ~p).sum()); tn = len(y) - tp - fp - fn
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1); F = 2 * P * R / max(P + R, 1e-9)
    mcc = (tp * tn - fp * fn) / math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=round(P, 3), recall=round(R, 3), f1=round(F, 4), mcc=round(mcc, 3))


lab = {r["address"]: r["label"] == "vulnerable" for r in config.contracts(name)}
y_all, p_all, pf = [], [], []
for k in range(5):
    te_set = set(folds[k])
    tr = [a for f in folds for a in f if a not in te_set and a in IDX]; te = [a for a in folds[k] if a in IDX]
    ytr = np.array([lab[a] for a in tr]); yte = np.array([lab[a] for a in te])
    Xtr = np.array([src(a) for a in tr]); Xte = np.array([src(a) for a in te])
    mk = lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=0, n_jobs=-1)
    oo = cross_val_predict(mk(), Xtr, ytr, cv=4, method="predict_proba")[:, 1]
    t = max(np.unique(np.quantile(oo, np.linspace(0, 1, 201))), key=lambda t: met(ytr, oo >= t)["f1"])
    p = mk().fit(Xtr, ytr).predict_proba(Xte)[:, 1] >= t; y_all += list(yte); p_all += list(p); pf.append(met(yte, p)["f1"])
print(name, met(np.array(y_all), np.array(p_all)), [round(x, 3) for x in pf])
