"""
Path-level baselines (Table 10): Top-k, logistic regression, XGBoost and LambdaMART, on the same folds as the agent.

The learned baselines build their own features from the raw path record (TF-IDF of code tokens and node types,
source and sink type, path length, functions involved). They do not see the fields from which the risk label is made.
Top-k ranks paths by the fixed priority score of the pool manager (no learning).
Equal budget: in each contract a baseline analyzes its highest-ranked paths, up to the share of paths the agent
analyzed in that fold. LR and XGBoost are also reported with their own threshold (chosen on the training folds).
Path F1 counts HIGH paths analyzed as TP and LOW paths analyzed as FP, as for the agent.

Needs the agent rows of cv/score_test_folds.py.
Usage:  python experiments/baselines/path_baselines.py balanced|imbalanced
Output: results/baselines/path_baselines_<set>.json
"""
import os, sys, json, glob, re, logging, numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import xgboost as xgb, lightgbm as lgb
from scipy import stats
logging.disable(logging.CRITICAL)
name = sys.argv[1]
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from env2_pool_manager import PoolManager
PDIR = str(config.PATH_DB[name])
folds = config.folds(name); fold_of = {a: k for k, f in enumerate(folds) for a in f}
# SmartTaintRL per fold (saved models, every path decided)
ours = []; NAN = {}
for k in range(1, 6):
    te = set(folds[k-1])
    R0 = json.load(open(config.out("agent_rows", f"{name}_fold{k}.json")))
    rows = [r for r in R0 if r["address"] in te]
    for r in rows: NAN[r["address"]] = r["analyzed"]
    tp, fp, fn = (sum(r[x] for r in rows) for x in ("tp", "fp", "fn")); an = sum(r["analyzed"] for r in rows); sk = sum(r["skipped"] for r in rows)
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1); ours.append(dict(P=P, R=R, F1=2 * P * R / max(P + R, 1e-9), pruning=sk / (an + sk)))
# raw path records: tokens of the path nodes, source/sink, functions
pm = PoolManager(); docs, meta, risk, F, A, prio = [], [], [], [], [], []
for a, k in fold_of.items():
    g = glob.glob(os.path.join(PDIR, f"{a}*_path_database.json"))
    if not g: continue
    for p in json.load(open(g[0]))["paths"]:
        b = p["basic_info"]; ge = p.get("graph_enrichment", {}); toks = []
        for n in ge.get("nodes_detail", []):
            toks += ["T_" + str(n.get("type"))] + re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[%^&|<>=!+\-*/]+", str(n.get("code", n.get("code_snippet", ""))))
        toks += ["SRC_" + str(b.get("source_type")), "SNK_" + str(b.get("sink_type"))] + ["FN_" + f for f in ge.get("functions_involved", [])]
        docs.append(" ".join(toks)); meta.append([b.get("path_length", 0) / 20.0, len(ge.get("functions_involved", []))])
        risk.append(b.get("risk_level")); F.append(k); A.append(a); prio.append(pm.calculate_priority_score(p))
risk, F, A, prio, meta = np.array(risk), np.array(F), np.array(A), np.array(prio, float), np.array(meta, float)
hl = np.isin(risk, ["HIGH", "LOW"]); print(f"[{name}] contracts {len(set(A))} paths {len(risk)}", flush=True)
def pm_(an, r):
    tp = np.sum(an & (r == "HIGH")); fp = np.sum(an & (r == "LOW")); fn = np.sum(~an & (r == "HIGH"))
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1); return dict(P=P, R=R, F1=2 * P * R / max(P + R, 1e-9))
# analyze the top `frac` share of paths in every contract
def topfrac(s, addr, frac=None):
    out = np.zeros(len(s), bool)
    for a in np.unique(addr):
        i = np.where(addr == a)[0]; n = int(np.ceil(frac * len(i))); out[i[np.argsort(-s[i])[:n]]] = True
    return out
res = {m: [] for m in ["Top-k", "Logistic Regression", "XGBoost", "LambdaMART", "Logistic Regression (own threshold)", "XGBoost (own threshold)"]}
for k in range(5):
    tr, te = F != k, F == k; frac = 1 - ours[k]["pruning"]   # same share as the agent in this fold
    v = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 1), min_df=3, max_features=20000, dtype=np.float32, sublinear_tf=True)
    Xtr = hstack([v.fit_transform([d for d, m in zip(docs, tr) if m]), csr_matrix(meta[tr])]).tocsr()
    Xte = hstack([v.transform([d for d, m in zip(docs, te) if m]), csr_matrix(meta[te])]).tocsr()
    ytr = risk[tr] == "HIGH"
    for mn in ("Logistic Regression", "XGBoost"):
        m = LogisticRegression(max_iter=5000) if mn.startswith("Log") else xgb.XGBClassifier(n_estimators=300, max_depth=6, eval_metric="logloss", n_jobs=2)
        m.fit(Xtr, ytr); s_tr = m.predict_proba(Xtr)[:, 1]; s_te = m.predict_proba(Xte)[:, 1]
        t = max(np.unique(np.quantile(s_tr[hl[tr]], np.linspace(0, 1, 201))), key=lambda c: pm_(s_tr >= c, risk[tr])["F1"])
        an = s_te >= t; res[mn + " (own threshold)"].append(dict(**pm_(an, risk[te]), pruning=float(1 - an.mean())))
        an = topfrac(s_te, A[te], frac); res[mn].append(dict(**pm_(an, risk[te]), pruning=float(1 - an.mean())))
    # LambdaMART: graded relevance HIGH=2, MEDIUM=1, LOW=0, one query group per contract
    rel = np.select([risk == "HIGH", risk == "MEDIUM"], [2, 1], 0)
    o = np.argsort(A[tr], kind="stable"); _, grp = np.unique(A[tr][o], return_counts=True)
    rk = lgb.LGBMRanker(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1).fit(Xtr[o], rel[tr][o], group=grp)
    an = topfrac(rk.predict(Xte), A[te], frac); res["LambdaMART"].append(dict(**pm_(an, risk[te]), pruning=float(1 - an.mean())))
    an = topfrac(prio[te], A[te], frac); res["Top-k"].append(dict(**pm_(an, risk[te]), pruning=float(1 - an.mean())))
    print(f" fold {k+1}: ours F1 {ours[k]['F1']:.3f} pr {ours[k]['pruning']:.3f} | " + " | ".join(f"{m} {res[m][-1]['F1']:.3f}/{res[m][-1]['pruning']:.3f}" for m in res if res[m]), flush=True)
res["SmartTaintRL"] = ours
summ = {}
for m, v in res.items():
    summ[m] = {x: [float(np.mean([o[x] for o in v])), float(np.std([o[x] for o in v]))] for x in ("P", "R", "F1", "pruning")}
    if m != "SmartTaintRL": summ[m]["p_vs_SmartTaintRL_F1"] = float(stats.ttest_rel([o["F1"] for o in ours], [o["F1"] for o in v]).pvalue)
json.dump(dict(dataset=name, summary=summ, per_fold=res), open(config.out("baselines", f"path_baselines_{name}.json"), "w"), indent=1)

for m, v in summ.items(): print(f"{m:38s} P {v['P'][0]:.3f} R {v['R'][0]:.3f} F1 {v['F1'][0]:.3f}+-{v['F1'][1]:.3f} pruning {v['pruning'][0]:.3f} p={v.get('p_vs_SmartTaintRL_F1','')}")
