"""
Table 8 and the appendix table: precision, recall, F1 and MCC of every method on the same
contracts and folds, with a paired t-test over the five folds against SmartTaintRL.

A failed or timed-out tool run counts as "safe" (not flagged). Tools whose output file is
missing are left out.

Usage:  python experiments/analysis/table8_stats.py
"""
import csv, json, math, os, sys, numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def met(y, p):
    y, p = np.array(y), np.array(p); tp = int((y & p).sum()); fp = int((~y & p).sum()); fn = int((y & ~p).sum()); tn = len(y) - tp - fp - fn
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1); F = 2 * P * R / max(P + R, 1e-9)
    mcc = (tp * tn - fp * fn) / math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    return dict(tp=tp, fp=fp, fn=fn, P=round(P, 3), R=round(R, 3), F1=round(F, 3), MCC=round(mcc, 3))


def csv_pred(fname, col="pred"):
    f = config.out("tools", fname)
    return {r["address"]: r for r in csv.DictReader(open(f))} if f.exists() else None


for name in ["balanced", "imbalanced"]:
    C = config.contracts(name)
    lab = {r["address"]: r["label"] == "vulnerable" for r in C}; fold = {r["address"]: int(r["fold"]) for r in C}; A = list(lab)
    tools = {"SmartTaintRL": json.load(open(config.out("contract_level", f"pred_{name}.json")))["pre-registered selection"]}

    S = csv_pred(f"slither_weakprng_{name}.csv")
    if S: tools["Slither"] = {a: S[a]["pred"] == "vulnerable" for a in A}
    M = csv_pred(f"mythril_{name}.csv")
    if M:
        M.update(csv_pred(f"mythril_{name}_retry.csv") or {})   # retried runs replace the first pass
        tools["Mythril"] = {a: M[a]["pred"] == "vulnerable" for a in A}
    RN = csv_pred(f"rnvuldet_{name}.csv")
    if RN: tools["RNVulDet"] = {a: RN[a]["predicted"] == "vulnerable" for a in A}
    TS = {}
    for k in range(1, 6):
        f = config.out("tools", "taintsentinel", f"{name}_fold{k}.json")
        if f.exists():
            for r in json.load(open(f))["rows"]: TS[r["address"]] = r["pred"]
    if len(TS) == len(A): tools["TaintSentinel"] = TS

    per_fold = lambda t: [met([lab[a] for a in A if fold[a] == k], [t[a] for a in A if fold[a] == k])["F1"] for k in range(1, 6)]
    ours = per_fold(tools["SmartTaintRL"])
    print(f"== {name} ({len(A)} contracts)")
    for n, t in tools.items():
        p = "" if n == "SmartTaintRL" else "p=%.4f" % stats.ttest_rel(ours, per_fold(t)).pvalue
        print(f"  {n:14s}", met([lab[a] for a in A], [t[a] for a in A]), p)
