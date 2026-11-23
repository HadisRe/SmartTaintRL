"""
Context test (Table 9), imbalanced set.

The "pattern" is a block variable and a modulo (%) in the same statement, which is what
weak-prng looks for. Two groups are reported:
  - safe contracts that contain the pattern  (a method that reads context should not flag them)
  - vulnerable contracts without the pattern (a method that reads context should still find them)

SmartTaintRL = the contract-level predictions of contract_head.py ("pre-registered selection").
Slither = weak-prng predictions of tools/run_slither.py.

Usage:  python experiments/analysis/context_test.py
"""
import csv, json, re, os, sys
from math import sqrt
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

NAME = "imbalanced"
IDX = config.source_index()
fold = {r["address"]: int(r["fold"]) for r in config.contracts(NAME)}
lab = {r["address"]: r["label"] == "vulnerable" for r in config.contracts(NAME)}
ours = json.load(open(config.out("contract_level", f"pred_{NAME}.json")))["pre-registered selection"]
S = {r["address"]: r for r in csv.DictReader(open(config.out("tools", f"slither_weakprng_{NAME}.csv")))}
A = [a for a in lab if a in ours]
sl = {a: S[a]["pred"] == "vulnerable" for a in A}

KW = re.compile(r"block\.timestamp|\bnow\b|block\.number|blockhash|block\.difficulty|block\.coinbase")
def strip(t): return re.sub(r"//.*?$|/\*.*?\*/", " ", t, flags=re.S | re.M)
pat = {a: any(KW.search(s) and "%" in s for s in strip(open(IDX[a], errors="ignore").read()).split(";")) for a in A}


def m(p, X):
    tp = sum(lab[a] and p[a] for a in X); fp = sum((not lab[a]) and p[a] for a in X); fn = sum(lab[a] and not p[a] for a in X)
    tn = len(X) - tp - fp - fn
    P = tp / (tp + fp) if tp + fp else 0; R = tp / (tp + fn) if tp + fn else 0; F = 2 * P * R / (P + R) if P + R else 0
    mcc = (tp * tn - fp * fn) / sqrt(max(1, (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return dict(n=len(X), tp=tp, fp=fp, fn=fn, tn=tn, P=round(P, 3), R=round(R, 3), F1=round(F, 3), mcc=round(mcc, 3))


print("contracts:", len(A), " Slither failures:", sum(S[a]["ok"] != "True" for a in A))
for n, p in [("Slither weak-prng", sl), ("SmartTaintRL", ours)]:
    print(f"{n:18s}", m(p, A))
fs = [m(sl, [a for a in A if fold[a] == k])["F1"] for k in range(1, 6)]
fo = [m(ours, [a for a in A if fold[a] == k])["F1"] for k in range(1, 6)]
print("per-fold F1  Slither", fs, " SmartTaintRL", fo, " paired t-test p=%.4f" % stats.ttest_rel(fo, fs).pvalue)

SA = [a for a in A if not lab[a] and pat[a]]; VB = [a for a in A if lab[a] and not pat[a]]
print("\nTable 9")
print("safe, pattern present:      %3d   flagged by Slither %3d, by SmartTaintRL %3d" % (len(SA), sum(sl[a] for a in SA), sum(ours[a] for a in SA)))
print("vulnerable, pattern absent: %3d   found by Slither   %3d, by SmartTaintRL %3d" % (len(VB), sum(sl[a] for a in VB), sum(ours[a] for a in VB)))
