"""
Fig. 2 / Section 4.2.3: share of HIGH-risk paths per randomness source in the vulnerable contracts
of the training folds only (dot = mean over the five training splits, bar = min-max over the splits).

Usage:  python experiments/analysis/source_prevalence.py
Output: results/figures/fig_source_prevalence.pdf/.png and source_prevalence_trainfolds.json
"""
import os, sys, json, collections, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.serif": ["DejaVu Serif"], "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0})
SETS = [("Balanced (284)", config.PATH_DB["balanced"], config.PROFILES["balanced"], "balanced", "#2a78d6"),
        ("Imbalanced (2,549)", config.PATH_DB["imbalanced"], config.PROFILES["imbalanced"], "imbalanced", "#eb6834")]
SRC = ["timestamp", "blocknumber", "blockhash", "difficulty", "sender", "other"]
res = {}
for name, pd, prof, ff, col in SETS:
    folds = config.folds(ff); per = {}
    for a in [a for f in folds for a in f]:
        c = collections.Counter()
        if json.load(open(f"{prof}/{a}_profile.json"))["label"] == "vulnerable":
            for p in json.load(open(f"{pd}/{a}_path_database.json"))["paths"]:
                if p["basic_info"].get("risk_level") == "HIGH":
                    s = p["basic_info"].get("source_type"); c[s if s in SRC[:-1] else "other"] += 1
        per[a] = c
    # shares over the four training folds of each split
    shares = []
    for k in range(5):
        tot = collections.Counter()
        for i, f in enumerate(folds):
            if i != k:
                for a in f: tot += per[a]
        n = sum(tot.values()); shares.append([100 * tot[s] / n for s in SRC])
    res[name] = np.array(shares)
json.dump({k: v.tolist() for k, v in res.items()}, open(config.out("figures", "source_prevalence_trainfolds.json"), "w"), indent=1)
fig, ax = plt.subplots(figsize=(5.6, 2.8)); y = np.arange(len(SRC))[::-1]
for j, (name, *_, col) in enumerate(SETS):
    a = res[name]; off = 0.16 if j == 0 else -0.16
    ax.hlines(y + off, a.min(0), a.max(0), color=col, lw=2.2, alpha=0.45, capstyle="round")
    ax.plot(a.mean(0), y + off, "o", ms=6.5, color=col, mec="white", mew=1.2, label=name, zorder=3)
    for yy, m in zip(y + off, a.mean(0)): ax.text(a.max(0)[list(y + off).index(yy)] + 1.2, yy, f"{m:.0f}%", va="center", fontsize=7.5, color="#444")
ax.set_yticks(y); ax.set_yticklabels([{"other": "other sources", "sender": "msg.sender"}.get(s, s) for s in SRC]); ax.set_xlim(0, 65)
ax.set_xlabel("Share of HIGH-risk paths in vulnerable training contracts (%)")
ax.grid(axis="x", color="#e3e3e3", lw=0.6); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
ax.legend(frameon=False, loc="lower right", fontsize=8, handletextpad=0.3)
fig.tight_layout(); fig.savefig(config.out("figures", "fig_source_prevalence.pdf")); fig.savefig(config.out("figures", "fig_source_prevalence.png"), dpi=200)
for n, a in res.items(): print(n, dict(zip(SRC, np.round(a.mean(0), 1))), "range", dict(zip(SRC, np.round(a.max(0) - a.min(0), 1))))
