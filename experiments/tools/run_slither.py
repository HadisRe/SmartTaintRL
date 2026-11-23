"""
Slither with only the weak-prng detector (the setting used in the TaintSentinel paper).
A contract is reported vulnerable if weak-prng fires at least once.
The solc version is taken from the pragma (installed versions in ~/.solcx, e.g. with py-solc-x).
With --solc X.Y.Z every contract is compiled with that version instead (Section 5, 0.4.24 and 0.5.17).

Usage:  python experiments/tools/run_slither.py balanced|imbalanced [--solc 0.4.24] [--budget seconds]
Output: results/tools/slither_weakprng_<set>.csv or slither_weakprng_<set>_solc<X.Y.Z>.csv
        (resumable: finished contracts are skipped)
"""
import os, re, csv, json, subprocess, time, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

args = sys.argv[1:]
NAME = args[0]
FIX = args[args.index("--solc") + 1] if "--solc" in args else None
BUDGET = float(args[args.index("--budget") + 1]) if "--budget" in args else 1e9
OUT = config.out("tools", f"slither_weakprng_{NAME}" + (f"_solc{FIX}" if FIX else "") + ".csv")
SOLCX = os.path.expanduser("~/.solcx")
HAVE = sorted([f[6:] for f in os.listdir(SOLCX)], key=lambda v: [int(x) for x in v.split('.')])
IDX = config.source_index()


def pick(txt):
    """Exact version for a pinned pragma, otherwise the newest installed patch of that minor version."""
    m = re.search(r"pragma\s+solidity\s+([^;]+);", txt); pr = m.group(1).replace(" ", "") if m else "^0.4.24"
    v = re.search(r"(\d+)\.(\d+)\.(\d+)", pr).group(0); minor = ".".join(v.split(".")[:2])
    if v in HAVE and (pr[0].isdigit() or pr.startswith("=")): return v
    same = [h for h in HAVE if h.startswith(minor + ".")]
    return same[-1] if same else "0.4.26"


def one(r):
    a = r["address"]; p = IDX[a]; v = FIX or pick(open(p, errors="ignore").read()); t0 = time.time()
    try:
        q = subprocess.run(["slither", p, "--solc", os.path.join(SOLCX, f"solc-v{v}"), "--detect", "weak-prng", "--json", "-"],
                           capture_output=True, text=True, timeout=300)
        d = json.loads(q.stdout) if q.stdout.strip() else {}
        ok = bool(d.get("success"))
        hits = sum(1 for x in (d.get("results", {}) or {}).get("detectors", []) or [] if x.get("check") == "weak-prng")
        err = "" if ok else str(d.get("error") or q.stderr)[-150:].replace("\n", " ")
    except subprocess.TimeoutExpired: ok, hits, err = False, 0, "timeout"
    except Exception as e: ok, hits, err = False, 0, str(e)[:150]
    return [a, r["label"], r["fold"], "vulnerable" if hits else "safe", hits, ok, round(time.time() - t0, 2), v, err]


rows = config.contracts(NAME)
done = {r["address"] for r in csv.DictReader(open(OUT))} if OUT.exists() else set()
if not done:
    csv.writer(open(OUT, "w", newline="")).writerow(["address", "label", "fold", "pred", "weak_prng_hits", "ok", "seconds", "solc", "err"])
todo = [r for r in rows if r["address"] not in done]; deadline = time.time() + BUDGET
with ThreadPoolExecutor(2) as ex:
    for i in range(0, len(todo), 2):
        if time.time() > deadline: break
        for row in ex.map(one, todo[i:i + 2]):
            with open(OUT, "a", newline="") as g: csv.writer(g).writerow(row)
print("done", sum(1 for _ in open(OUT)) - 1, "of", len(rows))
