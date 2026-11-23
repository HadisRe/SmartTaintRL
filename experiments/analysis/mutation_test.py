"""
Mutation test for Slither weak-prng (Section 5, appendix), on the safe contracts of the balanced set.

One function is appended to the main (last) contract of every safe contract and Slither is run again:
  M1 harmless: time bucketing with %, read-only, no randomness, no value  -> should stay safe
  M2 exploit : coin flip on keccak256(now, msg.sender) that pays out, no % -> vulnerable
  M3 exploit : coin flip on now % 2 that pays out (textbook pattern)       -> vulnerable
A contract counts only if the original and all three mutants compile.

Usage:  python experiments/analysis/mutation_test.py
Output: results/analysis/mutation/mutation_slither_safe.csv (mutants in results/analysis/mutation/sol/)
"""
import os, re, csv, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

OUT = str(config.out("analysis", "mutation", "sol", "x").parent)
IDX = config.source_index()
HAVE = sorted([f[6:] for f in os.listdir(os.path.expanduser("~/.solcx"))], key=lambda v: [int(x) for x in v.split('.')])
def pick(txt):
    m = re.search(r"pragma\s+solidity\s+([^;]+);", txt); pr = m.group(1).replace(" ", "") if m else "^0.4.24"
    v = re.search(r"(\d+)\.(\d+)\.(\d+)", pr).group(0); minor = ".".join(v.split(".")[:2])
    if v in HAVE and (pr[0].isdigit() or pr.startswith("=")): return v
    same = [h for h in HAVE if h.startswith(minor + ".")]
    return same[-1] if same else "0.4.26"
# keccak256 needs abi.encodePacked from 0.4.22 on
def hashexpr(v):
    a, b, c = map(int, v.split("."))
    return "keccak256(abi.encodePacked(now, msg.sender))" if (b, c) >= (4, 22) or b >= 5 else "keccak256(now, msg.sender)"
MUT = {
 "M1_harmless_bucket": lambda v: "\n    function currentHourBucket() public view returns (uint256) { return now - (now % 3600); }\n",
 "M2_exploit_no_mod":  lambda v: "\n    function luckyDraw() public payable { require(msg.value > 0); if (uint256(%s) > 2**255) { msg.sender.transfer(msg.value * 2); } }\n" % hashexpr(v),
 "M3_exploit_mod":     lambda v: "\n    function coinFlip() public payable { require(msg.value > 0); if (now % 2 == 0) { msg.sender.transfer(msg.value * 2); } }\n",
}
# position of the closing brace of the last contract
def last_contract_end(t):
    names = [m for m in re.finditer(r"^\s*contract\s+\w+[^{]*\{", t, flags=re.M)]
    if not names: return None
    i = names[-1].end(); depth = 1
    while i < len(t) and depth:
        if t[i] == "{": depth += 1
        elif t[i] == "}": depth -= 1
        i += 1
    return i - 1 if depth == 0 else None
def slither(path, v):
    q = subprocess.run(["slither", path, "--solc", os.path.expanduser(f"~/.solcx/solc-v{v}"), "--detect", "weak-prng", "--json", "-"], capture_output=True, text=True, timeout=300)
    try: d = json.loads(q.stdout)
    except Exception: return None
    if not d.get("success"): return None
    return sum(1 for x in d["results"].get("detectors", []) if x["check"] == "weak-prng")
def one(p):
    a = os.path.basename(p)[:-4]; t = open(p, errors="ignore").read(); v = pick(t); row = {"address": a, "solc": v}
    row["orig"] = slither(p, v)
    end = last_contract_end(t)
    for k, f in MUT.items():
        if end is None or row["orig"] is None: row[k] = None; continue
        mp = os.path.join(OUT, f"{a}_{k}.sol"); open(mp, "w").write(t[:end] + f(v) + t[end:])
        row[k] = slither(mp, v)
    return row
files = sorted(IDX[r["address"]] for r in config.contracts("balanced") if r["label"] == "safe")
with ThreadPoolExecutor(2) as ex: rows = list(ex.map(one, files))
with open(config.out("analysis", "mutation", "mutation_slither_safe.csv"), "w", newline="") as g:
    w = csv.DictWriter(g, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
ok = [r for r in rows if r["orig"] is not None and all(r[k] is not None for k in MUT)]
print("contracts usable (original + all 3 mutants compile):", len(ok), "of", len(rows))
print("original safe contract flagged:", sum(r["orig"] > 0 for r in ok))
for k in MUT:
    new = sum(r[k] > r["orig"] for r in ok)   # Slither reports MORE weak-prng findings than on the original
    print(f"{k:20s} Slither raises a new weak-prng finding in {new}/{len(ok)} ({100*new/len(ok):.1f}%)")
