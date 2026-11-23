"""
Mythril 0.24.8 (Table 8, balanced set).

First pass: the main (last-declared) contract, solc version from the pragma, --execution-timeout 120,
wall-clock limit 300 s. Runs that fail or time out are retried once on the whole file with a 600 s limit.
A contract is reported vulnerable if SWC-120 or SWC-116 is reported.

The solc binaries are taken from ~/.solcx (py-solc-x). --solv is not used, because the solc download
inside Mythril points to solc-bin.ethereum.org, which no longer resolves; the right binary is put first on PATH.

Usage:  python experiments/tools/run_mythril.py balanced [parallel runs, default 2]
Output: results/tools/mythril_<set>.csv and mythril_<set>_retry.csv   (resumable)
"""
import os, re, csv, json, time, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

NAME = sys.argv[1] if len(sys.argv) > 1 else "balanced"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2
SOLCX = os.path.expanduser("~/.solcx")
HAVE = sorted([f[6:] for f in os.listdir(SOLCX)], key=lambda v: [int(x) for x in v.split('.')])
IDX = config.source_index()
HEAD = ["address", "label", "fold", "pred", "hits", "ok", "timeout", "seconds", "solc", "err"]


def pick(txt):
    """Exact version for a pinned pragma, otherwise the newest installed patch of that minor version."""
    m = re.search(r"pragma\s+solidity\s+([^;]+);", txt); pr = m.group(1).replace(" ", "") if m else "^0.4.24"
    v = re.search(r"(\d+)\.(\d+)\.(\d+)", pr).group(0); minor = ".".join(v.split(".")[:2])
    if pr[0].isdigit() and v in HAVE: return v
    same = [h for h in HAVE if h.startswith(minor + ".")]
    return same[-1] if same else "0.4.26"


def bindir(v):
    """A folder that holds only `solc` -> ~/.solcx/solc-v<v>, to be put first on PATH."""
    d = os.path.expanduser(f"~/.solc_bins/{v}"); os.makedirs(d, exist_ok=True)
    exe = os.path.join(d, "solc")
    if not os.path.exists(exe): os.symlink(os.path.join(SOLCX, f"solc-v{v}"), exe)
    return d


def main_contract(txt):
    t = re.sub(r"//.*?$|/\*.*?\*/", " ", txt, flags=re.S | re.M)
    names = re.findall(r"^\s*contract\s+(\w+)", t, flags=re.M)
    return names[-1] if names else None


def run(r, whole_file, cap):
    a = r["address"]; p = IDX[a]; t0 = time.time()
    txt = open(p, errors="ignore").read(); v = pick(txt)
    mc = None if whole_file else main_contract(txt)
    tgt = f"{p}:{mc}" if mc else p
    try:
        env = dict(os.environ, PATH=bindir(v) + os.pathsep + os.environ.get("PATH", ""))
        out = subprocess.run(["myth", "analyze", tgt, "-o", "json", "--execution-timeout", "120"],
                             capture_output=True, text=True, timeout=cap, env=env)
        s = out.stdout; i = s.find("{"); d = json.loads(s[i:]) if i >= 0 else {}
        ok = bool(d.get("success"))
        hits = sum(1 for x in d.get("issues", []) if str(x.get("swc-id")) in ("120", "116"))
        to, err = False, "" if ok else (out.stderr or s)[-150:].replace("\n", " ")
    except subprocess.TimeoutExpired:
        ok, hits, to, err = False, 0, True, "timeout"
    except Exception as e:
        ok, hits, to, err = False, 0, False, str(e)[:150]
    return [a, r["label"], r["fold"], "vulnerable" if hits else "safe", hits, ok, to, round(time.time() - t0, 1), v, err]


def stage(fname, todo_rows, whole_file, cap):
    out = config.out("tools", fname)
    done = {x["address"] for x in csv.DictReader(open(out))} if out.exists() else set()
    if not done: csv.writer(open(out, "w", newline="")).writerow(HEAD)
    todo = [r for r in todo_rows if r["address"] not in done]
    print(f"{fname}: {len(done)} done, {len(todo)} to go", flush=True)
    with ThreadPoolExecutor(N) as ex:
        for row in ex.map(lambda r: run(r, whole_file, cap), todo):
            with open(out, "a", newline="") as f: csv.writer(f).writerow(row)
            print(row[:8], flush=True)
    return {x["address"]: x for x in csv.DictReader(open(out))}


rows = config.contracts(NAME)
first = stage(f"mythril_{NAME}.csv", rows, False, 300)
stage(f"mythril_{NAME}_retry.csv", [r for r in rows if first[r["address"]]["ok"] != "True"], True, 600)
print("finished", flush=True)
