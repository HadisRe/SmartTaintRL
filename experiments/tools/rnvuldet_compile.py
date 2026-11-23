"""
Runtime bytecode for RNVulDet, which analyzes bytecode.

Each contract is compiled from the dataset source with the solc version of its pragma, and the runtime
bytecode of the main (last-declared, non-empty) contract is kept, the same contract choice as for Mythril.
Library link placeholders (__<36 chars>__) are replaced by a zero address so that the hex is valid.

Usage:  python experiments/tools/rnvuldet_compile.py
Output: results/tools/rnvuldet_bytecode/<address>.hex and compile_log.json
"""
import os, re, json, subprocess, sys, collections
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

SOLCX = os.path.expanduser("~/.solcx")
HAVE = sorted([f[6:] for f in os.listdir(SOLCX)], key=lambda v: [int(x) for x in v.split('.')])
OUT = config.out("tools", "rnvuldet_bytecode", "x").parent
IDX = config.source_index()


def pick(txt):
    m = re.search(r"pragma\s+solidity\s+([^;]+);", txt); pr = m.group(1).replace(" ", "") if m else "^0.4.24"
    v = re.search(r"(\d+)\.(\d+)\.(\d+)", pr).group(0); minor = ".".join(v.split(".")[:2])
    if v in HAVE and (pr[0].isdigit() or pr.startswith("=")): return v
    same = [h for h in HAVE if h.startswith(minor + ".")]
    return same[-1] if same else None


def names(txt):
    t = re.sub(r"//.*?$|/\*.*?\*/", " ", txt, flags=re.S | re.M)
    return re.findall(r"^\s*contract\s+(\w+)", t, flags=re.M)


def one(a):
    dst = OUT / (a + ".hex")
    if dst.exists(): return a, "cached"
    if a not in IDX: return a, "no source"
    txt = open(IDX[a], errors="ignore").read(); v = pick(txt)
    if not v: return a, "no solc for pragma"
    r = subprocess.run([os.path.join(SOLCX, f"solc-v{v}"), "--combined-json", "bin-runtime", IDX[a]],
                       capture_output=True, text=True, timeout=120)
    try: cs = json.loads(r.stdout)["contracts"]
    except Exception: return a, "compile error " + v
    rt = {k.split(":")[-1]: c.get("bin-runtime", "") for k, c in cs.items()}
    order = [n for n in reversed(names(txt)) if rt.get(n)]
    best = order[0] if order else max(rt, key=lambda k: len(rt[k]))
    if not rt.get(best): return a, "empty runtime"
    open(dst, "w").write(re.sub(r"__.{36}__", "0" * 40, rt[best]))
    return a, f"ok {v} {best}"


addrs = sorted({r["address"] for n in ("balanced", "imbalanced") for r in config.contracts(n)})
with ThreadPoolExecutor(2) as ex:
    log = dict(ex.map(one, addrs))
json.dump(log, open(OUT / "compile_log.json", "w"), indent=0)
print(collections.Counter(s.split()[0] if s.startswith(("ok", "compile")) else s for s in log.values()))
