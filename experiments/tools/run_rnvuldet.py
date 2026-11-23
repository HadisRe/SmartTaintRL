"""
RNVulDet on the bytecode of rnvuldet_compile.py, with the authors' call and settings:
python main.py <file.hex>, 300 s limit, vulnerable if is_reported. A contract without bytecode
(compilation failed) or a failed run counts as safe.

Set RNVULDET_DIR to the RNVulDet checkout (the folder with main.py).
Usage:  python experiments/tools/run_rnvuldet.py balanced|imbalanced
Output: results/tools/rnvuldet_<set>.csv   (resumable)
"""
import os, sys, csv, json, time, subprocess
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

SET = sys.argv[1]
MAIN = os.path.join(os.environ.get("RNVULDET_DIR", "RNVulDet"), "main.py")
BC = config.out("tools", "rnvuldet_bytecode", "x").parent
OUT = config.out("tools", f"rnvuldet_{SET}.csv")
HEAD = ["address", "ground_truth", "fold", "predicted", "execution_time", "success", "error",
        "steps", "conditions", "call_values", "to_addresses"]


def one(r):
    a = r["address"]; f = BC / (a + ".hex"); t0 = time.time()
    if not f.exists(): return [a, r["label"], r["fold"], "safe", 0, False, "no bytecode (compile failed)", 0, 0, 0, 0]
    try:
        p = subprocess.run([sys.executable, MAIN, str(f)], capture_output=True, text=True, timeout=300)
        dt = time.time() - t0
        if p.returncode == 0 and p.stdout:
            o = json.loads(p.stdout)
            return [a, r["label"], r["fold"], "vulnerable" if o.get("is_reported") else "safe", round(dt, 2), True, "",
                    o.get("steps", 0), o.get("conditions", 0), o.get("call_values", 0), o.get("to_addresses", 0)]
        return [a, r["label"], r["fold"], "safe", round(dt, 2), False, (p.stderr or "error")[-150:].replace("\n", " "), 0, 0, 0, 0]
    except subprocess.TimeoutExpired:
        return [a, r["label"], r["fold"], "safe", 300, False, "Timeout (>300s)", 0, 0, 0, 0]


rows = config.contracts(SET)
done = {r["address"] for r in csv.DictReader(open(OUT))} if OUT.exists() else set()
if not done: csv.writer(open(OUT, "w", newline="")).writerow(HEAD)
todo = [r for r in rows if r["address"] not in done]
with ThreadPoolExecutor(2) as ex:
    for i in range(0, len(todo), 2):
        for row in ex.map(one, todo[i:i + 2]):
            with open(OUT, "a", newline="") as g: csv.writer(g).writerow(row)
print("done", sum(1 for _ in open(OUT)) - 1, "of", len(rows))
