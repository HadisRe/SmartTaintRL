"""
Wall-clock time of path extraction per contract (Section 5): AST with solc -> sources -> sinks ->
semantic graph -> taint paths, with the TaintSentinel preprocessing scripts (complete_parser1, source2,
sink3, ModularSemanticGraph1, TaintAnalysis). Random sample of 40 contracts of the paper.
solc binaries are taken from ~/.solcx (nothing is downloaded).

Set TAINTSENTINEL_DIR to the folder with those scripts.
Usage:  python experiments/analysis/time_pipeline.py
Output: results/analysis/timing/taintsentinel_timing.csv
"""
import sys, os, io, glob, json, time, random, shutil, contextlib, statistics as st, csv
import solcx
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
sys.path.insert(0, os.environ.get("TAINTSENTINEL_DIR", "TaintSentinel"))
# use the installed solc versions only; an unknown version falls back to the newest patch of its minor
HAVE = sorted([f[6:] for f in os.listdir(os.path.expanduser("~/.solcx"))], key=lambda v: [int(x) for x in v.split('.')])
solcx.install_solc = lambda v, *a, **k: None
_set = solcx.set_solc_version
def safe_set(v, *a, **k):
    v = str(v)
    if v not in HAVE:
        same = [h for h in HAVE if h.startswith(".".join(v.split(".")[:2]) + ".")]; v = same[-1] if same else "0.4.26"
    return _set(v, *a, **k)
solcx.set_solc_version = safe_set
import complete_parser1, source2, sink3, ModularSemanticGraph1 as MSG, TaintAnalysis as TA
complete_parser1.install_solc = solcx.install_solc; complete_parser1.set_solc_version = safe_set
import logging; logging.disable(logging.CRITICAL)
files = sorted(set(config.source_index().values())); random.seed(0); sample = random.sample(files, 40)
WORK = str(config.out("analysis", "timing", "work", "x").parent)
rows = []
for f in sample:
    a = os.path.basename(f)[:-4]; d = f"{WORK}/{a}"; shutil.rmtree(d, ignore_errors=True); os.makedirs(d + "/src"); os.makedirs(d + "/out")
    shutil.copy(f, d + "/src/"); t = {}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        t0 = time.time(); complete_parser1.ImprovedSolidityASTBuilder().process_directory(d + "/src", d + "/out"); t["ast"] = time.time() - t0
        t0 = time.time(); source2.update_json_with_sources(d + "/out", d + "/src"); sink3.update_json_with_sinks(d + "/out", d + "/src"); t["src_sink"] = time.time() - t0
        t0 = time.time(); MSG.process_all_contracts(d + "/out", d + "/src"); t["graph"] = time.time() - t0
        t0 = time.time()
        try: TA.TaintAnalyzer(a, d + "/out").analyze()
        except Exception: pass
        t["taint"] = time.time() - t0
    ok = os.path.exists(f"{d}/out/{a}_semantic_graph.json"); t["total"] = sum(t.values()); t["ok"] = ok; t["address"] = a; rows.append(t)
    print(a[:10], {k: round(v, 2) for k, v in t.items() if isinstance(v, float)}, ok, flush=True)
good = [r for r in rows if r["ok"]]
with open(config.out("analysis", "timing", "taintsentinel_timing.csv"), "w", newline="") as g:
    w = csv.DictWriter(g, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("ok", len(good), "of", len(rows)); [print(k, "mean %.2f s  median %.2f s" % (st.mean(r[k] for r in good), st.median(r[k] for r in good))) for k in ["ast", "src_sink", "graph", "taint", "total"]]
