"""Where the experiment scripts look for data, models and results.

Set SMARTTAINTRL_DATA to the folder that holds the dataset (default: <repo>/data).
Expected layout inside it:
    path_databases_balanced/      *_path_database.json   (284 contracts)
    contract_profiles_balanced/   *_profile.json
    path_databases_imbalanced/    *_path_database.json   (2,549 contracts)
    contract_profiles_imbalanced/ *_profile.json
    contracts/vulnerable/*.sol    contracts/safe/*.sol   (Solidity sources, named <address>.sol)
    path_databases_node_level/    *_enriched_paths.json  (localization)
    contract_ast1_clean/          *_semantic_graph.json  (localization)
"""
import csv
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("SMARTTAINTRL_DATA", REPO / "data"))
RESULTS = Path(os.environ.get("SMARTTAINTRL_RESULTS", REPO / "results"))
MODELS = REPO / "models"
FOLDS_DIR = REPO / "experiments" / "folds"

PATH_DB = {"balanced": DATA / "path_databases_balanced", "imbalanced": DATA / "path_databases_imbalanced"}
PROFILES = {"balanced": DATA / "contract_profiles_balanced", "imbalanced": DATA / "contract_profiles_imbalanced"}
SOL_DIRS = [DATA / "contracts" / "vulnerable", DATA / "contracts" / "safe"]
NODE_DB = DATA / "path_databases_node_level"
SEMANTIC_GRAPHS = DATA / "contract_ast1_clean"
LOC_MODEL = MODELS / "trained_dqn_model_2500.pth"   # model used for localization (Section 6)

# the agent code lives in src/rl_agent
sys.path.insert(0, str(REPO / "src" / "rl_agent"))


def folds(name):
    """The five stratified folds used for every method in the paper (lists of addresses)."""
    return json.load(open(FOLDS_DIR / f"folds_{name}.json"))["folds"]


def contracts(name):
    """Rows (address, label, fold 1-5) of a dataset."""
    return list(csv.DictReader(open(FOLDS_DIR / f"contracts_{name}.csv")))


def model(name, k):
    """Checkpoint of the agent trained on all folds except fold k (k = 1..5)."""
    return MODELS / f"dqn_{name}_fold{k}.pth"


def source_index():
    """address -> path of its .sol file."""
    idx = {}
    for d in SOL_DIRS:
        for p in d.rglob("*.sol"):
            idx.setdefault(p.stem, str(p))
    return idx


def out(*parts):
    """Path inside the results folder (created if needed)."""
    p = RESULTS.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
