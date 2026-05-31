
import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Make the localizer importable when this file is run directly.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from localizer import VulnerabilityLocalizer, compute_metrics


# ---------------------------------------------------------------------------
# Default file locations. Override on the command line if you keep the
# pre-processed databases somewhere else.
# ---------------------------------------------------------------------------
_DEFAULT_BASE = Path(__file__).resolve().parents[2] / "data"
DEFAULT_MODEL_PATH = _DEFAULT_BASE / "trained_dqn_model_2500.pth"
DEFAULT_GT_PATH = _HERE / "data" / "ground_truth" / "final_dataset_14_contracts.pkl"
DEFAULT_PATH_DB_DIR = _DEFAULT_BASE / "path_databases_updated"
DEFAULT_AST_DIR = _DEFAULT_BASE / "contract_ast1_clean"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_ground_truth(path: Path) -> Dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


def _extract_pred_nodes(node_list) -> List[str]:
    """`localize_nodes` returns (node_id, score) tuples; tests want IDs."""
    out = []
    for item in node_list or []:
        if isinstance(item, tuple):
            out.append(item[0])
        else:
            out.append(item)
    return out


def evaluate_configuration(localizer: VulnerabilityLocalizer,
                           gt_data: Dict,
                           path_db_dir: Path,
                           ast_dir: Path,
                           verbose: bool = True) -> Dict:
    """Run the localizer on every contract and aggregate metrics."""
    results: List[Dict] = []
    exact_matches = 0
    caller_matches = 0
    wrong = 0
    node_precisions: List[float] = []
    node_recalls: List[float] = []

    for idx, (address, info) in enumerate(gt_data.items(), 1):
        contract_id = f"C{idx}"
        gt_function = info.get('function', '')
        gt_nodes = info.get('nodes') or info.get('vulnerable_nodes') or []

        path_db_file = path_db_dir / f"{address}_enriched_paths.json"
        path_db = localizer.load_path_database(str(path_db_file))
        if not path_db:
            if verbose:
                print(f"  {contract_id}: path_database not found ({path_db_file.name})")
            results.append({'id': contract_id, 'gt': gt_function, 'pred': None,
                            'type': 'Missing', 'p5': 0.0, 'r5': 0.0})
            wrong += 1
            continue

        sg_file = ast_dir / f"{address}_semantic_graph.json"
        semantic_graph = localizer.load_semantic_graph(str(sg_file))

        loc_result = localizer.localize(path_db, semantic_graph,
                                        top_k_functions=5, top_k_nodes=5)
        ranked_funcs = loc_result.get('functions', []) or []
        call_graph = loc_result.get('call_graph', {}) or {}

        if not ranked_funcs:
            results.append({'id': contract_id, 'gt': gt_function, 'pred': None,
                            'type': 'Wrong', 'p5': 0.0, 'r5': 0.0})
            wrong += 1
            continue

        func_names = [f for f, _ in ranked_funcs]
        predicted = func_names[0]
        pred_nodes = _extract_pred_nodes(loc_result.get('nodes', {}).get(predicted, []))

        if predicted == gt_function:
            match_type = 'Exact'
            exact_matches += 1
        else:
            match_type = 'Wrong'
            for func in func_names:
                if localizer.check_caller_relationship(call_graph, func, gt_function):
                    match_type = 'Caller'
                    caller_matches += 1
                    break
            if match_type == 'Wrong':
                wrong += 1

        if match_type == 'Exact' and gt_nodes:
            m = compute_metrics(gt_nodes, pred_nodes, k=5)
            node_precisions.append(m['precision'])
            node_recalls.append(m['recall'])
            p5, r5 = m['precision'], m['recall']
        else:
            p5, r5 = 0.0, 0.0

        results.append({'id': contract_id, 'gt': gt_function, 'pred': predicted,
                        'type': match_type, 'p5': p5, 'r5': r5})

    total = len(gt_data)
    summary = {
        'total': total,
        'exact': exact_matches,
        'caller': caller_matches,
        'wrong': wrong,
        'strict_acc': exact_matches / total if total else 0.0,
        'relaxed_acc': (exact_matches + caller_matches) / total if total else 0.0,
    }
    if node_precisions:
        avg_p = sum(node_precisions) / len(node_precisions)
        avg_r = sum(node_recalls) / len(node_recalls)
        avg_f1 = (2 * avg_p * avg_r / (avg_p + avg_r)
                  if (avg_p + avg_r) > 0 else 0.0)
        summary.update({'node_p5': avg_p, 'node_r5': avg_r, 'node_f1': avg_f1,
                        'node_n': len(node_precisions)})
    else:
        summary.update({'node_p5': 0.0, 'node_r5': 0.0, 'node_f1': 0.0,
                        'node_n': 0})

    return {'summary': summary, 'per_contract': results}


def print_summary(name: str, summary: Dict) -> None:
    print(f"  {name:<28}"
          f"strict={summary['strict_acc']*100:5.1f}%   "
          f"relaxed={summary['relaxed_acc']*100:5.1f}%   "
          f"P@5={summary['node_p5']:.2f}  "
          f"R@5={summary['node_r5']:.2f}  "
          f"F1@5={summary['node_f1']:.2f}  "
          f"(n={summary['node_n']})")


def print_detailed(per_contract: List[Dict]) -> None:
    print("\n  Detailed per-contract results:")
    print(f"    {'ID':<5}{'GT Function':<23}{'Predicted':<23}"
          f"{'Type':<10}{'P@5':<6}{'R@5':<6}")
    print("    " + "-" * 70)
    for r in per_contract:
        print(f"    {r['id']:<5}{str(r['gt'])[:22]:<23}"
              f"{str(r['pred'])[:22]:<23}{r['type']:<10}"
              f"{r['p5']:.2f}  {r['r5']:.2f}")


# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------

ABLATION_CONFIGS: List[Tuple[str, Dict[str, float]]] = [
    ('Full (alpha=beta=gamma=delta=1)', dict(alpha=1, beta=1, gamma=1, delta=1)),
    ('Gradient only',                   dict(alpha=1, beta=0, gamma=0, delta=0)),
    ('Attention only',                  dict(alpha=0, beta=1, gamma=0, delta=0)),
    ('Propagation only',                dict(alpha=0, beta=0, gamma=1, delta=0)),
    ('Centrality only',                 dict(alpha=0, beta=0, gamma=0, delta=1)),
    ('No gradient',                     dict(alpha=0, beta=1, gamma=1, delta=1)),
    ('No attention',                    dict(alpha=1, beta=0, gamma=1, delta=1)),
    ('No propagation',                  dict(alpha=1, beta=1, gamma=0, delta=1)),
    ('No centrality',                   dict(alpha=1, beta=1, gamma=1, delta=0)),
]


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SmartTaintRL localization (with ablation).")
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL_PATH,
                        help="Path to trained DQN checkpoint (.pth).")
    parser.add_argument('--ground-truth', type=Path, default=DEFAULT_GT_PATH,
                        help="Path to final_dataset_14_contracts.pkl.")
    parser.add_argument('--path-db-dir', type=Path, default=DEFAULT_PATH_DB_DIR,
                        help="Directory of *_path_database.json files.")
    parser.add_argument('--ast-dir', type=Path, default=DEFAULT_AST_DIR,
                        help="Directory of *_semantic_graph.json files.")
    parser.add_argument('--ablation', action='store_true',
                        help="Run the full ablation study.")
    parser.add_argument('--quiet', action='store_true',
                        help="Suppress per-contract listing.")
    args = parser.parse_args()

    if not args.model.exists():
        sys.exit(f"[error] model checkpoint not found: {args.model}")
    if not args.ground_truth.exists():
        sys.exit(f"[error] ground truth not found: {args.ground_truth}")
    if not args.path_db_dir.exists():
        sys.exit(f"[error] path-database directory not found: {args.path_db_dir}")
    if not args.ast_dir.exists():
        sys.exit(f"[error] semantic-graph directory not found: {args.ast_dir}")

    print("=" * 78)
    print("SmartTaintRL -- Localization Evaluation")
    print("=" * 78)
    print(f"  model        : {args.model}")
    print(f"  ground truth : {args.ground_truth}")
    print(f"  path DBs     : {args.path_db_dir}")
    print(f"  semantic g.  : {args.ast_dir}")

    gt_data = load_ground_truth(args.ground_truth)
    print(f"  contracts    : {len(gt_data)}")

    configs = ABLATION_CONFIGS if args.ablation else ABLATION_CONFIGS[:1]

    print("\n" + "-" * 78)
    print("Results")
    print("-" * 78)

    all_summaries = []
    for name, weights in configs:
        localizer = VulnerabilityLocalizer(str(args.model), **weights)
        report = evaluate_configuration(localizer, gt_data,
                                        args.path_db_dir, args.ast_dir,
                                        verbose=False)
        print_summary(name, report['summary'])
        all_summaries.append((name, report))

    # Show the detailed per-contract table for the full configuration
    if not args.quiet:
        full = all_summaries[0][1]
        print_detailed(full['per_contract'])


if __name__ == "__main__":
    main()
