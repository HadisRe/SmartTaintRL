"""
SmartTaintRL Localization Evaluation
Evaluates function-level and node-level localization on 14 ground truth contracts
"""

import pickle
from pathlib import Path
from localizer import VulnerabilityLocalizer, compute_metrics


# Configuration
BASE_PATH = Path(r"C:\Users\Hadis\Documents\NewModel1")
MODEL_PATH = BASE_PATH / "trained_dqn_model_2500.pth"
GROUND_TRUTH_PATH = BASE_PATH / "final_dataset_14_contracts.pkl"
PATH_DB_DIR = BASE_PATH / "path_databases_updated"
AST_DIR = BASE_PATH / "contract_ast1_clean"


def load_ground_truth(path: Path) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


def evaluate():
    print("=" * 70)
    print("SmartTaintRL - Localization Evaluation")
    print("=" * 70)
    
    # Load model
    print("\nLoading model...")
    localizer = VulnerabilityLocalizer(str(MODEL_PATH))
    print(f"Device: {localizer.device}")
    
    # Load ground truth
    print("\nLoading ground truth...")
    gt_data = load_ground_truth(GROUND_TRUTH_PATH)
    print(f"Loaded {len(gt_data)} contracts")
    
    # Results storage
    results = []
    exact_matches = 0
    caller_matches = 0
    wrong = 0
    
    node_precisions = []
    node_recalls = []
    
    # Evaluate each contract
    print("\n" + "-" * 70)
    print("Running Localization...")
    print("-" * 70)
    
    for idx, (address, info) in enumerate(gt_data.items(), 1):
        contract_id = f"C{idx}"
        gt_function = info['function']
        gt_nodes = info.get('nodes', [])
        
        print(f"\n{contract_id}: {address[:15]}...")
        
        # Load path database
        path_db_file = PATH_DB_DIR / f"{address}_path_database.json"
        path_db = localizer.load_path_database(str(path_db_file))
        
        if not path_db:
            print(f"   ✗ path_database not found")
            results.append({
                'id': contract_id, 'gt': gt_function, 'pred': None, 
                'type': 'Missing', 'p5': 0, 'r5': 0
            })
            wrong += 1
            continue
        
        # Load semantic graph for caller detection
        sg_file = AST_DIR / f"{address}_semantic_graph.json"
        semantic_graph = localizer.load_semantic_graph(str(sg_file))
        
        # Run localization
        loc_result = localizer.localize(path_db, semantic_graph, top_k_functions=5, top_k_nodes=5)
        
        ranked_funcs = loc_result['functions']
        call_graph = loc_result.get('call_graph', {})
        
        if call_graph:
            print(f"   Call graph: {dict(list(call_graph.items())[:3])}...")
        
        if not ranked_funcs:
            print(f"   ✗ No functions found")
            results.append({
                'id': contract_id, 'gt': gt_function, 'pred': None,
                'type': 'Wrong', 'p5': 0, 'r5': 0
            })
            wrong += 1
            continue
        
        func_names = [f[0] for f in ranked_funcs]
        predicted = func_names[0]
        pred_nodes = loc_result['nodes'].get(predicted, [])
        
        print(f"   Functions: {func_names}")
        print(f"   GT: {gt_function} | Predicted: {predicted}")
        
        # Determine match type
        match_type = 'Wrong'
        
        if predicted == gt_function:
            match_type = 'Exact'
            exact_matches += 1
        else:
            # Check caller relationship for all ranked functions
            for func in func_names:
                if localizer.check_caller_relationship(call_graph, func, gt_function):
                    match_type = 'Caller'
                    caller_matches += 1
                    print(f"   ✓ Caller detected: {func} calls {gt_function}")
                    break
            
            if match_type == 'Wrong':
                wrong += 1
        
        # Compute node metrics (only for exact matches)
        if match_type == 'Exact' and gt_nodes:
            metrics = compute_metrics(gt_nodes, pred_nodes, k=5)
            node_precisions.append(metrics['precision'])
            node_recalls.append(metrics['recall'])
            p5, r5 = metrics['precision'], metrics['recall']
        else:
            p5, r5 = 0.0, 0.0
        
        results.append({
            'id': contract_id, 'gt': gt_function, 'pred': predicted,
            'type': match_type, 'p5': p5, 'r5': r5
        })
    
    # Summary
    total = len(gt_data)
    
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\n📈 Function-Level:")
    print(f"   Exact Matches: {exact_matches}/{total} ({100*exact_matches/total:.1f}%)")
    print(f"   Caller Detection: {caller_matches}/{total}")
    print(f"   Incorrect: {wrong}/{total}")
    print(f"   Strict Accuracy: {100*exact_matches/total:.1f}%")
    print(f"   Relaxed Accuracy: {100*(exact_matches+caller_matches)/total:.1f}%")
    
    if node_precisions:
        avg_p = sum(node_precisions) / len(node_precisions)
        avg_r = sum(node_recalls) / len(node_recalls)
        avg_f1 = 2 * avg_p * avg_r / (avg_p + avg_r) if (avg_p + avg_r) > 0 else 0
        
        print(f"\n📈 Node-Level (on {len(node_precisions)} exact matches):")
        print(f"   Precision@5: {avg_p:.2f}")
        print(f"   Recall@5: {avg_r:.2f}")
        print(f"   F1@5: {avg_f1:.2f}")
    
    # Detailed results
    print("\n" + "-" * 70)
    print("Detailed Results:")
    print("-" * 70)
    print(f"{'ID':<5}{'GT Function':<23}{'Predicted':<23}{'Type':<8}{'P@5':<6}{'R@5':<6}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['id']:<5}{r['gt']:<23}{str(r['pred']):<23}{r['type']:<8}{r['p5']:.2f}  {r['r5']:.2f}")
    
    print("\n" + "=" * 70)
    print("Expected (Table 9):")
    print("   Strict: 64.3%, Relaxed: 92.9%")
    print("   P@5: 0.65, R@5: 0.77, F1@5: 0.70")
    print("=" * 70)


if __name__ == "__main__":
    evaluate()
