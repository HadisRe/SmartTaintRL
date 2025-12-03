import json
import os
from datetime import datetime


def create_contract_profile_fixed(contract_address, label, base_dir):
    """
    ساخت پروفایل کامل برای یک contract با debug کامل
    """
    profile = {
        'address': contract_address,
        'label': label,
        'ast_features': {},
        'graph_statistics': {},
        'taint_summary': {},
        'consolidated_features': {},
        'debug_info': {
            'ast_status': 'pending',
            'graph_status': 'pending',
            'taint_status': 'pending',
            'errors': []
        }
    }

    # مسیرهای فایل
    ast_path = os.path.join(base_dir, 'contract_ast1_clean', f'{contract_address}_ast.json')
    graph_path = os.path.join(base_dir, 'contract_ast1_clean', f'{contract_address}_semantic_graph.json')
    taint_path = os.path.join(base_dir, 'contract_ast1_clean', f'{contract_address}_taint_analysis_filtered.json')

    # 1. AST Processing
    try:
        with open(ast_path, 'r', encoding='utf-8') as f:
            ast_data = json.load(f)

        features = {'functions_count': 0, 'state_vars_count': 0,
                    'functions_with_require': 0, 'functions_with_modifier': 0}

        if 'contracts' in ast_data:
            for contract_name, contract_data in ast_data['contracts'].items():
                if 'functions' in contract_data:
                    features['functions_count'] += len(contract_data['functions'])
                    for func_name, func_data in contract_data['functions'].items():
                        if func_data.get('modifiers') and len(func_data.get('modifiers', [])) > 0:
                            features['functions_with_modifier'] += 1
                        if func_data.get('body'):
                            func_str = json.dumps(func_data['body']).lower()
                            if 'require' in func_str:
                                features['functions_with_require'] += 1

                if 'state_variables' in contract_data:
                    features['state_vars_count'] += len(contract_data['state_variables'])

        profile['ast_features'] = features
        profile['debug_info']['ast_status'] = 'success'

    except Exception as e:
        profile['debug_info']['ast_status'] = 'failed'
        profile['debug_info']['errors'].append(f"AST: {str(e)[:100]}")

    # 2. Graph Processing - با فیلدهای صحیح
    try:
        with open(graph_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)

        if 'statistics' in graph_data:
            stats = graph_data['statistics']
            # استفاده از فیلدهای صحیح
            profile['graph_statistics'] = {
                'source_nodes': stats.get('source_nodes', 0),
                'sink_nodes': stats.get('sink_nodes', 0),
                'total_nodes': stats.get('total_nodes', 0),
                'total_edges': stats.get('total_edges', 0),
                'source_types': stats.get('source_types', {}),
                'sink_types': stats.get('sink_types', {})
            }
            profile['debug_info']['graph_status'] = 'success'
        else:
            profile['debug_info']['graph_status'] = 'no_statistics'

    except Exception as e:
        profile['debug_info']['graph_status'] = 'failed'
        profile['debug_info']['errors'].append(f"Graph: {str(e)[:100]}")

    # 3. Taint Processing
    try:
        with open(taint_path, 'r', encoding='utf-8') as f:
            taint_data = json.load(f)

        if 'summary' in taint_data:
            profile['taint_summary'] = taint_data['summary']

        if 'paths' in taint_data:
            paths = taint_data['paths']
            mit_counts = [len(p.get('mitigating_factors', [])) for p in paths]
            profile['taint_summary']['total_paths'] = len(paths)
            profile['taint_summary']['avg_mitigations'] = sum(mit_counts) / len(mit_counts) if mit_counts else 0
            profile['taint_summary']['paths_with_mitigation'] = sum(1 for c in mit_counts if c > 0)

        profile['debug_info']['taint_status'] = 'success'

    except Exception as e:
        profile['debug_info']['taint_status'] = 'failed'
        profile['debug_info']['errors'].append(f"Taint: {str(e)[:100]}")

    # 4. Consolidated Features
    try:
        cf = {}
        ast = profile['ast_features']
        graph = profile['graph_statistics']
        taint = profile['taint_summary']

        cf['complexity_score'] = ast.get('functions_count', 0) / 100.0
        cf['require_density'] = ast.get('functions_with_require', 0) / max(ast.get('functions_count', 1), 1)
        cf['modifier_usage'] = ast.get('functions_with_modifier', 0) / max(ast.get('functions_count', 1), 1)
        cf['source_sink_ratio'] = graph.get('source_nodes', 0) / max(graph.get('sink_nodes', 1), 1)
        cf['protection_coverage'] = taint.get('paths_with_mitigation', 0) / max(taint.get('total_paths', 1), 1)

        profile['consolidated_features'] = cf

    except Exception as e:
        profile['debug_info']['errors'].append(f"Consolidation: {str(e)[:100]}")

    return profile


def batch_process_contracts():
    """
    پردازش همه 40 contract با گزارش کامل
    """
    base_dir = r'C:\Users\Hadis\Documents\NewModel1'
    safe_dir = os.path.join(base_dir, 'Safe_contract_clean')
    vuln_dir = os.path.join(base_dir, 'Vulnerable_contract_clean')
    output_dir = os.path.join(base_dir, 'contract_profiles1')
    os.makedirs(output_dir, exist_ok=True)

    # safe_files = [f.replace('.sol', '') for f in os.listdir(safe_dir)[:200]]
    # vuln_files = [f.replace('.sol', '') for f in os.listdir(vuln_dir)[:200]]

    safe_files = [f.replace('.sol', '') for f in os.listdir(safe_dir)[:4085]]
    vuln_files = [f.replace('.sol', '') for f in os.listdir(vuln_dir)[:223]]

    print("=" * 70)
    print("🚀 CONTRACT PROFILE GENERATION - PHASE 1")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(
        f"📊 Processing: {len(safe_files)} Safe + {len(vuln_files)} Vulnerable = {len(safe_files) + len(vuln_files)} Total")
    print("-" * 70)

    results = {
        'successful': [],
        'partial': [],
        'failed': [],
        'statistics': {
            'safe': {'total': 0, 'success': 0, 'partial': 0, 'failed': 0},
            'vuln': {'total': 0, 'success': 0, 'partial': 0, 'failed': 0}
        }
    }

    # پردازش contracts
    all_contracts = [(addr, 'safe') for addr in safe_files] + [(addr, 'vulnerable') for addr in vuln_files]

    for i, (address, label) in enumerate(all_contracts, 1):
        print(f"\n[{i:02d}/40] Processing {address[:10]}... ({label})")

        profile = create_contract_profile_fixed(address, label, base_dir)

        # بررسی وضعیت
        debug = profile['debug_info']
        status = 'success'

        if debug['ast_status'] == 'success':
            print(f"    ✓ AST: {profile['ast_features']['functions_count']} functions")
        else:
            print(f"    ❌ AST: {debug['ast_status']}")
            status = 'partial' if status == 'success' else status

        if debug['graph_status'] == 'success':
            print(
                f"    ✓ Graph: {profile['graph_statistics']['source_nodes']} sources, {profile['graph_statistics']['sink_nodes']} sinks")
        else:
            print(f"    ❌ Graph: {debug['graph_status']}")
            status = 'partial' if status == 'success' else status

        if debug['taint_status'] == 'success':
            print(f"    ✓ Taint: {profile['taint_summary'].get('total_paths', 0)} paths")
        else:
            print(f"    ❌ Taint: {debug['taint_status']}")
            status = 'partial' if status == 'success' else status

        # ذخیره profile
        output_file = os.path.join(output_dir, f'{address}_profile.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(profile, f, indent=2)

        # ثبت نتیجه
        if all([debug['ast_status'] == 'success',
                debug['graph_status'] == 'success',
                debug['taint_status'] == 'success']):
            results['successful'].append(address)
            print(f"    ✅ Status: COMPLETE")
        elif any([debug['ast_status'] == 'failed',
                  debug['graph_status'] == 'failed',
                  debug['taint_status'] == 'failed']):
            results['partial'].append(address)
            print(f"    ⚠️ Status: PARTIAL")
        else:
            results['failed'].append(address)
            print(f"    ❌ Status: FAILED")

        # آمار
        label_key = 'safe' if label == 'safe' else 'vuln'
        results['statistics'][label_key]['total'] += 1
        if status == 'success':
            results['statistics'][label_key]['success'] += 1
        elif status == 'partial':
            results['statistics'][label_key]['partial'] += 1
        else:
            results['statistics'][label_key]['failed'] += 1

    # گزارش نهایی
    print("\n" + "=" * 70)
    print("📊 FINAL REPORT")
    print("=" * 70)
    print(f"✅ Complete Success: {len(results['successful'])}/40")
    print(f"⚠️ Partial Success: {len(results['partial'])}/40")
    print(f"❌ Failed: {len(results['failed'])}/40")
    print("\nBreakdown:")
    print(
        f"  Safe contracts: {results['statistics']['safe']['success']}/{results['statistics']['safe']['total']} success")
    print(
        f"  Vuln contracts: {results['statistics']['vuln']['success']}/{results['statistics']['vuln']['total']} success")

    # ذخیره گزارش
    report_file = os.path.join(output_dir, 'generation_report.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Report saved to: generation_report.json")
    print(f"💾 Profiles saved in: {output_dir}")

    return results


if __name__ == "__main__":
    results = batch_process_contracts()
