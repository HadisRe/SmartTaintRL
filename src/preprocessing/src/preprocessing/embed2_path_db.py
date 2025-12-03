import json
import os
from datetime import datetime
from collections import defaultdict, Counter


def create_nodes_dict(nodes_list):
    """
    تبدیل لیست nodes به dictionary برای دسترسی سریع
    """
    return {node['id']: node for node in nodes_list}


def extract_path_basic_info(path_data, debug=False):
    """
    استخراج اطلاعات پایه از یک path در taint analysis
    """
    info = {
        'path_nodes': path_data.get('path', []),
        'path_length': path_data.get('path_length', len(path_data.get('path', []))),
        'risk_level': path_data.get('risk_level', 'UNKNOWN'),  # Hidden - won't go to state
        'source_node': path_data.get('source', 'unknown'),
        'sink_node': path_data.get('sink', 'unknown'),
        'source_type': path_data.get('source_type', 'unknown'),
        'sink_type': path_data.get('sink_type', 'unknown'),
        'mitigation_count': len(path_data.get('mitigating_factors', [])),
        'risk_factors_count': len(path_data.get('risk_factors', [])),
        'has_data_flow': path_data.get('has_data_flow', False),
        'contains_loop': path_data.get('contains_loop', False),
        # ذخیره محتوا برای debugging ولی به agent نمیدهیم
        '_mitigation_content': path_data.get('mitigating_factors', []),
        '_risk_content': path_data.get('risk_factors', [])
    }

    if debug:
        print(f"    📍 Path: {info['source_node']} → {info['sink_node']}")
        print(f"       Length: {info['path_length']}, Risk: {info['risk_level']}, "
              f"Mitigations: {info['mitigation_count']}")
        print(f"       Source Type: {info['source_type']}, Sink Type: {info['sink_type']}")

    return info


def enrich_from_semantic_graph(path_info, graph_data, debug=False):
    """
    غنی‌سازی اطلاعات path از semantic graph
    """
    # تبدیل لیست nodes به dictionary
    nodes_list = graph_data.get('nodes', [])
    nodes_dict = create_nodes_dict(nodes_list)

    enriched = {
        'nodes_detail': [],
        'node_types_count': defaultdict(int),
        'functions_involved': set(),
        'require_nodes_in_path': 0,
        'condition_nodes_in_path': 0,
        'keccak_nodes_in_path': 0,
        'assignment_nodes_in_path': 0
    }

    # برای هر node در path
    for node_id in path_info['path_nodes']:
        if node_id in nodes_dict:
            node_data = nodes_dict[node_id]

            # اطلاعات کامل node
            node_detail = {
                'node_id': node_id,
                'type': node_data.get('type', 'unknown'),
                'function': node_data.get('function', 'unknown'),
                'is_source': node_data.get('is_source', False),
                'is_sink': node_data.get('is_sink', False),
                'code_snippet': node_data.get('code_snippet', '')[:100],  # فقط 100 کاراکتر اول
                'label': node_data.get('label', ''),
                'used_vars': node_data.get('used_vars', []),
                'defined_vars': node_data.get('defined_vars', [])
            }
            enriched['nodes_detail'].append(node_detail)

            # شمارش types
            node_type = node_data.get('type', 'unknown')
            enriched['node_types_count'][node_type] += 1

            # شمارش specific types
            if 'require' in node_type.lower() or 'require' in node_data.get('label', '').lower():
                enriched['require_nodes_in_path'] += 1
            elif node_type == 'condition' or 'if' in node_type.lower():
                enriched['condition_nodes_in_path'] += 1
            elif 'keccak' in node_type.lower() or 'hash' in node_type.lower():
                enriched['keccak_nodes_in_path'] += 1
            elif node_type == 'assignment' or '=' in node_data.get('label', ''):
                enriched['assignment_nodes_in_path'] += 1

            # functions involved
            func_name = node_data.get('function', 'unknown')
            if func_name and func_name != 'unknown' and func_name != '':
                enriched['functions_involved'].add(func_name)
        else:
            # اگر node در graph پیدا نشد
            enriched['nodes_detail'].append({
                'node_id': node_id,
                'type': 'not_found',
                'function': 'unknown'
            })

    # تبدیل set به list برای JSON
    enriched['functions_involved'] = list(enriched['functions_involved'])
    enriched['unique_functions_count'] = len(enriched['functions_involved'])

    if debug:
        print(f"      🔍 Enriched: {len(enriched['nodes_detail'])} nodes detailed")
        print(f"         Functions: {enriched['unique_functions_count']} unique")
        print(f"         Node types: Require={enriched['require_nodes_in_path']}, "
              f"Condition={enriched['condition_nodes_in_path']}, "
              f"Keccak={enriched['keccak_nodes_in_path']}")

    return enriched


def extract_function_context(path_info, enriched_info, ast_data, graph_data, debug=False):
    """
    استخراج context از function که path در آن قرار دارد
    """
    context = {
        'primary_function': 'unknown',
        'function_has_modifier': False,
        'function_visibility': 'unknown',
        'function_require_count': 0,  # کل requires در function
        'protection_outside_path': False,  # آیا خارج از path محافظت هست
        'modifier_names': [],
        'function_state_changes': False
    }

    # پیدا کردن primary function (function که بیشتر nodes در آن هستند)
    if enriched_info['functions_involved']:
        # شمارش تعداد nodes در هر function
        func_counter = Counter()
        for node in enriched_info['nodes_detail']:
            func = node.get('function', 'unknown')
            if func and func != 'unknown':
                func_counter[func] += 1

        if func_counter:
            context['primary_function'] = func_counter.most_common(1)[0][0]

    # اطلاعات function از AST
    if 'contracts' in ast_data and context['primary_function'] != 'unknown':
        for contract_name, contract_data in ast_data['contracts'].items():
            if 'functions' in contract_data:
                for func_name, func_data in contract_data['functions'].items():
                    if func_name == context['primary_function']:
                        # Modifiers
                        modifiers = func_data.get('modifiers', [])
                        context['function_has_modifier'] = len(modifiers) > 0
                        context['modifier_names'] = modifiers

                        # Visibility
                        context['function_visibility'] = func_data.get('visibility', 'unknown')

                        # State mutability
                        state_mut = func_data.get('state_mutability', '')
                        context['function_state_changes'] = state_mut not in ['view', 'pure']

                        # کل requires در function body
                        body_str = json.dumps(func_data.get('body', {})).lower()
                        context['function_require_count'] = body_str.count('require')

                        # آیا requires بیشتر از آنچه در path هست وجود دارد؟
                        if context['function_require_count'] > enriched_info['require_nodes_in_path']:
                            context['protection_outside_path'] = True

                        break

    # بررسی nodes در graph برای protections خارج از path
    if not context['protection_outside_path'] and context['primary_function'] != 'unknown':
        nodes_list = graph_data.get('nodes', [])
        nodes_dict = create_nodes_dict(nodes_list)

        # پیدا کردن همه nodes در همان function
        function_nodes = []
        for node in nodes_list:
            if node.get('function') == context['primary_function']:
                function_nodes.append(node['id'])

        # nodes که در function هستند ولی در path نیستند
        path_nodes_set = set(path_info['path_nodes'])
        outside_nodes = set(function_nodes) - path_nodes_set

        # آیا در nodes خارج از path، require یا protection وجود دارد؟
        for node_id in outside_nodes:
            if node_id in nodes_dict:
                node = nodes_dict[node_id]
                node_type = node.get('type', '')
                node_label = node.get('label', '').lower()

                if 'require' in node_type.lower() or 'require' in node_label:
                    # بررسی که آیا قبل از source است (با استفاده از line number)
                    if path_info['path_nodes']:
                        source_node_id = path_info['path_nodes'][0]
                        if source_node_id in nodes_dict:
                            source_line = nodes_dict[source_node_id].get('line_start', 999999)
                            require_line = node.get('line_start', 0)
                            if require_line < source_line:
                                context['protection_outside_path'] = True
                                break

    if debug:
        print(f"      📋 Context: Function={context['primary_function']}")
        print(f"         HasModifier={context['function_has_modifier']}, "
              f"Visibility={context['function_visibility']}")
        print(f"         ProtectionOutside={context['protection_outside_path']}")

    return context


def calculate_aggregate_features(path_info, enriched_info, context_info, debug=False):
    """
    محاسبه features نهایی aggregate شده برای embedding
    """
    path_length = path_info['path_length']

    features = {
        # Path structure features
        'path_length_normalized': min(path_length / 20.0, 1.0),  # Normalize و cap در 1
        'require_density': enriched_info['require_nodes_in_path'] / max(path_length, 1),
        'condition_density': enriched_info['condition_nodes_in_path'] / max(path_length, 1),
        'keccak_density': enriched_info['keccak_nodes_in_path'] / max(path_length, 1),

        # Protection features
        'mitigation_score': min(path_info['mitigation_count'] / 3.0, 1.0),  # Cap at 3
        'has_any_mitigation': int(path_info['mitigation_count'] > 0),
        'has_strong_mitigation': int(path_info['mitigation_count'] > 1),

        # Context features
        'has_modifier_protection': int(context_info['function_has_modifier']),
        'has_restricted_visibility': int(context_info['function_visibility'] in ['private', 'internal']),
        'has_external_protection': int(context_info['protection_outside_path']),
        'function_require_density': context_info['function_require_count'] / max(10, 1),  # Normalize to 10

        # Source/Sink features (encoded as categories)
        'source_type_encoded': encode_source_type(path_info['source_type']),
        'sink_type_encoded': encode_sink_type(path_info['sink_type']),

        # Complexity features
        'unique_functions_ratio': min(enriched_info['unique_functions_count'] / max(3, 1), 1.0),
        'node_diversity': len(enriched_info['node_types_count']) / max(path_length, 1),

        # Distance metrics
        'distance_to_sink': calculate_distance_to_sink(enriched_info),
        'distance_from_source': calculate_distance_from_source(enriched_info),

        # Data flow features
        'has_data_flow': int(path_info.get('has_data_flow', False)),
        'contains_loop': int(path_info.get('contains_loop', False))
    }

    if debug:
        print(f"      📊 Aggregated Features:")
        print(f"         RequireDensity={features['require_density']:.2f}, "
              f"MitigationScore={features['mitigation_score']:.2f}")
        print(f"         ExternalProtection={features['has_external_protection']}, "
              f"ModifierProtection={features['has_modifier_protection']}")

    return features


def encode_source_type(source_type):
    """One-hot encoding برای source types"""
    types = {
        'timestamp': [1, 0, 0, 0],
        'blockhash': [0, 1, 0, 0],
        'blocknumber': [0, 0, 1, 0],
        'block.number': [0, 0, 1, 0],
        'other': [0, 0, 0, 1]
    }
    return types.get(source_type.lower(), types['other'])


def encode_sink_type(sink_type):
    """One-hot encoding برای sink types"""
    types = {
        'transfer': [1, 0, 0, 0],
        'randomgeneration': [0, 1, 0, 0],
        'statemodification': [0, 0, 1, 0],
        'accesscontrol': [0, 0, 0, 1],  # نوع جدید که در داده دیدیم
        'other': [0, 0, 0, 1]
    }
    sink_lower = sink_type.lower()
    for key in types:
        if key in sink_lower or sink_lower in key:
            return types[key]
    return types['other']


def calculate_distance_to_sink(enriched_info):
    """محاسبه فاصله نزدیکترین require/condition تا sink"""
    if not enriched_info['nodes_detail']:
        return 1.0

    sink_index = len(enriched_info['nodes_detail']) - 1
    min_distance = sink_index

    for i, node in enumerate(enriched_info['nodes_detail']):
        node_type = node.get('type', '').lower()
        if 'require' in node_type or node_type == 'condition':
            distance = sink_index - i
            min_distance = min(min_distance, distance)

    # Normalize
    return min_distance / max(len(enriched_info['nodes_detail']), 1)


def calculate_distance_from_source(enriched_info):
    """محاسبه فاصله اولین require/condition از source"""
    if not enriched_info['nodes_detail']:
        return 1.0

    for i, node in enumerate(enriched_info['nodes_detail']):
        node_type = node.get('type', '').lower()
        if 'require' in node_type or node_type == 'condition':
            # Normalize
            return i / max(len(enriched_info['nodes_detail']), 1)

    return 1.0  # اگر هیچ protection نبود


def process_single_contract_paths(contract_address, base_dir):
    """
    پردازش کامل paths یک contract و ساخت path database
    """
    print(f"\n{'=' * 70}")
    print(f"🔄 Processing Contract: {contract_address[:10]}...")
    print(f"{'=' * 70}")

    # مسیرهای فایل
    ast_path = os.path.join(base_dir, 'contract_ast1_clean', f'{contract_address}_ast.json')
    graph_path = os.path.join(base_dir, 'contract_ast1_clean', f'{contract_address}_semantic_graph.json')
    taint_path = os.path.join(base_dir, 'contract_ast1_clean', f'{contract_address}_taint_analysis_filtered.json')
    profile_path = os.path.join(base_dir, 'contract_profiles1', f'{contract_address}_profile.json')


    result = {
        'contract_address': contract_address,
        'timestamp': datetime.now().isoformat(),
        'paths': [],
        'statistics': {
            'total_paths': 0,
            'paths_with_mitigation': 0,
            'paths_with_external_protection': 0,
            'paths_with_modifier': 0,
            'avg_path_length': 0,
            'risk_distribution': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'UNKNOWN': 0}
        },
        'debug_info': {
            'status': 'pending',
            'errors': []
        }
    }

    try:
        # خواندن داده‌ها
        print(f"📖 Loading data files...")
        with open(ast_path, 'r', encoding='utf-8') as f:
            ast_data = json.load(f)
        with open(graph_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        with open(taint_path, 'r', encoding='utf-8') as f:
            taint_data = json.load(f)
        with open(profile_path, 'r', encoding='utf-8') as f:
            profile_data = json.load(f)

        print(f"✅ Files loaded successfully")

        # پردازش هر path
        paths = taint_data.get('paths', [])
        print(f"\n📊 Found {len(paths)} paths to process")

        # پردازش حداکثر 20 path اول برای تست (بعداً می‌توان همه را پردازش کرد)
        max_paths = len(paths)  # همه paths

        for idx, path_data in enumerate(paths[:max_paths]):
            print(f"\n  Path {idx + 1}/{max_paths}:")

            # 1. اطلاعات پایه
            path_info = extract_path_basic_info(path_data, debug=True)

            # 2. Enrichment از graph
            enriched_info = enrich_from_semantic_graph(path_info, graph_data, debug=True)

            # 3. Context از function
            context_info = extract_function_context(path_info, enriched_info, ast_data, graph_data, debug=True)

            # 4. Aggregate features
            aggregate_features = calculate_aggregate_features(path_info, enriched_info, context_info, debug=True)

            # ترکیب همه اطلاعات
            path_entry = {
                'path_index': idx,
                'basic_info': path_info,
                'graph_enrichment': {
                    'nodes_detail': enriched_info['nodes_detail'],
                    'node_types_count': dict(enriched_info['node_types_count']),
                    'functions_involved': enriched_info['functions_involved'],
                    'unique_functions_count': enriched_info['unique_functions_count'],
                    'require_nodes': enriched_info['require_nodes_in_path'],
                    'condition_nodes': enriched_info['condition_nodes_in_path']
                },
                'function_context': context_info,
                'aggregate_features': aggregate_features
            }

            result['paths'].append(path_entry)

            # آمارگیری
            result['statistics']['total_paths'] += 1
            if path_info['mitigation_count'] > 0:
                result['statistics']['paths_with_mitigation'] += 1
            if context_info['protection_outside_path']:
                result['statistics']['paths_with_external_protection'] += 1
            if context_info['function_has_modifier']:
                result['statistics']['paths_with_modifier'] += 1

            risk = path_info['risk_level']
            if risk in result['statistics']['risk_distribution']:
                result['statistics']['risk_distribution'][risk] += 1
            else:
                result['statistics']['risk_distribution']['UNKNOWN'] += 1

        # محاسبه آمار نهایی
        if result['paths']:
            avg_length = sum(p['basic_info']['path_length'] for p in result['paths']) / len(result['paths'])
            result['statistics']['avg_path_length'] = avg_length

        result['debug_info']['status'] = 'success'

        print(f"\n{'=' * 50}")
        print(f"✅ SUCCESS: Processed {len(result['paths'])} paths")
        print(f"📈 Statistics:")
        print(f"   - Paths with mitigation: {result['statistics']['paths_with_mitigation']}")
        print(f"   - Paths with external protection: {result['statistics']['paths_with_external_protection']}")
        print(f"   - Paths with modifier: {result['statistics']['paths_with_modifier']}")
        print(f"   - Average path length: {result['statistics']['avg_path_length']:.2f}")
        print(f"   - Risk distribution: {result['statistics']['risk_distribution']}")

    except Exception as e:
        result['debug_info']['status'] = 'failed'
        result['debug_info']['errors'].append(str(e))
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

    return result


def batch_process_all_contracts():
    """
    پردازش همه contracts و ساخت path databases
    """
    base_dir = r'C:\Users\Hadis\Documents\NewModel1'
    safe_dir = os.path.join(base_dir, 'Safe_contract_clean')
    vuln_dir = os.path.join(base_dir, 'Vulnerable_contract_clean')
    output_dir = os.path.join(base_dir, 'path_databases1')
    os.makedirs(output_dir, exist_ok=True)

    # لیست contracts
    # safe_contracts = [f.replace('.sol', '') for f in os.listdir(safe_dir) if f.endswith('.sol')][:200]
    # vuln_contracts = [f.replace('.sol', '') for f in os.listdir(vuln_dir) if f.endswith('.sol')][:200]

    safe_contracts = [f.replace('.sol', '') for f in os.listdir(safe_dir) if f.endswith('.sol')][:4085]
    vuln_contracts = [f.replace('.sol', '') for f in os.listdir(vuln_dir) if f.endswith('.sol')][:223]

    print("=" * 80)
    print("🚀 PATH DATABASE CONSTRUCTION - PHASE 2 (FIXED)")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"📊 Processing: {len(safe_contracts)} Safe + {len(vuln_contracts)} Vulnerable")
    print("-" * 80)

    global_stats = {
        'successful': 0,
        'failed': 0,
        'total_paths_processed': 0,
        'contracts_with_external_protection': 0,
        'contracts_with_modifier_protection': 0
    }

    all_contracts = [(addr, 'safe') for addr in safe_contracts] + [(addr, 'vulnerable') for addr in vuln_contracts]

    for i, (contract_address, label) in enumerate(all_contracts, 1):
        print(f"\n[{i:02d}/40] Label: {label}")
        result = process_single_contract_paths(contract_address, base_dir)

        # ذخیره نتیجه
        output_file = os.path.join(output_dir, f'{contract_address}_path_database.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)

        # آمار کلی
        if result['debug_info']['status'] == 'success':
            global_stats['successful'] += 1
            global_stats['total_paths_processed'] += result['statistics']['total_paths']
            if result['statistics']['paths_with_external_protection'] > 0:
                global_stats['contracts_with_external_protection'] += 1
            if result['statistics']['paths_with_modifier'] > 0:
                global_stats['contracts_with_modifier_protection'] += 1
        else:
            global_stats['failed'] += 1

    # گزارش نهایی
    print("\n" + "=" * 80)
    print("📊 FINAL REPORT - PATH DATABASE CONSTRUCTION")
    print("=" * 80)
    print(f"✅ Successful: {global_stats['successful']}/40")
    print(f"❌ Failed: {global_stats['failed']}/40")
    print(f"📈 Total paths processed: {global_stats['total_paths_processed']}")
    print(f"🛡️ Contracts with external protection: {global_stats['contracts_with_external_protection']}")
    print(f"🔐 Contracts with modifier protection: {global_stats['contracts_with_modifier_protection']}")
    print(f"\n💾 Path databases saved in: {output_dir}")

    # ذخیره آمار کلی
    stats_file = os.path.join(output_dir, 'phase2_statistics.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(global_stats, f, indent=2)

    return global_stats


if __name__ == "__main__":
    # برای تست روی یک contract:
    # test_address = "0x00f90986cdd79744409f8a3c7747064afa4473b5"
    # result = process_single_contract_paths(test_address, r'C:\Users\Hadis\Documents\NewModel1')

    # برای پردازش همه contracts:
    batch_process_all_contracts()
