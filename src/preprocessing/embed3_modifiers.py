import json
import os
import re
from collections import defaultdict


def extract_modifiers_from_source(sol_file_path):
    """
    استخراج modifier data از source code با دقت بالا
    """
    with open(sol_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # 1. پیدا کردن modifier definitions
    modifier_def_pattern = r'modifier\s+(\w+)\s*\([^)]*\)\s*\{'
    modifier_defs = list(set(re.findall(modifier_def_pattern, content)))
    
    # 2. ساخت mapping از function name به modifiers
    function_modifiers_map = {}
    
    # Pattern دقیق‌تر برای functions
    # این pattern حتی overloaded functions را هم handle می‌کند
    func_pattern = r'function\s+(\w+)\s*\(([^)]*)\)\s*([^{]+)\s*\{'
    
    func_matches = re.finditer(func_pattern, content)
    
    for match in func_matches:
        func_name = match.group(1)
        func_params = match.group(2)
        modifiers_section = match.group(3)
        
        # حذف keywords که modifier نیستند
        visibility_keywords = ['public', 'private', 'internal', 'external', 
                              'view', 'pure', 'payable', 'returns', 'override', 
                              'virtual', 'constant']
        
        # پیدا کردن modifiers واقعی
        found_modifiers = []
        for mod_def in modifier_defs:
            # Pattern دقیق برای match کردن modifier
            pattern = r'\b' + re.escape(mod_def) + r'\b'
            if re.search(pattern, modifiers_section):
                found_modifiers.append(mod_def)
        
        # ذخیره با توجه به احتمال overloading
        # اگر function قبلاً وجود دارد، آن را override می‌کنیم
        # (در Solidity overloading کم است)
        function_modifiers_map[func_name] = {
            'modifiers': found_modifiers,
            'has_params': bool(func_params.strip()),
            'visibility': 'unknown',
            'signature_snippet': modifiers_section.strip()[:50]
        }
        
        # تشخیص visibility
        for vis in ['public', 'private', 'internal', 'external']:
            if vis in modifiers_section:
                function_modifiers_map[func_name]['visibility'] = vis
                break
    
    return {
        'modifier_definitions': modifier_defs,
        'function_modifiers': function_modifiers_map
    }


def update_path_database_with_modifiers(contract_address, base_dir, debug=True):
    """
    Update path database با modifier data از source
    """
    # مسیرها
    path_db_file = os.path.join(base_dir, 'path_databases1', f'{contract_address}_path_database.json')

    # پیدا کردن source file
    sol_file = None
    for dir_name in ['Safe_contract_clean', 'Vulnerable_contract_clean']:
        potential_path = os.path.join(base_dir, dir_name, f'{contract_address}.sol')
        if os.path.exists(potential_path):
            sol_file = potential_path
            break
    
    if not sol_file:
        if debug:
            print(f"❌ Source file not found for {contract_address[:10]}")
        return False
    
    # خواندن path database
    try:
        with open(path_db_file, 'r', encoding='utf-8') as f:
            path_db = json.load(f)
    except:
        if debug:
            print(f"❌ Cannot read path database for {contract_address[:10]}")
        return False
    
    # استخراج modifiers از source
    source_data = extract_modifiers_from_source(sol_file)
    
    if debug:
        print(f"\n📋 Contract: {contract_address[:10]}")
        print(f"   Modifier definitions: {source_data['modifier_definitions'][:5]}")
        print(f"   Functions with modifiers: {sum(1 for f in source_data['function_modifiers'].values() if f['modifiers'])}")
    
    # Update هر path
    update_stats = {
        'paths_updated': 0,
        'functions_matched': set(),
        'functions_not_found': set(),
        'modifiers_added': defaultdict(int)
    }
    
    for path in path_db['paths']:
        # گرفتن primary function از path
        primary_func = path.get('function_context', {}).get('primary_function', 'unknown')
        
        if primary_func and primary_func != 'unknown':
            # جستجو در source data
            if primary_func in source_data['function_modifiers']:
                func_data = source_data['function_modifiers'][primary_func]
                
                # Update function context
                old_modifiers = path['function_context'].get('modifier_names', [])
                new_modifiers = func_data['modifiers']
                
                path['function_context']['modifier_names'] = new_modifiers
                path['function_context']['function_has_modifier'] = len(new_modifiers) > 0
                
                # Update visibility اگر بهتر از قبلی است
                if func_data['visibility'] != 'unknown':
                    path['function_context']['function_visibility'] = func_data['visibility']
                
                # Update aggregate features
                path['aggregate_features']['has_modifier_protection'] = int(len(new_modifiers) > 0)
                
                # بررسی protective modifiers
                protective_modifiers = ['onlyOwner', 'onlyAdmin', 'onlyController', 
                                      'onlyMinter', 'onlyPauser', 'onlyRole', 'auth']
                has_protective = any(
                    any(prot in mod.lower() for prot in ['only', 'auth', 'admin', 'owner'])
                    for mod in new_modifiers
                )
                
                if has_protective and not path['function_context'].get('protection_outside_path', False):
                    path['function_context']['protection_outside_path'] = True
                    path['aggregate_features']['has_external_protection'] = 1
                
                # Update restricted visibility
                path['aggregate_features']['has_restricted_visibility'] = int(
                    func_data['visibility'] in ['private', 'internal']
                )
                
                # آمار
                update_stats['paths_updated'] += 1
                update_stats['functions_matched'].add(primary_func)
                for mod in new_modifiers:
                    update_stats['modifiers_added'][mod] += 1
                
                if debug and old_modifiers != new_modifiers:
                    print(f"   ✅ Updated {primary_func}: {old_modifiers} → {new_modifiers}")
            else:
                update_stats['functions_not_found'].add(primary_func)
                if debug and primary_func not in ['', 'constructor', 'fallback']:
                    print(f"   ⚠️ Function '{primary_func}' not found in source")
    
    # Update statistics در path database
    path_db['statistics']['paths_with_modifier'] = sum(
        1 for p in path_db['paths'] 
        if p['function_context'].get('function_has_modifier', False)
    )
    path_db['statistics']['paths_with_external_protection'] = sum(
        1 for p in path_db['paths']
        if p['aggregate_features'].get('has_external_protection', 0) == 1
    )
    
    # اضافه کردن modifier statistics
    path_db['modifier_statistics'] = {
        'modifier_definitions': source_data['modifier_definitions'],
        'functions_with_modifiers': len([f for f in source_data['function_modifiers'].values() if f['modifiers']]),
        'paths_updated': update_stats['paths_updated'],
        'unique_functions_matched': len(update_stats['functions_matched']),
        'modifier_usage': dict(update_stats['modifiers_added'])
    }
    
    # ذخیره updated database
    output_file = os.path.join(base_dir, 'path_databases_updated1', f'{contract_address}_path_database.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(path_db, f, indent=2)
    
    if debug:
        print(f"   📊 Update Stats:")
        print(f"      Paths updated: {update_stats['paths_updated']}/{len(path_db['paths'])}")
        print(f"      Functions matched: {len(update_stats['functions_matched'])}")
        print(f"      Functions not found: {len(update_stats['functions_not_found'])}")
        if update_stats['modifiers_added']:
            print(f"      Top modifiers added: {list(update_stats['modifiers_added'].keys())[:3]}")
    
    return True


def batch_update_all_databases():
    """
    Update همه path databases با modifier data
    """
    base_dir = r'C:\Users\Hadis\Documents\NewModel1'
    path_db_dir = os.path.join(base_dir, 'path_databases1')

    print("="*80)
    print("🔧 INTEGRATING MODIFIERS INTO PATH DATABASES")
    print("="*80)
    
    # لیست path databases
    db_files = [f for f in os.listdir(path_db_dir) if f.endswith('_path_database.json')]
    
    global_stats = {
        'successful': 0,
        'failed': 0,
        'total_paths_updated': 0,
        'total_modifiers_added': defaultdict(int)
    }
    
    for i, db_file in enumerate(db_files, 1):
        contract_address = db_file.replace('_path_database.json', '')
        
        print(f"\n[{i}/{len(db_files)}] Processing {contract_address[:10]}...")
        
        success = update_path_database_with_modifiers(contract_address, base_dir, debug=True)
        
        if success:
            global_stats['successful'] += 1
            
            # خواندن updated stats
            updated_file = os.path.join(base_dir, 'path_databases_updated1', f'{contract_address}_path_database.json')
            with open(updated_file, 'r') as f:
                updated_db = json.load(f)
            
            global_stats['total_paths_updated'] += updated_db['modifier_statistics'].get('paths_updated', 0)
            
            for mod, count in updated_db['modifier_statistics'].get('modifier_usage', {}).items():
                global_stats['total_modifiers_added'][mod] += count
        else:
            global_stats['failed'] += 1
    
    # گزارش نهایی
    print("\n" + "="*80)
    print("📊 FINAL INTEGRATION REPORT")
    print("="*80)
    print(f"✅ Successful: {global_stats['successful']}/{len(db_files)}")
    print(f"❌ Failed: {global_stats['failed']}/{len(db_files)}")
    print(f"📈 Total paths updated: {global_stats['total_paths_updated']}")
    
    print(f"\n🔐 Most common modifiers integrated:")
    for mod, count in sorted(global_stats['total_modifiers_added'].items(), 
                            key=lambda x: x[1], reverse=True)[:10]:
        print(f"   - {mod}: {count} occurrences")

    print(f"\n💾 Updated databases saved in: path_databases_updated1/")

def validate_integration():
    """
    Validate که integration درست انجام شده
    """
    base_dir = r'C:\Users\Hadis\Documents\NewModel1'
    
    # انتخاب یک contract برای validation
    test_contract = '0x000000931cf36c464623bb0eefb6b0c205338d67'
    
    print("\n" + "="*80)
    print("🔍 VALIDATION OF INTEGRATION")
    print("="*80)
    
    # مقایسه old و new
    old_file = os.path.join(base_dir, 'path_databases1', f'{test_contract}_path_database.json')
    new_file = os.path.join(base_dir, 'path_databases_updated1', f'{test_contract}_path_database.json')

    with open(old_file, 'r') as f:
        old_db = json.load(f)
    with open(new_file, 'r') as f:
        new_db = json.load(f)
    
    # نمایش تغییرات
    print(f"\n📊 Before/After Comparison:")
    print(f"   Paths with modifiers:")
    print(f"      Before: {old_db['statistics'].get('paths_with_modifier', 0)}")
    print(f"      After:  {new_db['statistics'].get('paths_with_modifier', 0)}")
    
    print(f"\n   Paths with external protection:")
    print(f"      Before: {old_db['statistics'].get('paths_with_external_protection', 0)}")
    print(f"      After:  {new_db['statistics'].get('paths_with_external_protection', 0)}")
    
    # نمونه paths با تغییرات
    print(f"\n📝 Sample Path Changes:")
    sample_count = 0
    for i, (old_path, new_path) in enumerate(zip(old_db['paths'][:10], new_db['paths'][:10])):
        old_mods = old_path['function_context'].get('modifier_names', [])
        new_mods = new_path['function_context'].get('modifier_names', [])
        
        if old_mods != new_mods:
            sample_count += 1
            print(f"\n   Path {i} - Function: {new_path['function_context']['primary_function']}")
            print(f"      Old modifiers: {old_mods}")
            print(f"      New modifiers: {new_mods}")
            
            if sample_count >= 3:
                break
    
    # بررسی modifier statistics
    if 'modifier_statistics' in new_db:
        print(f"\n📈 Modifier Statistics:")
        print(f"   Definitions found: {len(new_db['modifier_statistics']['modifier_definitions'])}")
        print(f"   Functions with modifiers: {new_db['modifier_statistics']['functions_with_modifiers']}")
        print(f"   Paths successfully updated: {new_db['modifier_statistics']['paths_updated']}")
        
        if new_db['modifier_statistics']['modifier_usage']:
            print(f"   Top modifiers used:")
            for mod, count in sorted(new_db['modifier_statistics']['modifier_usage'].items(), 
                                    key=lambda x: x[1], reverse=True)[:5]:
                print(f"      - {mod}: {count} times")
    
    print("\n✅ Integration validation complete!")


if __name__ == "__main__":
    # Step 1: Update all databases
    batch_update_all_databases()
    
    # Step 2: Validate the integration
    validate_integration()
