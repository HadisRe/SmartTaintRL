# state_builder.py (Complete revised version)
"""
State Vector Builder for RL Environment
Convert pool and contract data to state vector for agent
Enhanced version with 80 features per path
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from env2 import PoolState
from env1 import ContractData, ContractDataLoader

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class StateComponents:
    """Different components of state vector"""
    global_features: np.ndarray  # 8 dims
    pool_features: np.ndarray  # 1600 dims (20 paths × 80)
    contract_embedding: np.ndarray  # 64 dims
    pool_metadata: np.ndarray  # 8 dims

    def get_full_vector(self) -> np.ndarray:
        """Combine all components into one vector"""
        return np.concatenate([
            self.global_features,
            self.pool_features.flatten(),
            self.contract_embedding,
            self.pool_metadata
        ])


class StateBuilder:
    """
    Responsible for building state vector from environment state
    State Vector: 1680 dimensions total
    - Global: 8
    - Pool: 1600 (20 paths × 80 features)
    - Contract: 64
    - Metadata: 8
    """

    def __init__(self):
        """
        StateBuilder for structured state representation
        """
        self.max_paths = 20
        self.features_per_path = 100
        self.state_dim = (20, 100)  # Now it's a tuple, not a number

        # For tracking progress (these are for compatibility)
        self.episode_step = 0
        self.last_action = None

        logger.info(f"StateBuilder initialized for structured state {self.state_dim}")

    def _detect_critical_patterns(self, path: Dict) -> Dict[str, float]:
        """
        Detect critical security patterns
        Returns: Dictionary of pattern scores (0 or 1)
        """
        patterns = {}

        # Extract required information
        basic_info = path.get('basic_info', {})
        source_type = basic_info.get('source_type', '')
        sink_type = basic_info.get('sink_type', '')

        graph_enrichment = path.get('graph_enrichment', {})
        nodes_detail = graph_enrichment.get('nodes_detail', [])

        function_context = path.get('function_context', {})
        modifier_names = function_context.get('modifier_names', [])

        # Pattern 1: Weak randomness + Value transfer
        has_weak_random = source_type in ['timestamp', 'blocknumber', 'blockhash']
        has_value_transfer = sink_type in ['valueTransfer', 'transfer']
        patterns['weak_random_with_transfer'] = 1.0 if (has_weak_random and has_value_transfer) else 0.0

        # Pattern 2: Timestamp in condition + state modification
        has_timestamp = any('timestamp' in str(node.get('code_snippet', '')).lower()
                            for node in nodes_detail if node.get('type') == 'condition')
        has_state_mod = sink_type == 'stateModification'
        patterns['timestamp_in_condition'] = 1.0 if (has_timestamp and has_state_mod) else 0.0

        # Pattern 3: Block.number with gambling context
        has_blocknumber = source_type == 'blocknumber'
        has_gambling = any(word in str(nodes_detail).lower()
                           for word in ['lottery', 'gambling', 'bet', 'prize', 'jackpot'])
        patterns['blocknumber_gambling'] = 1.0 if (has_blocknumber and has_gambling) else 0.0

        # Pattern 4: whenNotPaused modifier with value transfer
        has_paused_modifier = 'whenNotPaused' in modifier_names
        patterns['paused_with_transfer'] = 1.0 if (has_paused_modifier and has_value_transfer) else 0.0

        # Pattern 5: Multiple randomness sources
        random_sources = sum([
            'timestamp' in str(nodes_detail).lower(),
            'block.number' in str(nodes_detail).lower(),
            'blockhash' in str(nodes_detail).lower()
        ])
        patterns['multiple_random_sources'] = 1.0 if random_sources >= 2 else 0.0

        # Pattern 6: Direct path (very short) with high impact
        path_length = basic_info.get('path_length', 0)
        patterns['short_high_impact'] = 1.0 if (path_length <= 3 and has_value_transfer) else 0.0

        # Pattern 7: No mitigation with financial operation
        mitigation_count = basic_info.get('mitigation_count', 0)
        patterns['no_mitigation_financial'] = 1.0 if (mitigation_count == 0 and has_value_transfer) else 0.0

        # Pattern 8: External call patterns
        has_transfer_node = any(node.get('type') == 'transfer' for node in nodes_detail)
        patterns['external_transfer'] = 1.0 if has_transfer_node else 0.0

        logger.debug(f"Detected patterns: {patterns}")
        return patterns

    def _extract_interaction_features(self, path: Dict) -> np.ndarray:
        """
        Extract interaction features between different elements
        Returns: 20-dimensional array
        """
        features = []

        basic_info = path.get('basic_info', {})
        aggregate = path.get('aggregate_features', {})
        function_context = path.get('function_context', {})

        source_type = basic_info.get('source_type', '')
        sink_type = basic_info.get('sink_type', '')

        # Interaction 1-4: Source-Sink combinations (4 features)
        features.append(1.0 if source_type == 'timestamp' and sink_type == 'valueTransfer' else 0.0)
        features.append(1.0 if source_type == 'blocknumber' and sink_type == 'valueTransfer' else 0.0)
        features.append(1.0 if source_type == 'blockhash' and sink_type == 'stateModification' else 0.0)
        features.append(1.0 if source_type == 'sender' and sink_type == 'randomGeneration' else 0.0)

        # Interaction 5-8: Protection interactions (4 features)
        has_modifier = aggregate.get('has_modifier_protection', 0)
        has_require = aggregate.get('require_density', 0) > 0
        has_external = aggregate.get('has_external_protection', 0)

        features.append(1.0 if has_modifier and source_type in ['timestamp', 'blocknumber'] else 0.0)
        features.append(1.0 if has_require and sink_type == 'valueTransfer' else 0.0)
        features.append(1.0 if has_external and source_type == 'blockhash' else 0.0)
        features.append(1.0 if not has_modifier and not has_require and sink_type == 'valueTransfer' else 0.0)

        # Interaction 9-12: Path complexity interactions (4 features)
        path_length = basic_info.get('path_length', 0)
        has_data_flow = aggregate.get('has_data_flow', 0)

        features.append(1.0 if path_length <= 5 and sink_type == 'valueTransfer' else 0.0)
        features.append(1.0 if path_length > 10 and has_data_flow else 0.0)
        features.append(1.0 if aggregate.get('keccak_density', 0) > 0 and source_type == 'timestamp' else 0.0)
        features.append(
            1.0 if aggregate.get('condition_density', 0) > 0.3 and sink_type == 'stateModification' else 0.0)

        # Interaction 13-16: Function context interactions (4 features)
        visibility = function_context.get('function_visibility', '')
        function_requires = function_context.get('function_require_count', 0)

        features.append(1.0 if visibility == 'public' and sink_type == 'valueTransfer' else 0.0)
        features.append(1.0 if visibility == 'internal' and source_type == 'timestamp' else 0.0)
        features.append(1.0 if function_requires == 0 and sink_type == 'valueTransfer' else 0.0)
        features.append(1.0 if function_requires >= 3 and source_type == 'blocknumber' else 0.0)

        # Interaction 17-20: Mixed interactions (4 features)
        features.append(aggregate.get('unique_functions_ratio', 0) * aggregate.get('has_data_flow', 0))
        features.append(aggregate.get('node_diversity', 0) * (1.0 if sink_type == 'valueTransfer' else 0.0))
        features.append(aggregate.get('distance_to_sink', 0) * (1.0 if source_type == 'timestamp' else 0.0))
        features.append(aggregate.get('mitigation_score', 0) * aggregate.get('require_density', 0))



        return np.array(features, dtype=np.float32)

    def _extract_semantic_features(self, path: Dict) -> np.ndarray:
        """
        Extract semantic features from code
        Returns: 15-dimensional array
        """
        features = []

        graph_enrichment = path.get('graph_enrichment', {})
        nodes_detail = graph_enrichment.get('nodes_detail', [])

        # Collect all code snippets
        all_code = ' '.join([node.get('code_snippet', '') for node in nodes_detail]).lower()

        # Feature 1-5: Presence of specific keywords (5 features)
        features.append(1.0 if 'keccak256' in all_code else 0.0)
        features.append(1.0 if 'transfer' in all_code else 0.0)
        features.append(1.0 if 'random' in all_code else 0.0)
        features.append(1.0 if 'now' in all_code or 'timestamp' in all_code else 0.0)
        features.append(1.0 if 'block.' in all_code else 0.0)

        # Feature 6-10: Types of operations (5 features)
        features.append(1.0 if 'msg.sender' in all_code else 0.0)
        features.append(1.0 if 'msg.value' in all_code else 0.0)
        features.append(1.0 if 'require(' in all_code else 0.0)
        features.append(1.0 if 'assert(' in all_code else 0.0)
        features.append(1.0 if 'revert(' in all_code else 0.0)

        # Feature 11-13: Code complexity (3 features)
        node_types = graph_enrichment.get('node_types_count', {})
        features.append(min(node_types.get('assignment', 0) / 10.0, 1.0))
        features.append(min(node_types.get('condition', 0) / 5.0, 1.0))
        features.append(min(node_types.get('require', 0) / 3.0, 1.0))

        # Feature 14-15: Functions involved (2 features)
        functions = graph_enrichment.get('functions_involved', [])
        features.append(min(len(functions) / 5.0, 1.0))
        features.append(1.0 if 'random' in str(functions).lower() else 0.0)

        return np.array(features, dtype=np.float32)

    def _extract_path_features(self, path: Dict) -> np.ndarray:
        """
        Actual feature extraction from path using all available information
        Returns: 100-dimensional array
        """
        all_features = []

        # =============== Section 1: Main features (25) ===============
        features = path.get('aggregate_features', {})
        basic_info = path.get('basic_info', {})

        numerical_features = [
            features.get('path_length_normalized', 0),
            features.get('require_density', 0),
            features.get('condition_density', 0),
            features.get('keccak_density', 0),
            features.get('mitigation_score', 0),
            features.get('has_any_mitigation', 0),
            features.get('has_strong_mitigation', 0),
            features.get('has_modifier_protection', 0),
            features.get('has_restricted_visibility', 0),
            features.get('has_external_protection', 0),
            features.get('function_require_density', 0),
            features.get('unique_functions_ratio', 0),
            features.get('node_diversity', 0),
            features.get('distance_to_sink', 0),
            features.get('distance_from_source', 0),
            features.get('has_data_flow', 0),
            features.get('contains_loop', 0)
        ]

        source_encoded = features.get('source_type_encoded', [0, 0, 0, 0])
        sink_encoded = features.get('sink_type_encoded', [0, 0, 0, 0])

        all_features.extend(numerical_features)
        all_features.extend(source_encoded)
        all_features.extend(sink_encoded)

        # =============== Section 2: Modifier Context (15) ===============
        function_context = path.get('function_context', {})
        modifier_names = function_context.get('modifier_names', [])

        # Specific modifiers that are important
        important_modifiers = [
            'onlyOwner', 'onlyAdmin', 'onlyMinter',
            'whenNotPaused', 'nonReentrant', 'lock',
            'isHuman', 'onlyEOA', 'onlyWhitelisted'
        ]

        for mod in important_modifiers:
            all_features.append(1.0 if mod in modifier_names else 0.0)

        # Total number of modifiers
        all_features.append(min(len(modifier_names) / 3.0, 1.0))

        # Visibility encoding (5 features)
        visibility = function_context.get('function_visibility', '')
        all_features.extend([
            1.0 if visibility == 'public' else 0.0,
            1.0 if visibility == 'external' else 0.0,
            1.0 if visibility == 'internal' else 0.0,
            1.0 if visibility == 'private' else 0.0,
            1.0 if visibility == '' else 0.0  # unknown
        ])

        # =============== Section 3: Sequence Analysis (20) ===============
        graph_enrichment = path.get('graph_enrichment', {})
        nodes_detail = graph_enrichment.get('nodes_detail', [])
        path_nodes = basic_info.get('path_nodes', [])

        # Position of require nodes relative to sensitive operations
        require_positions = []
        transfer_positions = []
        keccak_positions = []

        for i, node in enumerate(nodes_detail):
            node_type = node.get('type', '')
            if node_type == 'require':
                require_positions.append(i / max(len(nodes_detail), 1))
            elif node_type == 'transfer' or 'transfer' in node.get('code_snippet', ''):
                transfer_positions.append(i / max(len(nodes_detail), 1))
            elif node_type == 'keccak' or 'keccak' in node.get('code_snippet', ''):
                keccak_positions.append(i / max(len(nodes_detail), 1))

        # Features for positions
        all_features.extend([
            min(require_positions) if require_positions else 1.0,  # where is first require
            max(require_positions) if require_positions else 0.0,  # where is last require
            min(transfer_positions) if transfer_positions else 1.0,  # first transfer
            max(transfer_positions) if transfer_positions else 0.0,  # last transfer
            # Is require before transfer?
            1.0 if (require_positions and transfer_positions and
                    min(require_positions) < max(transfer_positions)) else 0.0,
            # Are all transfers after requires?
            1.0 if (require_positions and transfer_positions and
                    max(require_positions) < min(transfer_positions)) else 0.0,
        ])

        # Analysis of functions involved
        functions = graph_enrichment.get('functions_involved', [])
        all_features.extend([
            1.0 if 'random' in str(functions).lower() else 0.0,
            1.0 if 'transfer' in str(functions).lower() else 0.0,
            1.0 if 'withdraw' in str(functions).lower() else 0.0,
            1.0 if 'mint' in str(functions).lower() else 0.0,
            1.0 if 'burn' in str(functions).lower() else 0.0,
            1.0 if 'bet' in str(functions).lower() or 'gamble' in str(functions).lower() else 0.0,
        ])

        # Node type distribution in sequence
        node_types = graph_enrichment.get('node_types_count', {})
        total_nodes = sum(node_types.values()) if node_types else 1
        all_features.extend([
            node_types.get('require', 0) / total_nodes,
            node_types.get('assignment', 0) / total_nodes,
            node_types.get('condition', 0) / total_nodes,
            node_types.get('transfer', 0) / total_nodes,
            node_types.get('keccak', 0) / total_nodes,
            # Entropy of distribution
            -sum((c / total_nodes) * np.log(c / total_nodes + 1e-10)
                 for c in node_types.values()) if node_types else 0
        ])

        # Distance between critical operations
        if transfer_positions and require_positions:
            distances = [abs(t - r) for t in transfer_positions for r in require_positions]
            all_features.extend([
                min(distances),
                max(distances)
            ])
        else:
            all_features.extend([0.0, 0.0])

        # =============== Section 4: Code Pattern Analysis (20) ===============
        all_code = ' '.join([node.get('code_snippet', '') for node in nodes_detail]).lower()

        # Critical operations
        critical_ops = [
            'msg.sender', 'msg.value', 'tx.origin',
            'block.timestamp', 'block.number', 'block.difficulty',
            'blockhash', 'now', 'random',
            'transfer', 'send', 'call.value',
            'delegatecall', 'selfdestruct', 'suicide'
        ]

        for op in critical_ops:
            all_features.append(1.0 if op in all_code else 0.0)

        # Math operations (overflow potential)
        all_features.extend([
            1.0 if '*' in all_code and 'safem' not in all_code else 0.0,  # unsafe multiply
            1.0 if '+' in all_code and 'safem' not in all_code else 0.0,  # unsafe add
            1.0 if '-' in all_code and 'safem' not in all_code else 0.0,  # unsafe subtract
            1.0 if '/' in all_code else 0.0,  # division
            1.0 if '%' in all_code else 0.0,  # modulo (often in random)
        ])

        # =============== Section 5: Risk Context (20) ===============
        risk_factors = basic_info.get('_risk_content', [])
        mitigation_content = basic_info.get('_mitigation_content', [])

        # Parse risk factors
        risk_keywords = {
            'weak entropy': 'Weak entropy',
            'gambling': 'gambling',
            'lottery': 'lottery',
            'financial': 'financial',
            'direct impact': 'Direct',
            'short path': 'Short path',
            'admin': 'Admin'
        }

        for key, search_term in risk_keywords.items():
            found = any(search_term in str(factor) for factor in risk_factors)
            all_features.append(1.0 if found else 0.0)

        # Parse mitigations
        mitigation_keywords = {
            'access control': 'access control',
            'admin only': 'Admin-only',
            'require check': 'require',
            'modifier': 'modifier',
            'validation': 'validation'
        }

        for key, search_term in mitigation_keywords.items():
            found = any(search_term in str(mit) for mit in mitigation_content)
            all_features.append(1.0 if found else 0.0)

        # Combined risk assessment
        all_features.extend([
            len(risk_factors) / 5.0,  # normalized risk count
            len(mitigation_content) / 3.0,  # normalized mitigation count
            (len(risk_factors) - len(mitigation_content)) / 5.0,  # net risk
            basic_info.get('mitigation_count', 0) / 5.0,
            basic_info.get('risk_factors_count', 0) / 5.0,
            1.0 if basic_info.get('has_data_flow', False) else 0.0,
            1.0 if basic_info.get('contains_loop', False) else 0.0,
            basic_info.get('path_length', 0) / 20.0,  # normalized path length
        ])

        # Convert to numpy array
        feature_vector = np.array(all_features[:100], dtype=np.float32)

        # Pad if necessary
        if len(feature_vector) < 100:
            feature_vector = np.pad(feature_vector, (0, 100 - len(feature_vector)))

        return feature_vector

    def build_state(self, contract: ContractData, pool_state: PoolState,
                    budget_used: float = 0.0, max_steps: int = 50) -> np.ndarray:
        """
        Build structured state for per-path Q-learning
        Returns: array with shape (20, 100) - 20 paths, each with 100 features
        """
        max_paths = 20
        features_per_path = 100

        # Initialize state array
        state = np.zeros((max_paths, features_per_path), dtype=np.float32)

        # Fill features for each path in pool
        for i, path in enumerate(pool_state.current_pool[:max_paths]):
            # Extract 100 features for this path
            path_features = self._extract_path_features(path)

            # Ensure correct size
            if len(path_features) < features_per_path:
                path_features = np.pad(path_features,
                                       (0, features_per_path - len(path_features)))
            elif len(path_features) > features_per_path:
                path_features = path_features[:features_per_path]

            state[i] = path_features

        # Paths that don't exist are filled with 0 (padding)
        # This tells the network that these slots are empty

        logger.debug(f"Structured state built: shape={state.shape}, "
                     f"active paths={len(pool_state.current_pool)}")

        return state

    def build_state_with_metadata(self, contract: ContractData, pool_state: PoolState,
                                  budget_used: float = 0.0, max_steps: int = 50) -> Tuple[np.ndarray, List[Dict]]:
        """
        Build structured state along with metadata for localization

        Returns:
            state: array with shape (20, 100) - same as regular state
            metadata: list of metadata dictionaries for each path
        """
        max_paths = 20
        features_per_path = 100

        # Initialize state array
        state = np.zeros((max_paths, features_per_path), dtype=np.float32)
        metadata_list = []

        # Fill features and metadata for each path in pool
        for i, path in enumerate(pool_state.current_pool[:max_paths]):
            # Extract features (same procedure as before)
            path_features = self._extract_path_features(path)

            # Ensure correct size
            if len(path_features) < features_per_path:
                path_features = np.pad(path_features,
                                       (0, features_per_path - len(path_features)))
            elif len(path_features) > features_per_path:
                path_features = path_features[:features_per_path]

            state[i] = path_features

            # Extract metadata for localization
            basic_info = path.get('basic_info', {})
            graph_enrichment = path.get('graph_enrichment', {})
            nodes_detail = graph_enrichment.get('nodes_detail', [])

            # Build metadata for this path
            path_metadata = {
                'path_id': path.get('id', f'path_{i}'),
                'contract_address': contract.address if hasattr(contract, 'address') else 'unknown',
                'source_file': basic_info.get('source_file', 'unknown'),
                'path_length': basic_info.get('path_length', 0),
                'source_type': basic_info.get('source_type', 'unknown'),
                'sink_type': basic_info.get('sink_type', 'unknown'),
                'nodes': []
            }

            # Extract information for each node in path
            for node_idx, node in enumerate(nodes_detail):
                node_metadata = {
                    'node_index': node_idx,
                    'node_id': node.get('node_id', f'N{node_idx}'),
                    'node_type': node.get('type', 'unknown'),
                    'line_number': 0,
                    'column_number': 0,
                    'code_snippet': node.get('code_snippet', ''),
                    'function_name': node.get('function', ''),
                    'is_source': node.get('is_source', False),
                    'is_sink': node.get('is_sink', False)
                }
                path_metadata['nodes'].append(node_metadata)

            metadata_list.append(path_metadata)

        # For paths that don't exist (padding), add empty metadata
        while len(metadata_list) < max_paths:
            metadata_list.append(None)

        logger.debug(f"State with metadata built: shape={state.shape}, "
                     f"active paths={len([m for m in metadata_list if m is not None])}")

        return state, metadata_list

    def build_state_with_mapping(self, contract: ContractData, pool_state: PoolState,
                                 budget_used: float = 0.0, max_steps: int = 50) -> Tuple[
        np.ndarray, List[Dict], List[Dict]]:
        """
        Build structured state with metadata and feature-to-node mapping

        Returns:
            state: array with shape (20, 100)
            metadata: list of metadata dicts for each path
            feature_mapping: list of dicts mapping feature indices to node indices
        """

        max_paths = 20
        features_per_path = 100

        state = np.zeros((max_paths, features_per_path), dtype=np.float32)
        metadata_list = []
        mapping_list = []

        for i, path in enumerate(pool_state.current_pool[:max_paths]):
            path_features, feature_mapping = self._extract_path_features_with_mapping(path)

            if len(path_features) < features_per_path:
                path_features = np.pad(path_features, (0, features_per_path - len(path_features)))
            elif len(path_features) > features_per_path:
                path_features = path_features[:features_per_path]

            state[i] = path_features

            basic_info = path.get('basic_info', {})
            graph_enrichment = path.get('graph_enrichment', {})
            nodes_detail = graph_enrichment.get('nodes_detail', [])

            path_metadata = {
                'path_id': path.get('id', f'path_{i}'),
                'contract_address': contract.address if hasattr(contract, 'address') else 'unknown',
                'source_file': basic_info.get('source_file', 'unknown'),
                'path_length': basic_info.get('path_length', 0),
                'source_type': basic_info.get('source_type', 'unknown'),
                'sink_type': basic_info.get('sink_type', 'unknown'),
                'nodes': []
            }

            for node_idx, node in enumerate(nodes_detail):
                # print(f"Node {node_idx}: {node}")
                node_metadata = {
                    'node_index': node_idx,
                    'node_type': node.get('type', 'unknown'),
                    'line_number': node.get('line_number', 0),
                    'column_number': node.get('column_number', 0),
                    'code_snippet': node.get('code_snippet', ''),
                    'function_name': node.get('function', ''),
                    'node_id': node.get('node_id', ''),

                    'is_source': node.get('is_source', False),
                    'is_sink': node.get('is_sink', False)
                }
                path_metadata['nodes'].append(node_metadata)

            metadata_list.append(path_metadata)
            mapping_list.append(feature_mapping)

        while len(metadata_list) < max_paths:
            metadata_list.append(None)
            mapping_list.append(None)

        logger.debug(
            f"State with mapping built: shape={state.shape}, active paths={len([m for m in metadata_list if m is not None])}")

        return state, metadata_list, mapping_list

    def _extract_path_features_with_mapping(self, path: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Extract features and create mapping from features to nodes

        Returns:
            features: 100-dim array
            mapping: dict with structure {feature_index: node_index}
        """
        all_features = []
        feature_to_node = {}
        current_feature_idx = 0

        features = path.get('aggregate_features', {})
        basic_info = path.get('basic_info', {})
        graph_enrichment = path.get('graph_enrichment', {})
        nodes_detail = graph_enrichment.get('nodes_detail', [])

        # Section 1: Basic features (25 features) - these are aggregate, no specific node
        numerical_features = [
            features.get('path_length_normalized', 0),
            features.get('require_density', 0),
            features.get('condition_density', 0),
            features.get('keccak_density', 0),
            features.get('mitigation_score', 0),
            features.get('has_any_mitigation', 0),
            features.get('has_strong_mitigation', 0),
            features.get('has_modifier_protection', 0),
            features.get('has_restricted_visibility', 0),
            features.get('has_external_protection', 0),
            features.get('function_require_density', 0),
            features.get('unique_functions_ratio', 0),
            features.get('node_diversity', 0),
            features.get('distance_to_sink', 0),
            features.get('distance_from_source', 0),
            features.get('has_data_flow', 0),
            features.get('contains_loop', 0)
        ]

        source_encoded = features.get('source_type_encoded', [0, 0, 0, 0])
        sink_encoded = features.get('sink_type_encoded', [0, 0, 0, 0])

        all_features.extend(numerical_features)
        all_features.extend(source_encoded)
        all_features.extend(sink_encoded)

        for idx in range(current_feature_idx, len(all_features)):
            feature_to_node[idx] = 'aggregate'

        current_feature_idx = len(all_features)

        # Section 2: Node-specific features
        # Map each node to specific feature ranges
        for node_idx, node in enumerate(nodes_detail[:20]):
            start_idx = current_feature_idx

            node_type = node.get('type', 'unknown')
            code_snippet = node.get('code_snippet', '').lower()

            node_features = [
                1.0 if node_type == 'require' else 0.0,
                1.0 if node_type == 'condition' else 0.0,
                1.0 if node_type == 'transfer' else 0.0,
                1.0 if node_type == 'keccak' else 0.0,
                1.0 if 'timestamp' in code_snippet else 0.0,
                1.0 if 'block.number' in code_snippet else 0.0,
                1.0 if 'blockhash' in code_snippet else 0.0,
                1.0 if 'msg.sender' in code_snippet else 0.0,
                1.0 if 'msg.value' in code_snippet else 0.0,
                1.0 if node.get('is_source', False) else 0.0,
                1.0 if node.get('is_sink', False) else 0.0,
            ]

            all_features.extend(node_features)

            end_idx = len(all_features)
            for feat_idx in range(start_idx, end_idx):
                feature_to_node[feat_idx] = node_idx

            current_feature_idx = len(all_features)

            if current_feature_idx >= 100:
                break

        feature_vector = np.array(all_features[:100], dtype=np.float32)

        if len(feature_vector) < 100:
            feature_vector = np.pad(feature_vector, (0, 100 - len(feature_vector)))

        return feature_vector, feature_to_node

    def _build_global_features(self,
                               pool_state: PoolState,
                               budget_used: float,
                               max_steps: int) -> np.ndarray:
        """
        Build global features (8 dimensions)
        """
        total_paths = (len(pool_state.current_pool) +
                       len(pool_state.explored_paths) +
                       len(pool_state.available_paths))

        features = [
            # 1. Budget consumption ratio
            np.clip(budget_used, 0, 1),

            # 2. Normalized total path count
            min(total_paths / 100.0, 1.0),

            # 3. Exploration progress ratio
            len(pool_state.explored_paths) / max(total_paths, 1),

            # 4. Pool utilization rate
            len(pool_state.current_pool) / max(self.max_pool_size, 1),

            # 5-8. Last action one-hot (4 dims: ANALYZE, SKIP, NONE, OTHER)
            1.0 if self.last_action == 'ANALYZE' else 0.0,
            1.0 if self.last_action == 'SKIP' else 0.0,
            1.0 if self.last_action is None else 0.0,
            0.0  # Reserved for future actions
        ]

        features = np.array(features, dtype=np.float32)

        logger.debug(f"Global features: budget={features[0]:.2f}, "
                     f"exploration={features[2]:.2f}, pool_util={features[3]:.2f}")

        return features

    def _build_pool_features(self, current_pool: List[Dict]) -> np.ndarray:
        """
        Build pool features (1600 dimensions = 20 paths × 80 features)
        """
        pool_features = np.zeros((self.max_pool_size, self.path_features_dim),
                                 dtype=np.float32)

        for i, path in enumerate(current_pool[:self.max_pool_size]):
            # Extract 80 features from path with new method
            path_vector = self._extract_path_features(path)

            # Validation
            if path_vector.shape[0] != self.path_features_dim:
                logger.warning(f"Path {i} has {path_vector.shape[0]} features, "
                               f"expected {self.path_features_dim}")
                # Pad or truncate if necessary
                if path_vector.shape[0] < self.path_features_dim:
                    path_vector = np.pad(path_vector,
                                         (0, self.path_features_dim - path_vector.shape[0]))
                else:
                    path_vector = path_vector[:self.path_features_dim]

            pool_features[i] = path_vector

        # Log statistics
        non_zero_paths = np.sum(np.any(pool_features != 0, axis=1))
        mean_features = np.mean(np.sum(pool_features != 0, axis=1))

        logger.debug(f"Pool features: {non_zero_paths}/{self.max_pool_size} non-zero paths, "
                     f"avg non-zero features per path: {mean_features:.1f}")

        return pool_features

    def _build_contract_embedding(self, contract: ContractData) -> np.ndarray:
        """
        Build contract embedding (64 dimensions)
        Note: We don't include label and risk_distribution (information leak)
        """
        embedding = []

        # 1. AST features (20 dims)
        ast_features = contract.profile.get('ast_features', {})
        embedding.extend([
            ast_features.get('functions_count', 0) / 100.0,  # normalized
            ast_features.get('state_vars_count', 0) / 50.0,
            ast_features.get('functions_with_require', 0) / 50.0,
            ast_features.get('functions_with_modifier', 0) / 50.0,
        ])
        # Pad to 20
        embedding.extend([0.0] * 16)

        # 2. Graph statistics (20 dims)
        graph_stats = contract.profile.get('graph_statistics', {})
        embedding.extend([
            graph_stats.get('source_nodes', 0) / 10.0,
            graph_stats.get('sink_nodes', 0) / 50.0,
            graph_stats.get('total_nodes', 0) / 300.0,
            graph_stats.get('total_edges', 0) / 400.0,
        ])
        # Source/sink type distributions
        source_types = graph_stats.get('source_types', {})
        embedding.extend([
            source_types.get('timestamp', 0) / 10.0,
            source_types.get('blockhash', 0) / 10.0,
            source_types.get('blocknumber', 0) / 10.0,
            source_types.get('sender', 0) / 10.0,
        ])
        # Pad to 20
        embedding.extend([0.0] * 8)

        # 3. Path features from profile (20 dims)
        # Note: only total_paths and avg_mitigations, not risk_distribution!
        taint_summary = contract.profile.get('taint_summary', {})
        embedding.extend([
            taint_summary.get('total_paths', 0) / 100.0,
            taint_summary.get('avg_mitigations', 0),
            taint_summary.get('paths_with_mitigation', 0) / max(taint_summary.get('total_paths', 1), 1),
        ])
        # Pad to 20
        embedding.extend([0.0] * 17)

        # 4. Consolidated features (4 dims)
        consolidated = contract.profile.get('consolidated_features', {})
        embedding.extend([
            consolidated.get('complexity_score', 0),
            consolidated.get('require_density', 0),
            consolidated.get('modifier_usage', 0),
            consolidated.get('source_sink_ratio', 0),
            # not protection_coverage (might be a leak)
        ])

        embedding = np.array(embedding[:self.contract_embedding_dim], dtype=np.float32)

        # Pad if necessary to exactly 64
        if len(embedding) < self.contract_embedding_dim:
            embedding = np.pad(embedding,
                               (0, self.contract_embedding_dim - len(embedding)))

        logger.debug(f"Contract embedding built: {len(embedding)} dims, "
                     f"range=[{embedding.min():.3f}, {embedding.max():.3f}]")

        return embedding

    def _build_pool_metadata(self, pool_state: PoolState) -> np.ndarray:
        """
        Build pool metadata (8 dimensions)
        """
        total_paths = (len(pool_state.current_pool) +
                       len(pool_state.explored_paths) +
                       len(pool_state.available_paths))

        metadata = [
            # 1. Current pool size (normalized)
            len(pool_state.current_pool) / self.max_pool_size,

            # 2. Explored ratio
            len(pool_state.explored_paths) / max(total_paths, 1),

            # 3. Available ratio
            len(pool_state.available_paths) / max(total_paths, 1),

            # 4. Refill count (normalized)
            min(pool_state.refill_count / 5.0, 1.0),

            # 5. Average analysis rate
            pool_state.total_analyzed / max(self.episode_step, 1) if self.episode_step > 0 else 0,

            # 6. Pool diversity (placeholder)
            0.5,  # TODO: calculate actual diversity

            # 7. Episode progress
            min(self.episode_step / 50.0, 1.0),

            # 8. Reserved
            0.0
        ]

        metadata = np.array(metadata, dtype=np.float32)

        logger.debug(f"Pool metadata: explored={metadata[1]:.2f}, "
                     f"available={metadata[2]:.2f}, refills={pool_state.refill_count}")

        return metadata

    def update_step(self, action: str):
        """Update for next step"""
        self.episode_step += 1
        self.last_action = action
        logger.debug(f"Step updated: {self.episode_step}, last_action={action}")

    def reset(self):
        """Reset for new episode"""
        self.episode_step = 0
        self.last_action = None
        logger.debug("StateBuilder reset for new episode")

    def get_state_info(self) -> Dict:
        """Debug information about enhanced state"""
        return {
            'state_dim': self.state_dim,
            'max_pool_size': self.max_pool_size,
            'path_features_dim': self.path_features_dim,
            'feature_breakdown': {
                'original': 25,
                'interaction': 20,
                'semantic': 15,
                'pattern': 8,
                'reserved': 12,
                'total': 80
            },
            'contract_embedding_dim': self.contract_embedding_dim,
            'episode_step': self.episode_step,
            'last_action': self.last_action
        }
