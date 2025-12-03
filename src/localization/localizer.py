 """
SmartTaintRL Vulnerability Localizer
Function-level and Node-level localization for Bad Randomness vulnerabilities
"""

import torch
import torch.nn as nn
import numpy as np
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class AttentionDQN(nn.Module):
    """Attention-based Deep Q-Network for path analysis"""
    
    def __init__(self, state_dim: int = 100, hidden_dim: int = 64):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=state_dim, num_heads=4, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        attn_out, attn_weights = self.attention(x, x, x)
        pooled = attn_out.mean(dim=1)
        q_values = self.fc(pooled)
        return q_values, attn_weights


class VulnerabilityLocalizer:
    """Localizes Bad Randomness vulnerabilities at function and node levels"""
    
    def __init__(self, model_path: str, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.model.eval()
    
    def _load_model(self, path: str) -> AttentionDQN:
        model = AttentionDQN()
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        return model.to(self.device)
    
    def load_path_database(self, path: str) -> Optional[Dict]:
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except:
            return None
    
    def load_semantic_graph(self, path: str) -> Optional[Dict]:
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except:
            return None
    
    def group_paths_by_function(self, path_db: Dict) -> Dict[str, List]:
        """Group paths by their primary function"""
        grouped = defaultdict(list)
        for path in path_db.get('paths', []):
            func_ctx = path.get('function_context', {})
            func_name = func_ctx.get('primary_function', 'unknown')
            if func_name and func_name != 'unknown':
                grouped[func_name].append(path)
        return dict(grouped)
    
    def extract_features(self, path: Dict) -> np.ndarray:
        """Extract 100-dimensional feature vector from path"""
        features = np.zeros(100)
        
        basic = path.get('basic_info', {})
        features[0] = basic.get('path_length', 0) / 20.0
        features[1] = basic.get('total_edges', 0) / 30.0
        
        graph = path.get('graph_enrichment', {})
        features[10] = graph.get('has_loop', 0)
        features[11] = graph.get('has_external_call', 0)
        features[12] = graph.get('has_state_modification', 0)
        
        agg = path.get('aggregate_features', {})
        features[20] = agg.get('source_count', 0) / 5.0
        features[21] = agg.get('sink_count', 0) / 5.0
        features[22] = agg.get('has_block_number', 0)
        features[23] = agg.get('has_block_timestamp', 0)
        features[24] = agg.get('has_blockhash', 0)
        features[25] = agg.get('has_block_difficulty', 0)
        features[26] = agg.get('has_keccak', 0)
        features[27] = agg.get('has_transfer', 0)
        features[28] = agg.get('has_send', 0)
        features[29] = agg.get('has_call_value', 0)
        
        return features
    
    def rank_functions(self, grouped_paths: Dict[str, List], top_k: int = 5) -> List[Tuple[str, float]]:
        """Rank functions by vulnerability score using Q-values"""
        scores = {}
        
        for func_name, paths in grouped_paths.items():
            q_sum = 0.0
            analyze_votes = 0
            
            for path in paths:
                features = self.extract_features(path)
                state = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    q_values, _ = self.model(state)
                    q_analyze = q_values[0, 0].item()
                    q_skip = q_values[0, 1].item()
                
                q_sum += q_analyze
                if q_analyze > q_skip:
                    analyze_votes += 1
            
            scores[func_name] = q_sum + analyze_votes
        
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
    
    def localize_nodes(self, paths: List[Dict], top_k: int = 5) -> List[str]:
        """Localize vulnerable nodes within paths using gradient-based attribution"""
        node_scores = defaultdict(float)
        
        for path in paths:
            features = self.extract_features(path)
            state = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            state.requires_grad = True
            
            q_values, attn_weights = self.model(state)
            q_analyze = q_values[0, 0]
            q_analyze.backward()
            
            gradients = state.grad[0].abs().cpu().numpy()
            
            basic = path.get('basic_info', {})
            source_node = basic.get('source_node', '')
            sink_node = basic.get('sink_node', '')
            
            source_grad = float(np.mean(gradients[20:30]))
            sink_grad = float(np.mean(gradients[27:30]))
            
            if source_node:
                node_scores[source_node] += source_grad * 2.0
            if sink_node:
                node_scores[sink_node] += sink_grad * 2.0
        
        ranked = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
        return [node for node, _ in ranked[:top_k]]
    
    def extract_call_graph(self, semantic_graph: Dict) -> Dict[str, List[str]]:
        """Extract function call relationships from semantic graph"""
        call_graph = defaultdict(list)
        
        func_nodes = {}
        for node in semantic_graph.get('nodes', []):
            func = node.get('function', '')
            if func:
                if func not in func_nodes:
                    func_nodes[func] = []
                func_nodes[func].append(node)
        
        for func, nodes in func_nodes.items():
            for node in nodes:
                code = node.get('code_snippet', '')
                if code:
                    for other_func in func_nodes.keys():
                        if other_func != func:
                            pattern = rf'\b{re.escape(other_func)}\s*\('
                            if re.search(pattern, code):
                                if other_func not in call_graph[func]:
                                    call_graph[func].append(other_func)
        
        return dict(call_graph)
    
    def check_caller_relationship(self, call_graph: Dict, predicted: str, ground_truth: str) -> bool:
        """Check if predicted function calls the ground truth function"""
        if predicted in call_graph:
            if ground_truth in call_graph[predicted]:
                return True
        return False
    
    def localize(self, path_db: Dict, semantic_graph: Dict = None, 
                 top_k_functions: int = 5, top_k_nodes: int = 5) -> Dict:
        """
        Main localization method
        
        Returns:
            Dict with 'functions' (ranked list) and 'nodes' (per-function node lists)
        """
        grouped = self.group_paths_by_function(path_db)
        if not grouped:
            return {'functions': [], 'nodes': {}}
        
        ranked_funcs = self.rank_functions(grouped, top_k_functions)
        
        result = {
            'functions': ranked_funcs,
            'nodes': {}
        }
        
        for func_name, _ in ranked_funcs:
            if func_name in grouped:
                nodes = self.localize_nodes(grouped[func_name], top_k_nodes)
                result['nodes'][func_name] = nodes
        
        if semantic_graph:
            result['call_graph'] = self.extract_call_graph(semantic_graph)
        
        return result


def compute_metrics(gt_nodes: List[str], pred_nodes: List[str], k: int = 5) -> Dict[str, float]:
    """Compute precision, recall, F1 at k"""
    pred_k = pred_nodes[:k]
    
    if not gt_nodes:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    hits = len(set(pred_k) & set(gt_nodes))
    precision = hits / len(pred_k) if pred_k else 0.0
    recall = hits / len(gt_nodes) if gt_nodes else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}
