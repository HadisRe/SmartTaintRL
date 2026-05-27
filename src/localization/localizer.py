

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import re
import sys

import networkx as nx
import numpy as np
import torch
import torch.nn as nn


# Local copy of the network used during RL training (see
# rl_agent/dqn_agent.py). Keeping a copy here avoids importing the whole
# training stack just to run localization. If the agent's architecture
# changes, mirror the change below.
class DQN(nn.Module):
    """Per-path Q-network with self-attention over the path pool."""

    def __init__(self, path_features_dim: int = 100):
        super().__init__()
        self.path_features_dim = path_features_dim

        self.attention_query = nn.Linear(path_features_dim, 64)
        self.attention_key = nn.Linear(path_features_dim, 64)
        self.attention_value = nn.Linear(path_features_dim, 64)
        self.attention_scale = 64 ** 0.5

        self.path_evaluator = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, state: torch.Tensor, return_attention: bool = False):
        """state: (batch, num_paths, features) -> Q: (batch, num_paths, 2)."""
        batch_size, num_paths, features = state.shape

        state_reshaped = state.view(-1, features)
        Q = self.attention_query(state_reshaped).view(batch_size, num_paths, 64)
        K = self.attention_key(state_reshaped).view(batch_size, num_paths, 64)
        V = self.attention_value(state_reshaped).view(batch_size, num_paths, 64)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.attention_scale
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V).view(-1, 64)

        q_values = self.path_evaluator(attention_output)
        q_values = q_values.view(batch_size, num_paths, 2)

        if return_attention:
            return q_values, attention_weights
        return q_values



# ---------------------------------------------------------------------------
# Node-type weights used when distributing per-feature gradients to nodes.
# These weights reflect how informative each node type is for Bad Randomness
# (entropy sources and value-transfer sinks are most informative).
# ---------------------------------------------------------------------------
NODE_TYPE_WEIGHT = {
    'source': 1.0,
    'sink': 1.0,
    'require': 0.5,
    'condition': 0.5,
    'keccak': 0.7,
    'transfer': 0.9,
    'assignment': 0.3,
    'unknown': 0.2,
}


class VulnerabilityLocalizer:
    """Two-level (function, node) localizer for Bad Randomness vulnerabilities."""

    def __init__(self, model_path: str, device: Optional[str] = None,
                 alpha: float = 1.0, beta: float = 1.0,
                 gamma: float = 1.0, delta: float = 1.0,
                 propagation_decay: float = 0.7):
        """
        Args:
            model_path: path to the trained DQN checkpoint (.pth).
            device: 'cuda' or 'cpu'. Auto-detected if None.
            alpha, beta, gamma, delta: weights of the four scoring
                components. Setting one to zero disables that channel.
            propagation_decay: multiplicative decay applied at each hop
                during forward/backward propagation on the path subgraph.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.propagation_decay = propagation_decay

        self.model = self._load_model(model_path)
        self.model.eval()

    # ------------------------------------------------------------------ I/O

    def _load_model(self, path: str) -> DQN:
        """Load the DQN checkpoint produced by dqn_agent.py."""
        model = DQN(path_features_dim=100)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # The training pipeline may save either the online or the target net;
        # both share the same architecture so a plain load is sufficient.
        model.load_state_dict(state_dict)
        return model.to(self.device)

    @staticmethod
    def load_path_database(path: str) -> Optional[Dict]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def load_semantic_graph(path: str) -> Optional[Dict]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    # -------------------------------------------------------------- Features

    # 100-dim feature layout used by the training pipeline (see
    # rl_agent.env3_state_builder.StateBuilder._extract_path_features). If
    # the layout there changes, mirror the change below.
    _IMPORTANT_MODIFIERS = (
        'onlyOwner', 'onlyAdmin', 'onlyMinter',
        'whenNotPaused', 'nonReentrant', 'lock',
        'isHuman', 'onlyEOA', 'onlyWhitelisted',
    )
    _RISK_KEYWORDS = (
        ('weak entropy', 'Weak entropy'),
        ('gambling', 'gambling'),
        ('lottery', 'lottery'),
        ('financial', 'financial'),
        ('direct impact', 'Direct'),
        ('short path', 'Short path'),
        ('admin', 'Admin'),
    )
    _MITIGATION_KEYWORDS = (
        ('access control', 'access control'),
        ('admin only', 'Admin-only'),
        ('require check', 'require'),
        ('modifier', 'modifier'),
        ('validation', 'validation'),
    )

    def _extract_features(self, path: Dict) -> np.ndarray:
        """100-dim feature vector matching the layout used during training."""
        feats: List[float] = []

        agg = path.get('aggregate_features', {}) or {}
        basic = path.get('basic_info', {}) or {}

        # ---------- Section 1: main features (25) ----------
        feats.extend([
            agg.get('path_length_normalized', 0),
            agg.get('require_density', 0),
            agg.get('condition_density', 0),
            agg.get('keccak_density', 0),
            agg.get('mitigation_score', 0),
            agg.get('has_any_mitigation', 0),
            agg.get('has_strong_mitigation', 0),
            agg.get('has_modifier_protection', 0),
            agg.get('has_restricted_visibility', 0),
            agg.get('has_external_protection', 0),
            agg.get('function_require_density', 0),
            agg.get('unique_functions_ratio', 0),
            agg.get('node_diversity', 0),
            agg.get('distance_to_sink', 0),
            agg.get('distance_from_source', 0),
            agg.get('has_data_flow', 0),
            agg.get('contains_loop', 0),
        ])
        src_enc = agg.get('source_type_encoded', [0, 0, 0, 0])
        snk_enc = agg.get('sink_type_encoded', [0, 0, 0, 0])
        feats.extend(src_enc[:4] + [0] * max(0, 4 - len(src_enc)))
        feats.extend(snk_enc[:4] + [0] * max(0, 4 - len(snk_enc)))

        # ---------- Section 2: modifier context (15) ----------
        ctx = path.get('function_context', {}) or {}
        modifier_names = ctx.get('modifier_names', []) or []
        for mod in self._IMPORTANT_MODIFIERS:
            feats.append(1.0 if mod in modifier_names else 0.0)
        feats.append(min(len(modifier_names) / 3.0, 1.0))
        vis = ctx.get('function_visibility', '')
        feats.extend([
            1.0 if vis == 'public'   else 0.0,
            1.0 if vis == 'external' else 0.0,
            1.0 if vis == 'internal' else 0.0,
            1.0 if vis == 'private'  else 0.0,
            1.0 if vis == ''         else 0.0,
        ])

        # ---------- Section 3: sequence analysis (20) ----------
        graph_enrichment = path.get('graph_enrichment', {}) or {}
        nodes_detail = graph_enrichment.get('nodes_detail', []) or []
        n_total = max(len(nodes_detail), 1)

        req_pos, tr_pos, kk_pos = [], [], []
        for i, node in enumerate(nodes_detail):
            ntype = (node.get('type') or '').lower()
            code = (node.get('code_snippet') or '').lower()
            pos = i / n_total
            if ntype == 'require':
                req_pos.append(pos)
            elif ntype == 'transfer' or 'transfer' in code:
                tr_pos.append(pos)
            elif ntype == 'keccak' or 'keccak' in code:
                kk_pos.append(pos)

        feats.extend([
            min(req_pos) if req_pos else 1.0,
            max(req_pos) if req_pos else 0.0,
            min(tr_pos)  if tr_pos  else 1.0,
            max(tr_pos)  if tr_pos  else 0.0,
            1.0 if (req_pos and tr_pos and min(req_pos) < max(tr_pos)) else 0.0,
            1.0 if (req_pos and tr_pos and max(req_pos) < min(tr_pos)) else 0.0,
        ])

        funcs = str(graph_enrichment.get('functions_involved', '')).lower()
        feats.extend([
            1.0 if 'random'   in funcs else 0.0,
            1.0 if 'transfer' in funcs else 0.0,
            1.0 if 'withdraw' in funcs else 0.0,
            1.0 if 'mint'     in funcs else 0.0,
            1.0 if 'burn'     in funcs else 0.0,
            1.0 if ('bet' in funcs or 'gamble' in funcs) else 0.0,
        ])

        ntc = graph_enrichment.get('node_types_count', {}) or {}
        ntc_total = sum(ntc.values()) if ntc else 1
        feats.extend([
            ntc.get('require',    0) / ntc_total,
            ntc.get('assignment', 0) / ntc_total,
            ntc.get('condition',  0) / ntc_total,
            ntc.get('transfer',   0) / ntc_total,
            ntc.get('keccak',     0) / ntc_total,
            (-sum((c / ntc_total) * np.log(c / ntc_total + 1e-10)
                  for c in ntc.values()) if ntc else 0.0),
        ])

        if tr_pos and req_pos:
            dists = [abs(t - r) for t in tr_pos for r in req_pos]
            feats.extend([min(dists), max(dists)])
        else:
            feats.extend([0.0, 0.0])

        # ---------- Section 4: code-pattern analysis (20) ----------
        all_code = ' '.join((n.get('code_snippet') or '') for n in nodes_detail).lower()
        for kw in (
            'msg.sender', 'msg.value', 'tx.origin',
            'block.timestamp', 'block.number', 'block.difficulty',
            'blockhash', 'now', 'random',
            'transfer', 'send', 'call.value',
            'delegatecall', 'selfdestruct', 'suicide',
        ):
            feats.append(1.0 if kw in all_code else 0.0)
        feats.extend([
            1.0 if ('*' in all_code and 'safem' not in all_code) else 0.0,
            1.0 if ('+' in all_code and 'safem' not in all_code) else 0.0,
            1.0 if ('-' in all_code and 'safem' not in all_code) else 0.0,
            1.0 if '/' in all_code else 0.0,
            1.0 if '%' in all_code else 0.0,
        ])

        # ---------- Section 5: risk context (20) ----------
        risk_factors = basic.get('_risk_content', []) or []
        mitig_content = basic.get('_mitigation_content', []) or []
        for _, term in self._RISK_KEYWORDS:
            feats.append(1.0 if any(term in str(f) for f in risk_factors) else 0.0)
        for _, term in self._MITIGATION_KEYWORDS:
            feats.append(1.0 if any(term in str(m) for m in mitig_content) else 0.0)
        feats.extend([
            len(risk_factors) / 5.0,
            len(mitig_content) / 3.0,
            (len(risk_factors) - len(mitig_content)) / 5.0,
            basic.get('mitigation_count', 0) / 5.0,
            basic.get('risk_factors_count', 0) / 5.0,
            1.0 if basic.get('has_data_flow', False) else 0.0,
            1.0 if basic.get('contains_loop', False) else 0.0,
            basic.get('path_length', 0) / 20.0,
        ])

        vec = np.asarray(feats[:100], dtype=np.float32)
        if vec.size < 100:
            vec = np.pad(vec, (0, 100 - vec.size))
        return vec

    # ----------------------------------------------------- Function ranking

    def group_paths_by_function(self, path_db: Dict) -> Dict[str, List[Dict]]:
        grouped: Dict[str, List[Dict]] = defaultdict(list)
        for path in path_db.get('paths', []):
            func_ctx = path.get('function_context', {}) or {}
            func_name = func_ctx.get('primary_function', 'unknown')
            if func_name and func_name != 'unknown':
                grouped[func_name].append(path)
        return dict(grouped)

    def rank_functions(self, grouped_paths: Dict[str, List[Dict]],
                       top_k: int = 5) -> List[Tuple[str, float]]:
        """Rank candidate functions by aggregate Q(ANALYZE) signal."""
        scores: Dict[str, float] = {}

        for func_name, paths in grouped_paths.items():
            if not paths:
                continue

            state_matrix = np.stack([self._extract_features(p) for p in paths])
            state_t = torch.from_numpy(state_matrix).unsqueeze(0).to(self.device)

            with torch.no_grad():
                q_values = self.model(state_t)  # (1, num_paths, 2)
            q_values = q_values[0].cpu().numpy()

            q_analyze = q_values[:, 0]
            q_skip = q_values[:, 1]
            analyze_votes = int(np.sum(q_analyze > q_skip))
            scores[func_name] = float(q_analyze.sum()) + analyze_votes

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]

    # ---------------------------------------------------- Per-path signals

    def _per_path_signals(self, paths: List[Dict]):
        """
        Run a single batched forward+backward pass over all paths of a
        function. Returns:
            gradients      : np.ndarray (num_paths, 100), |dQ_analyze/df|
            attention      : np.ndarray (num_paths,), mean attention per path
            q_analyze      : np.ndarray (num_paths,)
        """
        features = np.stack([self._extract_features(p) for p in paths])
        state = torch.from_numpy(features).unsqueeze(0).to(self.device)
        state.requires_grad_(True)

        q_values, attn = self.model(state, return_attention=True)
        # q_values: (1, num_paths, 2);  attn: (1, num_paths, num_paths)
        q_analyze = q_values[0, :, 0]

        # Sum over paths -- gradient w.r.t. each path's own input is then
        # contained in the corresponding row of state.grad.
        q_analyze.sum().backward()
        grads = state.grad[0].detach().abs().cpu().numpy()   # (num_paths, 100)

        attn = attn[0].detach().cpu().numpy()                # (num_paths, num_paths)
        # The mean column tells us how much, on average, other paths attend
        # to path i -- a reasonable proxy for "how informative is this path".
        attention_per_path = attn.mean(axis=0)

        return grads, attention_per_path, q_analyze.detach().cpu().numpy()

    # ----------------------------------------------- Feature->node mapping

    # Approximate slices of the 100-dim feature vector that are most relevant
    # to particular node types. The indices are taken from the layout defined
    # in StateBuilder._extract_path_features.
    FEATURE_SLICE_FOR_NODE_TYPE = {
        'require':    (1, 11),     # require_density, function_require_density
        'condition':  (2, 30),     # condition_density, sequence stats
        'keccak':     (3, 75),     # keccak_density, code-pattern: keccak ops
        'transfer':   (60, 75),    # critical-ops region containing transfer
        'assignment': (40, 60),    # node-type distribution region
    }

    # Critical-op keywords -> approximate feature index inside Section 4.
    # The exact offset is the order in which they appear in
    # state_builder._extract_path_features (`critical_ops` list).
    CRITICAL_OP_KEYWORDS = [
        'msg.sender', 'msg.value', 'tx.origin',
        'block.timestamp', 'block.number', 'block.difficulty',
        'blockhash', 'now', 'random',
        'transfer', 'send', 'call.value',
        'delegatecall', 'selfdestruct', 'suicide',
    ]
    # Section 4 starts after sections 1+2+3 = 25 + 15 + 20 = 60.
    _SECTION4_START = 60

    def _gradient_to_nodes(self, gradients_per_path: np.ndarray,
                           paths: List[Dict]) -> Dict[str, float]:
        """
        Distribute per-feature gradients of each path to the actual nodes that
        make up that path. The mapping uses node type and code_snippet to
        pick the most relevant slice of the gradient vector.
        """
        node_scores: Dict[str, float] = defaultdict(float)

        for path_idx, path in enumerate(paths):
            grad = gradients_per_path[path_idx]
            graph_enrichment = path.get('graph_enrichment', {}) or {}
            nodes_detail = graph_enrichment.get('nodes_detail', []) or []
            basic_info = path.get('basic_info', {}) or {}
            source_node = basic_info.get('source_node', '')
            sink_node = basic_info.get('sink_node', '')

            for node in nodes_detail:
                node_id = node.get('node_id') or node.get('id') or ''
                if not node_id:
                    continue

                node_type = (node.get('type') or 'unknown').lower()
                code = (node.get('code_snippet') or '').lower()

                # 1) base contribution from the slice associated with the
                #    node's syntactic type.
                base = 0.0
                slc = self.FEATURE_SLICE_FOR_NODE_TYPE.get(node_type)
                if slc is not None:
                    lo, hi = slc
                    base = float(np.mean(grad[lo:hi]))

                # 2) extra contribution if the node's code mentions a known
                #    critical operation -- we look up the corresponding
                #    feature index inside Section 4.
                extra = 0.0
                for k, kw in enumerate(self.CRITICAL_OP_KEYWORDS):
                    if kw in code:
                        idx = self._SECTION4_START + k
                        if idx < len(grad):
                            extra += float(grad[idx])

                # 3) source/sink nodes always get the strongest weight.
                role_weight = NODE_TYPE_WEIGHT.get(node_type, NODE_TYPE_WEIGHT['unknown'])
                if node_id == source_node:
                    role_weight = max(role_weight, NODE_TYPE_WEIGHT['source'])
                if node_id == sink_node:
                    role_weight = max(role_weight, NODE_TYPE_WEIGHT['sink'])

                node_scores[node_id] += role_weight * (base + extra)

        return dict(node_scores)

    def _attention_to_nodes(self, attention_per_path: np.ndarray,
                            paths: List[Dict]) -> Dict[str, float]:
        """Spread each path's attention weight uniformly across its nodes."""
        node_scores: Dict[str, float] = defaultdict(float)
        for path_idx, path in enumerate(paths):
            graph_enrichment = path.get('graph_enrichment', {}) or {}
            nodes_detail = graph_enrichment.get('nodes_detail', []) or []
            if not nodes_detail:
                continue
            w = float(attention_per_path[path_idx]) / len(nodes_detail)
            for node in nodes_detail:
                node_id = node.get('node_id') or node.get('id') or ''
                if node_id:
                    node_scores[node_id] += w
        return dict(node_scores)

    # ------------------------------------------- Graph-based score channels

    @staticmethod
    def _parse_edges(graph_data: Dict) -> List[Tuple[str, str]]:
        """
        Semantic-graph exports use one of two edge formats:
            (a) list of dicts with 'source'/'target' or 'from'/'to' keys
            (b) list of two-element [src, dst] tuples
        We accept both.
        """
        edges: List[Tuple[str, str]] = []
        for e in graph_data.get('edges', []) or []:
            if isinstance(e, dict):
                src = e.get('source') or e.get('from') or e.get('src')
                dst = e.get('target') or e.get('to') or e.get('dst')
            elif isinstance(e, (list, tuple)) and len(e) >= 2:
                src, dst = e[0], e[1]
            else:
                continue
            if src is not None and dst is not None:
                edges.append((str(src), str(dst)))
        return edges

    def _build_subgraph(self, paths: List[Dict],
                        semantic_graph: Optional[Dict]) -> nx.DiGraph:
        """
        Build a directed subgraph containing exactly the nodes that appear on
        at least one of the function's paths, restricted to edges of the
        semantic graph that connect two such nodes.
        """
        G = nx.DiGraph()

        path_node_ids = set()
        for path in paths:
            for nid in path.get('basic_info', {}).get('path_nodes', []) or []:
                path_node_ids.add(str(nid))
            for nd in path.get('graph_enrichment', {}).get('nodes_detail', []) or []:
                nid = nd.get('node_id') or nd.get('id')
                if nid:
                    path_node_ids.add(str(nid))

        for nid in path_node_ids:
            G.add_node(nid)

        if semantic_graph is not None:
            for src, dst in self._parse_edges(semantic_graph):
                if src in path_node_ids and dst in path_node_ids:
                    G.add_edge(src, dst)

        # Add the path's own sequence as a fallback so that propagation still
        # works when the semantic graph does not expose explicit edges.
        for path in paths:
            seq = [str(n) for n in path.get('basic_info', {}).get('path_nodes', []) or []]
            for u, v in zip(seq[:-1], seq[1:]):
                if not G.has_edge(u, v):
                    G.add_edge(u, v)

        return G

    def _propagation_scores(self, G: nx.DiGraph,
                            gradient_scores: Dict[str, float],
                            paths: List[Dict]) -> Dict[str, float]:
        """
        Bidirectional propagation on the subgraph: starting from sources
        forward and from sinks backward, multiplied by `propagation_decay`
        at every hop. The initial value of each seed node is its gradient
        score.
        """
        if G.number_of_nodes() == 0:
            return {}

        sources = set()
        sinks = set()
        for path in paths:
            bi = path.get('basic_info', {}) or {}
            if bi.get('source_node'):
                sources.add(str(bi['source_node']))
            if bi.get('sink_node'):
                sinks.add(str(bi['sink_node']))

        prop: Dict[str, float] = defaultdict(float)

        def _bfs_propagate(seeds, graph):
            visited = {s: 0 for s in seeds if s in graph}
            frontier = list(visited.keys())
            while frontier:
                next_frontier = []
                for u in frontier:
                    seed_val = gradient_scores.get(u, 0.0)
                    decay = self.propagation_decay ** visited[u]
                    prop[u] += seed_val * decay
                    for v in graph.successors(u):
                        if v not in visited:
                            visited[v] = visited[u] + 1
                            next_frontier.append(v)
                frontier = next_frontier

        _bfs_propagate(sources, G)
        _bfs_propagate(sinks, G.reverse(copy=False))

        return dict(prop)

    @staticmethod
    def _centrality_scores(G: nx.DiGraph) -> Dict[str, float]:
        """Normalised (degree + betweenness)/2 centrality on the subgraph."""
        if G.number_of_nodes() == 0:
            return {}

        deg = nx.degree_centrality(G)
        try:
            btw = nx.betweenness_centrality(G)
        except Exception:
            btw = {n: 0.0 for n in G.nodes()}

        return {n: 0.5 * deg.get(n, 0.0) + 0.5 * btw.get(n, 0.0)
                for n in G.nodes()}

    # ------------------------------------------------------- Node ranking

    def localize_nodes(self, paths: List[Dict],
                       semantic_graph: Optional[Dict] = None,
                       top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Return the top-k most suspicious nodes among the paths of one
        function, each with its aggregated score.
        """
        if not paths:
            return []

        gradients, attention, _ = self._per_path_signals(paths)

        grad_scores = self._gradient_to_nodes(gradients, paths)
        attn_scores = self._attention_to_nodes(attention, paths)

        G = self._build_subgraph(paths, semantic_graph)
        prop_scores = self._propagation_scores(G, grad_scores, paths)
        cent_scores = self._centrality_scores(G)

        all_nodes = set(grad_scores) | set(attn_scores) | set(prop_scores) | set(cent_scores)
        final: Dict[str, float] = {}
        for nid in all_nodes:
            final[nid] = (self.alpha * grad_scores.get(nid, 0.0)
                          + self.beta * attn_scores.get(nid, 0.0)
                          + self.gamma * prop_scores.get(nid, 0.0)
                          + self.delta * cent_scores.get(nid, 0.0))

        ranked = sorted(final.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]

    # --------------------------------------------------- Caller detection

    @staticmethod
    def extract_call_graph(semantic_graph: Dict) -> Dict[str, List[str]]:
        """Build a caller->callee adjacency list from the semantic graph."""
        call_graph: Dict[str, List[str]] = defaultdict(list)

        func_nodes: Dict[str, List[Dict]] = defaultdict(list)
        for node in semantic_graph.get('nodes', []) or []:
            func = node.get('function', '')
            if func:
                func_nodes[func].append(node)

        for caller, nodes in func_nodes.items():
            for node in nodes:
                code = node.get('code_snippet', '') or ''
                if not code:
                    continue
                for callee in func_nodes.keys():
                    if callee == caller:
                        continue
                    if callee in call_graph[caller]:
                        continue
                    if re.search(rf'\b{re.escape(callee)}\s*\(', code):
                        call_graph[caller].append(callee)

        return dict(call_graph)

    @staticmethod
    def check_caller_relationship(call_graph: Dict[str, List[str]],
                                  predicted: str, ground_truth: str) -> bool:
        return ground_truth in call_graph.get(predicted, [])

    # -------------------------------------------------- High-level entry

    def localize(self, path_db: Dict, semantic_graph: Optional[Dict] = None,
                 top_k_functions: int = 5, top_k_nodes: int = 5) -> Dict:
        """End-to-end localization for one contract."""
        grouped = self.group_paths_by_function(path_db)
        if not grouped:
            return {'functions': [], 'nodes': {}, 'call_graph': {}}

        ranked_funcs = self.rank_functions(grouped, top_k_functions)

        node_results: Dict[str, List[Tuple[str, float]]] = {}
        for func_name, _ in ranked_funcs:
            paths_in_func = grouped.get(func_name, [])
            node_results[func_name] = self.localize_nodes(
                paths_in_func, semantic_graph, top_k=top_k_nodes
            )

        call_graph = self.extract_call_graph(semantic_graph) if semantic_graph else {}

        return {
            'functions': ranked_funcs,
            'nodes': node_results,
            'call_graph': call_graph,
        }


def compute_metrics(gt_nodes: List[str], pred_nodes: List[str],
                    k: int = 5) -> Dict[str, float]:
    """Precision/recall/F1 at top-k, treating nodes as an unordered set."""
    pred_k = pred_nodes[:k]
    if not gt_nodes:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    hits = len(set(pred_k) & set(gt_nodes))
    precision = hits / len(pred_k) if pred_k else 0.0
    recall = hits / len(gt_nodes) if gt_nodes else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {'precision': precision, 'recall': recall, 'f1': f1}
