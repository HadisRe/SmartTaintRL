import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from collections import deque
import random
from typing import Tuple, List, Dict
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec
from typing import Dict, List
# Import our environment components
from evn4 import BadRandomnessEnv
from env1 import ContractDataLoader
from env2 import PoolManager
from env3 import StateBuilder
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import numpy as np
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from typing import Dict, List, Set, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_q_values(agent, env, num_episodes=50):
    """تحلیل Q-values برای paths مختلف"""
    q_values_analysis = {
        'with_timestamp': [],
        'without_timestamp': [],
        'with_require': [],
        'without_require': []
    }

    for ep in range(num_episodes):
        state, info = env.reset()

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = agent.q_network(state_tensor).squeeze(0)  # (20, 2)

            # برای هر path در pool
            for i in range(min(10, len(env.pool_state.current_pool))):
                path = env.pool_state.current_pool[i]
                features = path.get('aggregate_features', {})

                # Q-values برای این path
                q_analyze = q_values[i, 0].item()
                q_skip = q_values[i, 1].item()

                # بررسی ویژگی‌ها
                source_type = path.get('basic_info', {}).get('source_type', '')
                require_density = features.get('require_density', 0)

                if 'timestamp' in source_type.lower():
                    q_values_analysis['with_timestamp'].append(q_analyze - q_skip)
                else:
                    q_values_analysis['without_timestamp'].append(q_analyze - q_skip)

                if require_density > 0.3:
                    q_values_analysis['with_require'].append(q_analyze - q_skip)
                else:
                    q_values_analysis['without_require'].append(q_analyze - q_skip)

    # رسم نمودار
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # نمودار timestamp
    axes[0].hist(q_values_analysis['with_timestamp'], alpha=0.5, label='With Timestamp', bins=20, color='red')
    axes[0].hist(q_values_analysis['without_timestamp'], alpha=0.5, label='Without Timestamp', bins=20, color='blue')
    axes[0].set_xlabel('Q(ANALYZE) - Q(SKIP)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Q-value Difference by Timestamp Presence')
    axes[0].legend()
    axes[0].axvline(x=0, color='black', linestyle='--', alpha=0.3)

    # نمودار require
    axes[1].hist(q_values_analysis['with_require'], alpha=0.5, label='High Require', bins=20, color='green')
    axes[1].hist(q_values_analysis['without_require'], alpha=0.5, label='Low Require', bins=20, color='orange')
    axes[1].set_xlabel('Q(ANALYZE) - Q(SKIP)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Q-value Difference by Require Density')
    axes[1].legend()
    axes[1].axvline(x=0, color='black', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig('q_values_analysis.png')
    plt.show()

    return q_values_analysis


def track_action_evolution(training_history):
    """تحلیل تغییر رفتار agent در طول آموزش"""

    # فرض: training_history شامل action_counts برای هر 10 episode است
    episodes_bins = []
    analyze_ratios = []

    # از داده‌هایی که در training loop ذخیره کردید
    for i in range(0, len(training_history), 10):
        batch = training_history[i:i + 10]
        total_analyze = sum(ep.get('analyze_count', 0) for ep in batch)
        total_skip = sum(ep.get('skip_count', 0) for ep in batch)

        if total_analyze + total_skip > 0:
            ratio = total_analyze / (total_analyze + total_skip)
            analyze_ratios.append(ratio)
            episodes_bins.append(i + 5)

    # رسم نمودار
    plt.figure(figsize=(10, 6))
    plt.plot(episodes_bins, analyze_ratios, marker='o', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Analyze Ratio')
    plt.title('Agent Behavior Evolution: From Exploration to Selective Analysis')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
    plt.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Target (10%)')
    plt.legend()
    plt.savefig('action_evolution.png')
    plt.show()


def analyze_performance_breakdown(metrics_collector):
    """تحلیل دقیق performance"""

    # داده‌ها از metrics_collector
    tp = metrics_collector.true_positives
    fp = metrics_collector.false_positives
    tn = metrics_collector.true_negatives
    fn = metrics_collector.false_negatives

    # محاسبات
    total = tp + fp + tn + fn
    actual_vulnerable = tp + fn
    actual_safe = tn + fp

    # Baseline: همه را safe فرض کن
    baseline_accuracy = actual_safe / total
    baseline_recall = 0

    # مدل شما
    model_accuracy = (tp + tn) / total
    model_recall = tp / max(actual_vulnerable, 1)
    model_precision = tp / max(tp + fp, 1)

    # نمودار مقایسه
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Performance comparison
    metrics = ['Accuracy', 'Recall', 'Precision']
    baseline_values = [baseline_accuracy, baseline_recall, 0]
    model_values = [model_accuracy, model_recall, model_precision]

    x = np.arange(len(metrics))
    width = 0.35

    axes[0].bar(x - width / 2, baseline_values, width, label='Baseline (All Safe)', color='gray')
    axes[0].bar(x + width / 2, model_values, width, label='Our Model', color='blue')
    axes[0].set_xlabel('Metrics')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Performance Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].legend()
    axes[0].set_ylim([0, 1])

    # Confusion Matrix
    cm = np.array([[tp, fn], [fp, tn]])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_title('Confusion Matrix')
    axes[1].set_xticklabels(['Vulnerable', 'Safe'])
    axes[1].set_yticklabels(['Vulnerable', 'Safe'])

    plt.tight_layout()
    plt.savefig('performance_analysis.png')
    plt.show()

    print(f"\nKey Insights:")
    print(f"- Baseline would achieve {baseline_accuracy:.1%} accuracy but miss ALL vulnerabilities")
    print(f"- Our model achieves {model_recall:.1%} recall despite 1:18 imbalance")
    print(f"- This proves active learning, not passive classification")


class DQN(nn.Module):
    """Path-aware DQN with attention mechanism for localization"""

    def __init__(self, path_features_dim: int = 100):
        super(DQN, self).__init__()

        self.path_features_dim = path_features_dim

        # Attention layer for localization
        self.attention_query = nn.Linear(path_features_dim, 64)
        self.attention_key = nn.Linear(path_features_dim, 64)
        self.attention_value = nn.Linear(path_features_dim, 64)
        self.attention_scale = 64 ** 0.5

        # Original path evaluator
        self.path_evaluator = nn.Sequential(
            nn.Linear(64, 128),  # Changed from path_features_dim to 64 (attention output)
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # 2 outputs: Q(ANALYZE), Q(SKIP)
        )

        logger.info(f"Path-aware DQN with attention initialized: {path_features_dim} features per path")

    def forward(self, state: torch.Tensor, return_attention: bool = False):
        """
        Input: state with shape (batch_size, num_paths, features_per_path)
        Output: Q-values with shape (batch_size, num_paths, 2)

        Args:
            state: Input state tensor
            return_attention: If True, returns (q_values, attention_weights)
        """
        batch_size, num_paths, features = state.shape

        state_reshaped = state.view(-1, features)

        Q = self.attention_query(state_reshaped)
        K = self.attention_key(state_reshaped)
        V = self.attention_value(state_reshaped)

        Q = Q.view(batch_size, num_paths, 64)
        K = K.view(batch_size, num_paths, 64)
        V = V.view(batch_size, num_paths, 64)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.attention_scale
        attention_weights = torch.softmax(attention_scores, dim=-1)

        attention_output = torch.matmul(attention_weights, V)
        attention_output_reshaped = attention_output.view(-1, 64)

        q_values = self.path_evaluator(attention_output_reshaped)
        q_values = q_values.view(batch_size, num_paths, 2)

        if return_attention:
            return q_values, attention_weights
        return q_values


class ReplayBuffer:
    """Experience Replay Buffer for DQN"""

    def __init__(self, capacity: int = 4000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        logger.info(f"ReplayBuffer initialized with capacity {capacity}")

    def push(self, state: np.ndarray, action: Tuple[int, str], reward: float,
             next_state: np.ndarray, done: bool, info: Dict = None):
        """Store experience با tuple action"""
        path_index, action_type = action  # تجزیه tuple
        experience = {
            'state': state,
            'path_index': path_index,  # به جای 'action'
            'action_type': 0 if action_type == 'ANALYZE' else 1,  # جدید
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'info': info or {}
        }
        self.buffer.append(experience)
    def sample(self, batch_size: int) -> Dict:
        """Sample batch of experiences"""
        batch = random.sample(self.buffer, batch_size)

        states = np.array([e['state'] for e in batch])
        path_indices = np.array([e['path_index'] for e in batch])  # جایگزین actions
        action_types = np.array([e['action_type'] for e in batch])
        rewards = np.array([e['reward'] for e in batch])
        next_states = np.array([e['next_state'] for e in batch])
        dones = np.array([e['done'] for e in batch])

        return {
            'states': states,
            'path_indices': path_indices,  # جدید
            'action_types': action_types,  # جدید
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
    def __len__(self):
        return len(self.buffer)

    def get_statistics(self) -> Dict:
        """Get buffer statistics for debugging"""
        if len(self.buffer) == 0:
            return {'size': 0}

        rewards = [e['reward'] for e in self.buffer]
        actions = [e['action'] for e in self.buffer]
        dones = [e['done'] for e in self.buffer]

        # Action distribution
        action_counts = {}
        for a in actions:
            action_type = 'ANALYZE' if a < 20 else 'SKIP'
            action_counts[action_type] = action_counts.get(action_type, 0) + 1

        return {
            'size': len(self.buffer),
            'capacity_used': len(self.buffer) / self.capacity,
            'reward_mean': np.mean(rewards),
            'reward_std': np.std(rewards),
            'reward_min': np.min(rewards),
            'reward_max': np.max(rewards),
            'done_ratio': np.mean(dones),
            'action_distribution': action_counts,
            'analyze_ratio': action_counts.get('ANALYZE', 0) / len(actions) if actions else 0
        }


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, alpha=0.6):
        super().__init__(capacity)
        self.priorities = np.zeros(capacity)
        self.alpha = alpha
        self.pos = 0

    def push(self, state: np.ndarray, action: Tuple[int, str], reward: float,
             next_state: np.ndarray, done: bool, info: Dict = None):
        """Store experience با tuple action"""
        path_index, action_type = action
        experience = {
            'state': state,
            'path_index': path_index,
            'action_type': 0 if action_type == 'ANALYZE' else 1,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'info': info or {}
        }
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Dict:
        """Sample batch با format جدید"""
        batch = random.sample(self.buffer, batch_size)

        states = np.array([e['state'] for e in batch])
        path_indices = np.array([e['path_index'] for e in batch])
        action_types = np.array([e['action_type'] for e in batch])
        rewards = np.array([e['reward'] for e in batch])
        next_states = np.array([e['next_state'] for e in batch])
        dones = np.array([e['done'] for e in batch])

        return {
            'states': states,
            'path_indices': path_indices,
            'action_types': action_types,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }



class EnhancedMetricsCollector:
    """Enhanced metrics collection with contract-level and advanced analysis"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics for new training session"""
        # Path-level metrics (existing)
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0

        # Importance tracking
        self.importance_when_analyze = []
        self.importance_when_skip = []

        # Contract-level tracking
        self.contract_results = defaultdict(lambda: {
            'paths_total': 0,
            'paths_analyzed': 0,
            'high_risk_paths': 0,
            'high_risk_detected': 0,
            'has_vulnerability': False,
            'vulnerability_detected': False
        })
        self.current_contract_id = None

        # Mitigation effectiveness tracking
        self.mitigation_decisions = {
            'high_mitigation_analyzed': 0,
            'high_mitigation_skipped': 0,
            'low_mitigation_analyzed': 0,
            'low_mitigation_skipped': 0
        }

        # Source-sink vulnerability tracking
        self.source_sink_performance = defaultdict(lambda: {
            'analyzed': 0, 'skipped': 0,
            'true_vulnerable': 0, 'false_vulnerable': 0
        })

        # Pattern discovery timeline
        self.pattern_discovery_timeline = []
        self.patterns_per_episode = []
        self.unique_patterns_cumulative = []

        # Learning stability metrics
        self.decision_consistency = []  # Track if decisions change for same pattern
        self.q_value_variance = []
        self.action_entropy = []  # Measure exploration vs exploitation

        # Complexity analysis
        self.performance_by_contract_size = defaultdict(list)

        # Risk distribution
        self.risk_distribution_analyzed = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'UNKNOWN': 0}
        self.risk_distribution_skipped = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'UNKNOWN': 0}

    def update(self, action_type: str, path_risk: str, importance: float,
               pattern: tuple, contract_label: str, path_info: Dict = None):
        """Update all metrics with new action"""

        # Path-level metrics (existing logic)
        if path_risk == 'HIGH':
            if action_type == 'ANALYZE':
                self.true_positives += 1
            else:
                self.false_negatives += 1
        elif path_risk == 'LOW':
            if action_type == 'SKIP':
                self.true_negatives += 1
            else:
                self.false_positives += 1

        # Importance tracking
        if action_type == 'ANALYZE':
            self.importance_when_analyze.append(importance)
            self.risk_distribution_analyzed[path_risk] += 1
        else:
            self.importance_when_skip.append(importance)
            self.risk_distribution_skipped[path_risk] += 1

        # Contract-level update
        if self.current_contract_id:
            contract = self.contract_results[self.current_contract_id]
            contract['paths_total'] += 1
            if action_type == 'ANALYZE':
                contract['paths_analyzed'] += 1
            if path_risk == 'HIGH':
                contract['high_risk_paths'] += 1
                contract['has_vulnerability'] = True
                if action_type == 'ANALYZE':
                    contract['high_risk_detected'] += 1
                    contract['vulnerability_detected'] = True

        # Mitigation tracking
        if path_info:
            mitigation_score = path_info.get('mitigation_score', 0)
            if mitigation_score > 0.5:  # High mitigation
                if action_type == 'ANALYZE':
                    self.mitigation_decisions['high_mitigation_analyzed'] += 1
                else:
                    self.mitigation_decisions['high_mitigation_skipped'] += 1
            else:  # Low mitigation
                if action_type == 'ANALYZE':
                    self.mitigation_decisions['low_mitigation_analyzed'] += 1
                else:
                    self.mitigation_decisions['low_mitigation_skipped'] += 1

        # Source-sink tracking
        if path_info:
            source = path_info.get('source_type', 'unknown')
            sink = path_info.get('sink_type', 'unknown')
            key = f"{source}_{sink}"

            if action_type == 'ANALYZE':
                self.source_sink_performance[key]['analyzed'] += 1
            else:
                self.source_sink_performance[key]['skipped'] += 1

            if path_risk == 'HIGH':
                self.source_sink_performance[key]['true_vulnerable'] += 1
            else:
                self.source_sink_performance[key]['false_vulnerable'] += 1

    def set_contract(self, contract_id: str, contract_size: int = None):
        """Set current contract being analyzed"""
        self.current_contract_id = contract_id
        if contract_size:
            self.contract_results[contract_id]['size'] = contract_size

    def update_pattern_discovery(self, episode: int, unique_patterns: Set, new_patterns: int):
        """Track pattern discovery progress"""
        self.patterns_per_episode.append(new_patterns)
        self.unique_patterns_cumulative.append(len(unique_patterns))
        self.pattern_discovery_timeline.append({
            'episode': episode,
            'total_unique': len(unique_patterns),
            'new_this_episode': new_patterns
        })

    def update_learning_stability(self, q_values: np.ndarray, action_probs: np.ndarray = None):
        """Track learning stability metrics"""
        if len(q_values) > 0:
            self.q_value_variance.append(np.var(q_values))

        if action_probs is not None and len(action_probs) > 0:
            # Calculate entropy: -sum(p * log(p))
            entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))
            self.action_entropy.append(entropy)

    def get_contract_level_metrics(self) -> Dict:
        """Calculate contract-level performance metrics"""
        total_contracts = len(self.contract_results)
        if total_contracts == 0:
            return {}

        vulnerable_contracts = sum(1 for c in self.contract_results.values()
                                   if c['has_vulnerability'])
        detected_vulnerable = sum(1 for c in self.contract_results.values()
                                  if c['vulnerability_detected'])

        # Contract-level confusion matrix
        contract_tp = sum(1 for c in self.contract_results.values()
                          if c['has_vulnerability'] and c['vulnerability_detected'])
        contract_fn = sum(1 for c in self.contract_results.values()
                          if c['has_vulnerability'] and not c['vulnerability_detected'])
        contract_fp = sum(1 for c in self.contract_results.values()
                          if not c['has_vulnerability'] and c['paths_analyzed'] > 0)
        contract_tn = sum(1 for c in self.contract_results.values()
                          if not c['has_vulnerability'] and c['paths_analyzed'] == 0)

        contract_precision = contract_tp / max(contract_tp + contract_fp, 1)
        contract_recall = contract_tp / max(contract_tp + contract_fn, 1)

        # Average pruning per contract
        pruning_ratios = []
        for contract in self.contract_results.values():
            if contract['paths_total'] > 0:
                pruning_ratio = 1 - (contract['paths_analyzed'] / contract['paths_total'])
                pruning_ratios.append(pruning_ratio)

        return {
            'total_contracts': total_contracts,
            'vulnerable_contracts': vulnerable_contracts,
            'detection_rate': detected_vulnerable / max(vulnerable_contracts, 1),
            'contract_precision': contract_precision,
            'contract_recall': contract_recall,
            'contract_f1': 2 * contract_precision * contract_recall /
                           max(contract_precision + contract_recall, 0.001),
            'avg_pruning_per_contract': np.mean(pruning_ratios) if pruning_ratios else 0,
            'pruning_std': np.std(pruning_ratios) if pruning_ratios else 0,
            'contract_confusion': {
                'tp': contract_tp, 'fp': contract_fp,
                'tn': contract_tn, 'fn': contract_fn
            }
        }

    def get_mitigation_effectiveness(self) -> Dict:
        """Analyze how well the agent handles mitigation"""
        total_high_mit = (self.mitigation_decisions['high_mitigation_analyzed'] +
                          self.mitigation_decisions['high_mitigation_skipped'])
        total_low_mit = (self.mitigation_decisions['low_mitigation_analyzed'] +
                         self.mitigation_decisions['low_mitigation_skipped'])

        if total_high_mit > 0:
            high_mit_skip_rate = self.mitigation_decisions['high_mitigation_skipped'] / total_high_mit
        else:
            high_mit_skip_rate = 0

        if total_low_mit > 0:
            low_mit_analyze_rate = self.mitigation_decisions['low_mitigation_analyzed'] / total_low_mit
        else:
            low_mit_analyze_rate = 0

        return {
            'high_mitigation_skip_rate': high_mit_skip_rate,
            'low_mitigation_analyze_rate': low_mit_analyze_rate,
            'mitigation_awareness_score': (high_mit_skip_rate + low_mit_analyze_rate) / 2,
            'total_high_mitigation_paths': total_high_mit,
            'total_low_mitigation_paths': total_low_mit
        }

    def get_source_sink_analysis(self) -> Dict:
        """Analyze performance by source-sink combinations"""
        analysis = {}
        for key, stats in self.source_sink_performance.items():
            total = stats['analyzed'] + stats['skipped']
            if total > 0:
                analysis[key] = {
                    'total_paths': total,
                    'analyze_rate': stats['analyzed'] / total,
                    'vulnerability_rate': stats['true_vulnerable'] / total,
                    'false_negative_rate': (stats['true_vulnerable'] -
                                            min(stats['analyzed'], stats['true_vulnerable'])) /
                                           max(stats['true_vulnerable'], 1)
                }
        return analysis

    def get_learning_stability(self) -> Dict:
        """Analyze learning stability over time"""
        if not self.q_value_variance:
            return {}

        recent_variance = np.mean(self.q_value_variance[-100:]) if len(self.q_value_variance) > 100 else np.mean(
            self.q_value_variance)
        early_variance = np.mean(self.q_value_variance[:100]) if len(self.q_value_variance) > 100 else np.mean(
            self.q_value_variance)

        recent_entropy = np.mean(self.action_entropy[-100:]) if len(self.action_entropy) > 100 else 0
        early_entropy = np.mean(self.action_entropy[:100]) if len(self.action_entropy) > 100 else 0

        return {
            'variance_reduction': (early_variance - recent_variance) / max(early_variance, 0.001),
            'current_variance': recent_variance,
            'entropy_reduction': (early_entropy - recent_entropy) / max(early_entropy, 0.001),
            'current_entropy': recent_entropy,
            'is_stable': recent_variance < early_variance * 0.5
        }

    def get_comprehensive_metrics(self) -> Dict:
        """Get all metrics in one call"""
        path_metrics = self.get_metrics()  # Existing path-level metrics
        contract_metrics = self.get_contract_level_metrics()
        mitigation_metrics = self.get_mitigation_effectiveness()
        stability_metrics = self.get_learning_stability()

        return {
            'path_level': path_metrics,
            'contract_level': contract_metrics,
            'mitigation': mitigation_metrics,
            'stability': stability_metrics
        }

    def get_metrics(self) -> Dict:
        """Original path-level metrics (backward compatibility)"""
        total_decisions = (self.true_positives + self.false_positives +
                           self.true_negatives + self.false_negatives)
        if total_decisions == 0:
            return {}

        precision = self.true_positives / max(self.true_positives + self.false_positives, 1)
        recall = self.true_positives / max(self.true_positives + self.false_negatives, 1)
        specificity = self.true_negatives / max(self.true_negatives + self.false_positives, 1)
        accuracy = (self.true_positives + self.true_negatives) / total_decisions
        f1 = 2 * precision * recall / max(precision + recall, 0.001)

        return {
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'accuracy': accuracy,
            'f1_score': f1,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'true_negatives': self.true_negatives,
            'false_negatives': self.false_negatives,
            'total_analyzed': sum(self.risk_distribution_analyzed.values()),
            'total_skipped': sum(self.risk_distribution_skipped.values()),
            'pruning_ratio': sum(self.risk_distribution_skipped.values()) /
                             max(sum(self.risk_distribution_analyzed.values()) +
                                 sum(self.risk_distribution_skipped.values()), 1)
        }

class DQNAgent:
    """Complete DQN Agent with training capability"""

    def __init__(self, learning_rate: float = 1e-5,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay: int = 500):
        # Networks
        self.q_network = DQN(path_features_dim=100)  # حذف state_dim, action_dim
        self.target_network = DQN(path_features_dim=100)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimistic initialization برای معماری جدید
        for layer in self.q_network.modules():
            if isinstance(layer, nn.Linear):
                if layer.out_features == 2:  # آخرین لایه که 2 output دارد
                    layer.bias.data[0] = 1.0  # ANALYZE bias
                    layer.bias.data[1] = 0.0  # SKIP bias
        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Parameters
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        # Tracking
        self.losses = []
        self.q_values_history = []
        self.last_importance_values = []

        logger.info(f"DQNAgent initialized with lr={learning_rate}, gamma={gamma}")

    def select_action(self, state: np.ndarray, valid_paths: int = None) -> Tuple[int, str]:
        """
        انتخاب بهترین path-action pair

        Args:
            state: structured state با shape (20, 100)
            valid_paths: تعداد paths واقعی در pool (بقیه padding هستند)

        Returns:
            tuple: (path_index, action_type) مثلاً (3, 'ANALYZE')
        """
        # Epsilon-greedy
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                        np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        # تعداد paths معتبر
        if valid_paths is None:
            # پیدا کردن paths که non-zero هستند
            valid_paths = np.sum(np.any(state != 0, axis=1))

        # Exploration
        if random.random() < eps_threshold:
            # انتخاب تصادفی یک path و action
            path_idx = random.randint(0, min(valid_paths - 1, 19))
            action_type = random.choice(['ANALYZE', 'SKIP'])
            return path_idx, action_type

        # Exploitation
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # (1, 20, 100)
            q_values = self.q_network(state_tensor).squeeze(0)  # (20, 2)

            # Mask invalid paths (padding)
            q_values_masked = q_values.clone()
            q_values_masked[valid_paths:] = -float('inf')

            # پیدا کردن بهترین path-action pair
            # q_values[:, 0] = Q values برای ANALYZE
            # q_values[:, 1] = Q values برای SKIP

            # Flatten کردن برای پیدا کردن max
            q_flat = q_values_masked.view(-1)  # (40,)
            best_idx = q_flat.argmax().item()

            # تبدیل index به path_index و action_type
            path_idx = best_idx // 2
            action_idx = best_idx % 2
            action_type = 'ANALYZE' if action_idx == 0 else 'SKIP'

        return path_idx, action_type

    def update(self, batch: Dict, batch_size: int = 32):
        """
        Update Q-network با structured states و tuple actions
        """
        states = torch.FloatTensor(batch['states'])  # (batch, 20, 100)
        # actions حالا tuple است: (path_index, action_type)
        path_indices = torch.LongTensor(batch['path_indices'])  # (batch,)
        action_types = torch.LongTensor(batch['action_types'])  # (batch,) - 0=ANALYZE, 1=SKIP
        rewards = torch.FloatTensor(batch['rewards']).unsqueeze(1)  # (batch, 1)
        next_states = torch.FloatTensor(batch['next_states'])  # (batch, 20, 100)
        dones = torch.FloatTensor(batch['dones']).unsqueeze(1)  # (batch, 1)

        # Current Q values
        current_q_all = self.q_network(states)  # (batch, 20, 2)
        # Select Q-value برای path و action انتخاب شده
        batch_indices = torch.arange(batch_size)
        current_q_values = current_q_all[batch_indices, path_indices, action_types].unsqueeze(1)

        # Next Q values with Double DQN
        with torch.no_grad():
            # انتخاب best action با main network
            next_q_all = self.q_network(next_states)  # (batch, 20, 2)
            # Flatten برای پیدا کردن best overall action
            next_q_flat = next_q_all.view(batch_size, -1)  # (batch, 40)
            best_next_actions = next_q_flat.max(1)[1]  # (batch,)

            # تبدیل به path_index و action_type
            best_path_indices = best_next_actions // 2
            best_action_types = best_next_actions % 2

            # Evaluate با target network
            target_q_all = self.target_network(next_states)  # (batch, 20, 2)
            next_q_values = target_q_all[batch_indices, best_path_indices, best_action_types].unsqueeze(1)

            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
            target_q_values = torch.clamp(target_q_values, -10, 10)

        # Loss
        loss = F.mse_loss(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 0.5)
        self.optimizer.step()

        # Track
        self.losses.append(loss.item())
        self.q_values_history.append(current_q_values.mean().item())

        return loss.item()

    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        logger.debug("Target network updated"),

    def load_model(self, filepath: str):
        """Load trained model from file"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model loaded from {filepath}")





class EnhancedMetricsVisualizer:
    """Advanced visualization for all metrics including contract-level and mitigation"""

    def __init__(self):
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")
        plt.rcParams.update({
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': 150,
            'savefig.dpi': 300,
        })

        # Storage for time series
        self.episode_rewards = []
        self.episode_lengths = []
        self.analyze_ratios = []
        self.losses = []
        self.q_values = []

        # Enhanced metrics storage
        self.contract_metrics_history = []
        self.mitigation_effectiveness_history = []
        self.pattern_discovery_rate = []
        self.stability_metrics = []

        # Path-level metrics
        self.precision_history = []
        self.recall_history = []
        self.f1_history = []
        self.pruning_history = []

        # Accuracy history (جدید)
        self.accuracy_history = []

        # Confusion matrix values for path-level (جدید)
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0

        # Pattern Bank metrics (جدید)
        self.pattern_bank_usage = []
        self.pattern_frequencies = {}
    def update_episode(self, episode_data: Dict):
        """Update metrics from episode data"""
        # Basic metrics
        if 'reward' in episode_data:
            self.episode_rewards.append(episode_data['reward'])
        if 'length' in episode_data:
            self.episode_lengths.append(episode_data['length'])
        if 'analyze_ratio' in episode_data:
            self.analyze_ratios.append(episode_data['analyze_ratio'])

        # Loss and Q-values - flatten if needed
        if 'losses' in episode_data:
            losses = episode_data['losses']
            if isinstance(losses, list):
                self.losses.extend(losses)

        if 'q_values' in episode_data:
            q_vals = episode_data['q_values']
            if isinstance(q_vals, list):
                # Flatten if it's a list of arrays
                for q in q_vals:
                    if isinstance(q, np.ndarray):
                        self.q_values.extend(q.flatten().tolist())
                    else:
                        self.q_values.append(q)

        # Enhanced metrics
        if 'contract_metrics' in episode_data:
            self.contract_metrics_history.append(episode_data['contract_metrics'])
        if 'mitigation_effectiveness' in episode_data:
            self.mitigation_effectiveness_history.append(episode_data['mitigation_effectiveness'])

        # Pattern discovery - handle single value
        if 'pattern_discovery' in episode_data:
            new_patterns = episode_data['pattern_discovery']
            # Ensure it's non-negative
            new_patterns = max(0, new_patterns)
            self.pattern_discovery_rate.append(new_patterns)

        # Pattern Bank usage (جدید برای نمودار Pattern Bank)
        if 'pattern_bank_hit_rate' in episode_data:
            if not hasattr(self, 'pattern_bank_usage'):
                self.pattern_bank_usage = []
            self.pattern_bank_usage.append(episode_data['pattern_bank_hit_rate'] * 100)

        # Pattern frequencies (جدید برای heatmap)
        if 'pattern_frequencies' in episode_data:
            self.pattern_frequencies = episode_data['pattern_frequencies']

        # Path-level performance
        if 'precision' in episode_data:
            self.precision_history.append(episode_data['precision'])
        if 'recall' in episode_data:
            self.recall_history.append(episode_data['recall'])
        if 'f1' in episode_data:
            self.f1_history.append(episode_data['f1'])
        if 'pruning_ratio' in episode_data:
            self.pruning_history.append(episode_data['pruning_ratio'])

        # Accuracy - اصلاح شده
        if 'accuracy' in episode_data:
            if not hasattr(self, 'accuracy_history'):
                self.accuracy_history = []
            self.accuracy_history.append(episode_data['accuracy'])

        # Confusion matrix values - اصلاح شده
        if 'confusion' in episode_data:
            self.true_positives = episode_data['confusion'].get('tp', 0)
            self.false_positives = episode_data['confusion'].get('fp', 0)
            self.true_negatives = episode_data['confusion'].get('tn', 0)
            self.false_negatives = episode_data['confusion'].get('fn', 0)

        # Stability metrics - calculate from q_values if available
        if len(self.q_values) > 100:
            recent_variance = np.var(self.q_values[-100:])
            self.stability_metrics.append(recent_variance)

    def plot_comprehensive_analysis(self, save_path='comprehensive_analysis.png'):
        """Create comprehensive visualization with all metrics"""

        fig = plt.figure(figsize=(24, 16))
        gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3)

        # ==== Row 1: Core Performance ====

        # 1. Episode Rewards with correct visualization
        ax1 = fig.add_subplot(gs[0, 0])
        if self.episode_rewards:
            episodes = range(1, len(self.episode_rewards) + 1)

            # نمایش rewards اصلی با alpha کم
            ax1.plot(episodes, self.episode_rewards, alpha=0.3, color='lightblue', linewidth=0.8,
                     label='Episode Rewards')

            # محاسبه moving average
            if len(self.episode_rewards) > 10:
                window = min(10, len(self.episode_rewards) // 5)
                moving_avg = np.convolve(self.episode_rewards, np.ones(window) / window, mode='valid')
                ma_episodes = range(window, len(self.episode_rewards) + 1)
                ax1.plot(ma_episodes, moving_avg, color='darkred', linewidth=2.5,
                         label=f'MA({window})')

            # نمایش میانگین کل
            total_mean = np.mean(self.episode_rewards)
            ax1.axhline(y=total_mean, color='green', linestyle='--',
                        label=f'Mean: {total_mean:.2f}')

            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Episode Reward')
            ax1.set_title(f'Reward Evolution (Final Avg: {total_mean:.2f})')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='best')

            # تنظیم محدوده y
            y_min = np.percentile(self.episode_rewards, 5)
            y_max = np.percentile(self.episode_rewards, 95)
            margin = (y_max - y_min) * 0.1
            ax1.set_ylim([y_min - margin, y_max + margin])

        # 2. Loss Evolution (Normalized)
        ax2 = fig.add_subplot(gs[0, 1])
        if self.losses and len(self.losses) > 100:
            window = min(200, len(self.losses) // 10)
            smoothed_loss = np.convolve(self.losses, np.ones(window) / window, mode='valid')

            if len(smoothed_loss) > 0 and smoothed_loss.max() > smoothed_loss.min():
                normalized_loss = (smoothed_loss - smoothed_loss.min()) / (
                        smoothed_loss.max() - smoothed_loss.min() + 1e-8)

                x_axis = np.linspace(0, len(self.episode_rewards), len(normalized_loss))
                ax2.plot(x_axis, normalized_loss, color='darkblue', linewidth=2.5)
                ax2.fill_between(x_axis, 0, normalized_loss, alpha=0.3, color='lightblue')

                z = np.polyfit(x_axis, normalized_loss, 1)
                p = np.poly1d(z)
                ax2.plot(x_axis, p(x_axis), "r--", alpha=0.5, label='Trend')

                ax2.set_xlabel('Episode')
                ax2.set_ylabel('Normalized Loss (0=Best)')
                ax2.set_title(f'Training Progress (↓{(1 - normalized_loss[-1]) * 100:.1f}%)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

        # 3. Path vs Contract Performance - Bar Chart با Accuracy کامل
        ax3 = fig.add_subplot(gs[0, 2])
        if self.recall_history and self.contract_metrics_history:
            metrics_names = ['Precision', 'Recall', 'F1-Score', 'Accuracy']

            # Path-level values
            path_accuracy = 0
            if hasattr(self, 'accuracy_history') and self.accuracy_history:
                path_accuracy = self.accuracy_history[-1]

            path_values = [
                self.precision_history[-1] if self.precision_history else 0,
                self.recall_history[-1] if self.recall_history else 0,
                self.f1_history[-1] if self.f1_history else 0,
                path_accuracy
            ]

            # Contract-level values با محاسبه accuracy
            last_contract = self.contract_metrics_history[-1]
            contract_confusion = last_contract.get('contract_confusion', {})

            # محاسبه contract accuracy از confusion matrix
            contract_tp = contract_confusion.get('tp', 0)
            contract_tn = contract_confusion.get('tn', 0)
            contract_fp = contract_confusion.get('fp', 0)
            contract_fn = contract_confusion.get('fn', 0)
            contract_total = contract_tp + contract_tn + contract_fp + contract_fn
            contract_accuracy = (contract_tp + contract_tn) / contract_total if contract_total > 0 else 0

            contract_values = [
                last_contract.get('contract_precision', 0),
                last_contract.get('contract_recall', 0),
                last_contract.get('contract_f1', 0),
                contract_accuracy
            ]

            x = np.arange(len(metrics_names))
            width = 0.35

            bars1 = ax3.bar(x - width / 2, path_values, width, label='Path-Level', color='steelblue')
            bars2 = ax3.bar(x + width / 2, contract_values, width, label='Contract-Level', color='darkgreen')

            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width() / 2., height,
                             f'{height:.2%}',
                             ha='center', va='bottom', fontsize=9)

            ax3.set_ylabel('Score')
            ax3.set_title('Detection Performance Comparison')
            ax3.set_xticks(x)
            ax3.set_xticklabels(metrics_names)
            ax3.legend()
            ax3.set_ylim([0, 1.15])
            ax3.grid(True, alpha=0.3, axis='y')

        # 4. Pruning Evolution
        ax4 = fig.add_subplot(gs[0, 3])
        if self.pruning_history:
            episodes = range(1, len(self.pruning_history) + 1)
            ax4.plot(episodes, np.array(self.pruning_history) * 100,
                     color='purple', linewidth=2)
            ax4.fill_between(episodes, 0, np.array(self.pruning_history) * 100,
                             alpha=0.3, color='purple')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Pruning Rate (%)')
            ax4.set_title('Path Pruning Evolution')
            ax4.set_ylim([0, 50])
            ax4.grid(True, alpha=0.3)

        # ==== Row 2: Advanced Analysis ====

        # 5. Mitigation Effectiveness
        ax5 = fig.add_subplot(gs[1, 0])
        if self.mitigation_effectiveness_history and len(self.mitigation_effectiveness_history) > 0:
            high_mit_skip = [m.get('high_mitigation_skip_rate', 0)
                             for m in self.mitigation_effectiveness_history]
            low_mit_analyze = [m.get('low_mitigation_analyze_rate', 0)
                               for m in self.mitigation_effectiveness_history]

            if high_mit_skip and low_mit_analyze:
                episodes = range(1, len(high_mit_skip) + 1)
                ax5.plot(episodes, high_mit_skip, label='High Mitigation → Skip Rate',
                         color='green', linewidth=2)
                ax5.plot(episodes, low_mit_analyze, label='Low Mitigation → Analyze Rate',
                         color='red', linewidth=2)

                ax5.set_xlabel('Episode')
                ax5.set_ylabel('Rate')
                ax5.set_title('Mitigation Awareness')
                ax5.set_ylim([0, 1])
                ax5.legend()
                ax5.grid(True, alpha=0.3)

        # 6. Pattern Discovery Progress
        ax6 = fig.add_subplot(gs[1, 1])
        if self.pattern_discovery_rate and len(self.pattern_discovery_rate) > 0:
            episodes = range(1, len(self.pattern_discovery_rate) + 1)

            # Bar chart for new patterns per episode
            ax6.bar(episodes, self.pattern_discovery_rate, color='orange', alpha=0.7, width=0.8)

            # Cumulative patterns on secondary axis
            ax6_twin = ax6.twinx()
            cumulative = np.cumsum(self.pattern_discovery_rate)
            ax6_twin.plot(episodes, cumulative, color='darkred', linewidth=2,
                          label='Cumulative Patterns')

            ax6.set_xlabel('Episode')
            ax6.set_ylabel('New Patterns per Episode', color='orange')
            ax6_twin.set_ylabel('Total Unique Patterns', color='darkred')
            ax6.set_title('Pattern Discovery Progress')
            ax6_twin.legend()
            ax6.grid(True, alpha=0.3)

        # 7. F1 Score Evolution
        ax7 = fig.add_subplot(gs[1, 2])
        if self.f1_history:
            episodes = range(1, len(self.f1_history) + 1)
            ax7.plot(episodes, self.f1_history, color='darkgreen', linewidth=2)
            ax7.fill_between(episodes, 0.5, self.f1_history,
                             where=np.array(self.f1_history) > 0.5,
                             alpha=0.3, color='lightgreen',
                             interpolate=True)

            best_f1 = max(self.f1_history)
            best_episode = self.f1_history.index(best_f1) + 1
            ax7.scatter(best_episode, best_f1, color='red', s=100, zorder=5)
            ax7.annotate(f'Best: {best_f1:.3f}',
                         xy=(best_episode, best_f1),
                         xytext=(10, 10), textcoords='offset points',
                         bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

            ax7.set_xlabel('Episode')
            ax7.set_ylabel('F1 Score')
            ax7.set_title('F1 Score Evolution')
            ax7.set_ylim([0.5, 1.0])
            ax7.grid(True, alpha=0.3)

        # 8. Q-Value Variance (Learning Stability)
        ax8 = fig.add_subplot(gs[1, 3])
        if self.q_values and len(self.q_values) > 100:
            window_size = 100
            q_variance = []
            for i in range(0, len(self.q_values) - window_size, window_size):
                window = self.q_values[i:i + window_size]
                q_variance.append(np.var(window))

            if q_variance:
                x_axis = np.linspace(1, len(self.episode_rewards), len(q_variance))
                ax8.plot(x_axis, q_variance, color='darkblue', linewidth=2)
                ax8.fill_between(x_axis, 0, q_variance, alpha=0.3, color='lightblue')
                ax8.set_xlabel('Training Progress')
                ax8.set_ylabel('Q-value Variance')
                ax8.set_title('Learning Stability (Lower = More Stable)')
                ax8.grid(True, alpha=0.3)

        # ==== Row 3: Confusion Matrices and Analysis ====

        # 9. Contract-level Confusion Matrix
        ax9 = fig.add_subplot(gs[2, 0])
        if self.contract_metrics_history and len(self.contract_metrics_history) > 0:
            last_metrics = self.contract_metrics_history[-1]
            if 'contract_confusion' in last_metrics:
                conf = last_metrics['contract_confusion']
                cm = np.array([[conf['tp'], conf['fn']],
                               [conf['fp'], conf['tn']]])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax9,
                            xticklabels=['Detected', 'Missed'],
                            yticklabels=['Vulnerable', 'Safe'])
                ax9.set_title('Contract-Level Detection')

        # 10. Path-level Confusion Matrix
        ax10 = fig.add_subplot(gs[2, 1])
        if hasattr(self, 'true_positives'):
            path_cm = np.array([[self.true_positives, self.false_negatives],
                                [self.false_positives, self.true_negatives]])
            sns.heatmap(path_cm, annot=True, fmt='d', cmap='YlOrRd', ax=ax10,
                        xticklabels=['Analyzed', 'Skipped'],
                        yticklabels=['High Risk', 'Low Risk'])
            ax10.set_title('Path-Level Actions')
            ax10.set_xlabel('Action Taken')
            ax10.set_ylabel('True Risk Level')

        # 11. Pruning Distribution
        ax11 = fig.add_subplot(gs[2, 2])
        if self.pruning_history:
            ax11.hist(np.array(self.pruning_history) * 100, bins=20,
                      color='purple', alpha=0.7, edgecolor='black')
            ax11.axvline(x=np.mean(self.pruning_history) * 100,
                         color='red', linestyle='--', linewidth=2,
                         label=f'Mean: {np.mean(self.pruning_history) * 100:.1f}%')
            ax11.set_xlabel('Pruning Rate (%)')
            ax11.set_ylabel('Frequency')
            ax11.set_title('Pruning Rate Distribution')
            ax11.legend()
            ax11.grid(True, alpha=0.3, axis='y')

        # 12. Recall Stability Box Plot
        ax12 = fig.add_subplot(gs[2, 3])
        if len(self.recall_history) >= 40:
            quarter = len(self.recall_history) // 4
            quarters_data = {
                'Q1': self.recall_history[:quarter],
                'Q2': self.recall_history[quarter:2 * quarter],
                'Q3': self.recall_history[2 * quarter:3 * quarter],
                'Q4': self.recall_history[3 * quarter:]
            }

            quarters_data = {k: v for k, v in quarters_data.items() if v}

            if quarters_data:
                bp = ax12.boxplot(quarters_data.values(),
                                  tick_labels=list(quarters_data.keys()),
                                  patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                    patch.set_alpha(0.7)
                ax12.set_xlabel('Training Quarter')
                ax12.set_ylabel('Recall')
                ax12.set_title('Recall Stability Over Time')
                ax12.grid(True, alpha=0.3, axis='y')

        # ==== Row 4: خالی - بدون KPI Dashboard ====
        # هیچ چیز نمایش داده نمی‌شود

        plt.suptitle('RL-Based Bad Randomness Detection: Comprehensive Training Analysis',
                     fontsize=16, fontweight='bold', y=0.995)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=150)
        plt.show()

        print(f"\n📊 Comprehensive analysis saved to {save_path}")


    def _generate_comprehensive_summary(self):
        """Generate detailed summary statistics"""
        lines = []

        # Basic performance
        if self.recall_history:
            lines.append(f"Path-Level Performance:")
            lines.append(f"  • Best Recall: {max(self.recall_history):.2%}")
            lines.append(f"  • Final Recall: {self.recall_history[-1]:.2%}")

        if self.f1_history:
            lines.append(f"  • Best F1-Score: {max(self.f1_history):.3f}")
            lines.append(f"  • Final F1-Score: {self.f1_history[-1]:.3f}")

        # Contract-level performance
        if self.contract_metrics_history:
            last_contract = self.contract_metrics_history[-1]
            lines.append(f"\nContract-Level Performance:")
            lines.append(f"  • Contract Detection Rate: {last_contract.get('detection_rate', 0):.2%}")
            lines.append(f"  • Contract F1-Score: {last_contract.get('contract_f1', 0):.3f}")

        # Pruning
        if self.pruning_history:
            lines.append(f"\nPruning Performance:")
            lines.append(f"  • Average Pruning: {np.mean(self.pruning_history) * 100:.1f}%")
            lines.append(f"  • Final Pruning: {self.pruning_history[-1] * 100:.1f}%")

        # Mitigation
        if self.mitigation_effectiveness_history:
            last_mit = self.mitigation_effectiveness_history[-1]
            lines.append(f"\nMitigation Awareness:")
            lines.append(f"  • High Mitigation Skip Rate: {last_mit.get('high_mitigation_skip_rate', 0):.2%}")
            lines.append(f"  • Mitigation Awareness Score: {last_mit.get('mitigation_awareness_score', 0):.2f}")

        # Learning
        if self.losses:
            initial_loss = np.mean(self.losses[:100]) if len(self.losses) > 100 else self.losses[0]
            final_loss = np.mean(self.losses[-100:]) if len(self.losses) > 100 else self.losses[-1]
            lines.append(f"\nLearning Progress:")
            lines.append(f"  • Loss Reduction: {(1 - final_loss / initial_loss) * 100:.1f}%")

        if self.episode_rewards:
            lines.append(f"  • Total Episodes: {len(self.episode_rewards)}")
            lines.append(f"  • Average Reward: {np.mean(self.episode_rewards):.2f}")

        return '\n'.join(lines)

def get_pattern_bank_stats(env):
    """استخراج آمار pattern bank"""
    stats = {
        'unique_patterns': len(env.discovered_patterns),
        'total_patterns_seen': sum(env.pattern_bank.values()),
        'most_common_pattern': max(env.pattern_bank.items(), key=lambda x: x[1]) if env.pattern_bank else None,
        'pattern_distribution': env.pattern_bank
    }
    return stats


def test_training_loop():
    """Test training loop with comprehensive metrics and analysis"""
    print("\n" + "=" * 60)
    print("Testing Training Loop with New Architecture")
    print("=" * 60)

    # Initialize environment with explicit paths
    data_loader = ContractDataLoader(
        path_db_dir=r"C:\Users\Hadis\Documents\NewModel1\path_databases_updated",
        profile_dir=r"C:\Users\Hadis\Documents\NewModel1\contract_profiles"
    )

    # Verify dataset
    stats = data_loader.get_statistics()
    print(f"\n📊 Training with dataset:")
    print(f"   Total: {stats['total_contracts']} contracts")
    print(f"   Safe: {stats['safe_contracts']}")
    print(f"   Vulnerable: {stats['vulnerable_contracts']}")

    env = BadRandomnessEnv(
        data_loader=data_loader,
        pool_manager=PoolManager(),
        state_builder=StateBuilder(),
        debug_mode=False
    )

    # Create agent
    agent = DQNAgent(
        learning_rate=1e-5,
        gamma=0.95,
        epsilon_decay=500,
        epsilon_end=0.05
    )

    # Use PrioritizedReplayBuffer
    buffer = PrioritizedReplayBuffer(capacity=5000)

    # Initialize metrics
    try:
        metrics_collector = EnhancedMetricsCollector()
        visualizer = EnhancedMetricsVisualizer()
        use_enhanced = True
    except:
        metrics_collector = MetricsCollector()
        visualizer = MetricsVisualizer()
        use_enhanced = False

    # Initialize localizer (once at the beginning)
    from localizer import AttentionBasedLocalizer
    localizer = AttentionBasedLocalizer()
    localization_results = []

    # Training tracking variables
    episode_rewards = []
    episode_lengths = []
    importance_analyzed = []
    patterns_discovered = []
    action_counts = {'ANALYZE': 0, 'SKIP': 0}
    episode_action_ratios = []
    previous_patterns_count = 0

    # New tracking for analysis
    action_history_detailed = []
    q_values_per_episode = []
    importance_history = {'analyzed': [], 'skipped': []}

    # Run training episodes
    for episode in range(4500):
        state, info = env.reset()

        # Set contract for enhanced metrics
        if use_enhanced and hasattr(metrics_collector, 'set_contract'):
            metrics_collector.set_contract(
                info.get('contract_address', 'unknown'),
                info.get('pool_size', 0)
            )

        # Episode tracking
        episode_reward = 0
        episode_length = 0
        importances = []
        analyze_ratio = 0
        episode_q_values = []
        episode_actions = {'analyze_count': 0, 'skip_count': 0}
        episode_importance = {'analyzed': [], 'skipped': []}

        done = False
        while not done and episode_length < 50:
            # Get valid paths count
            valid_paths = info.get('valid_paths', info.get('pool_size', 10))

            # Select action
            action = agent.select_action(state, valid_paths=valid_paths)

            # Track Q-values for analysis
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = agent.q_network(state_tensor, return_attention=False)
                q_values_np = q_values.numpy()
                episode_q_values.append(q_values_np.flatten())

            # Execute action
            next_state, reward, done, info = env.step(action)

            # Extract components
            path_index, action_type = action

            # LOCALIZATION: Perform localization for analyzed paths
            if action_type == 'ANALYZE' and path_index < len(env.pool_state.current_pool):
                vulnerable_path = env.pool_state.current_pool[path_index]

                # Build state with metadata for localization
                single_path_state, single_path_metadata, single_path_mapping = env.state_builder.build_state_with_mapping(
                    env.current_contract,
                    env.pool_state
                )

                # Get metadata for the analyzed path
                path_metadata = single_path_metadata[path_index] if path_index < len(single_path_metadata) else None

                if path_metadata is not None:
                    # Perform localization
                    try:
                        localization_result = localizer.localize(
                            state=single_path_state[path_index],
                            path_metadata=single_path_metadata[path_index],
                            feature_mapping=single_path_mapping[path_index],
                            model=agent.q_network,
                            device='cpu'
                        )

                        # Store result
                        localization_results.append(localization_result)

                        # Print detailed report every 500 episodes for first analyzed path
                        if episode % 500 == 0 and episode_actions['analyze_count'] == 0:
                            report = localizer.format_report(localization_result)
                            print("\n" + report)

                    except Exception as e:
                        logger.warning(f"Localization failed for episode {episode}: {e}")

            # Track actions
            action_counts[action_type] += 1
            episode_actions[f'{action_type.lower()}_count'] += 1

            # Track importance
            if 'path_importance' in info:
                imp = info['path_importance']
                if action_type == 'ANALYZE':
                    episode_importance['analyzed'].append(imp)
                    importance_history['analyzed'].append(imp)
                else:
                    episode_importance['skipped'].append(imp)
                    importance_history['skipped'].append(imp)

            # Get path information for metrics
            path = None
            path_info = None

            if path_index < len(env.pool_state.current_pool):
                path = env.pool_state.current_pool[path_index]
                pattern = env._extract_pattern_signature(path)
                contract_label = info.get('contract_label', 'unknown')

                if use_enhanced:
                    path_info = {
                        'mitigation_score': path.get('aggregate_features', {}).get('mitigation_score', 0),
                        'source_type': path.get('basic_info', {}).get('source_type', 'unknown'),
                        'sink_type': path.get('basic_info', {}).get('sink_type', 'unknown')
                    }

                # Update metrics
                if use_enhanced and path_info:
                    metrics_collector.update(
                        action_type,
                        info.get('path_risk', 'UNKNOWN'),
                        info.get('path_importance', 0),
                        pattern,
                        contract_label,
                        path_info=path_info
                    )
                else:
                    metrics_collector.update(
                        action_type,
                        info.get('path_risk', 'UNKNOWN'),
                        info.get('path_importance', 0),
                        pattern,
                        contract_label
                    )

            # Clip reward
            reward = np.clip(reward, -10, 10)

            # Store experience
            buffer.push(state, action, reward, next_state, done, info)

            # Track for episode summary
            if action_type == 'ANALYZE' and 'path_importance' in info:
                importances.append(info.get('path_importance', 0))

            # Update network if enough experiences
            if len(buffer) >= 32:
                batch = buffer.sample(32)
                loss = agent.update(batch)

                # Update target network periodically
                if agent.steps_done % 50 == 0:
                    agent.update_target_network()

            state = next_state
            episode_reward += reward
            episode_length += 1

        # Episode summary
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        importance_analyzed.append(np.mean(importances) if importances else 0)
        action_history_detailed.append(episode_actions)

        # Track Q-values
        if episode_q_values:
            q_values_per_episode.append(np.concatenate(episode_q_values))

        # Pattern discovery
        current_patterns_count = len(env.discovered_patterns)
        new_patterns_this_episode = current_patterns_count - previous_patterns_count
        patterns_discovered.append(current_patterns_count)
        previous_patterns_count = current_patterns_count

        # Action ratio
        total_actions = action_counts['ANALYZE'] + action_counts['SKIP']
        analyze_ratio = action_counts['ANALYZE'] / total_actions if total_actions > 0 else 0
        episode_action_ratios.append(analyze_ratio)

        # Update stability metrics
        if use_enhanced and episode_q_values:
            all_q_values = np.concatenate(episode_q_values)
            metrics_collector.update_learning_stability(all_q_values)

        # Update pattern discovery
        if use_enhanced:
            metrics_collector.update_pattern_discovery(
                episode,
                env.discovered_patterns,
                new_patterns_this_episode
            )

        # Update visualizer every 10 episodes
        if episode % 10 == 0:
            if use_enhanced:
                comprehensive_metrics = metrics_collector.get_comprehensive_metrics()

                pattern_bank_hit_rate = 0
                if hasattr(env, 'pattern_bank_hits') and hasattr(env, 'total_pattern_checks'):
                    if env.total_pattern_checks > 0:
                        pattern_bank_hit_rate = env.pattern_bank_hits / env.total_pattern_checks

                pattern_frequencies = {}
                if hasattr(env, 'pattern_bank'):
                    pattern_frequencies = dict(env.pattern_bank)

                episode_data = {
                    'reward': episode_reward,
                    'length': episode_length,
                    'analyze_ratio': analyze_ratio,
                    'importance_analyzed': importances,
                    'importance_skipped': [],
                    'losses': agent.losses[-episode_length:] if agent.losses else [],
                    'q_values': agent.q_values_history[-episode_length:] if agent.q_values_history else [],
                    'precision': comprehensive_metrics['path_level']['precision'],
                    'recall': comprehensive_metrics['path_level']['recall'],
                    'accuracy': comprehensive_metrics['path_level']['accuracy'],
                    'f1': comprehensive_metrics['path_level']['f1_score'],
                    'pruning_ratio': comprehensive_metrics['path_level']['pruning_ratio'],
                    'contract_metrics': comprehensive_metrics['contract_level'],
                    'mitigation_effectiveness': comprehensive_metrics['mitigation'],
                    'pattern_discovery': new_patterns_this_episode,
                    'pattern_bank_hit_rate': pattern_bank_hit_rate,
                    'pattern_frequencies': pattern_frequencies,
                    'confusion': {
                        'tp': metrics_collector.true_positives,
                        'fp': metrics_collector.false_positives,
                        'tn': metrics_collector.true_negatives,
                        'fn': metrics_collector.false_negatives
                    }
                }
            else:
                current_metrics = metrics_collector.get_metrics()
                episode_data = {
                    'reward': episode_reward,
                    'length': episode_length,
                    'analyze_ratio': analyze_ratio,
                    'importance_analyzed': importances,
                    'importance_skipped': [],
                    'losses': agent.losses[-episode_length:] if agent.losses else [],
                    'q_values': agent.q_values_history[-episode_length:] if agent.q_values_history else [],
                    'pattern_discovery': new_patterns_this_episode,
                    'confusion': {
                        'tp': metrics_collector.true_positives,
                        'fp': metrics_collector.false_positives,
                        'tn': metrics_collector.true_negatives,
                        'fn': metrics_collector.false_negatives
                    }
                }

            visualizer.update_episode(episode_data)

        # Print progress
        if episode % 10 == 0:
            print(f"\nEpisode {episode + 1}:")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Length: {episode_length}")
            print(f"  Avg Importance (Analyzed): {importance_analyzed[-1]:.3f}")
            print(f"  Patterns: {patterns_discovered[-1]} (new: {new_patterns_this_episode})")
            print(
                f"  Epsilon: {agent.epsilon_end + (agent.epsilon_start - agent.epsilon_end) * np.exp(-1. * agent.steps_done / agent.epsilon_decay):.3f}")

            if agent.losses:
                print(f"  Avg Loss: {np.mean(agent.losses[-10:]):.4f}")
            if agent.q_values_history:
                print(f"  Avg Q-value: {np.mean(agent.q_values_history[-10:]):.4f}")

            print(f"  Actions: ANALYZE={action_counts['ANALYZE']}, SKIP={action_counts['SKIP']}")
            print(f"  Analyze ratio: {analyze_ratio:.3f}")

            # Print localization summary
            if localization_results:
                print(f"  Localizations performed: {len(localization_results)}")

            action_counts = {'ANALYZE': 0, 'SKIP': 0}

    # Training complete - Analysis
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - GENERATING ANALYSIS")
    print("=" * 60)

    # Localization statistics
    print(f"\nLocalization Summary:")
    print(f"  Total localizations: {len(localization_results)}")

    if localization_results:
        avg_nodes = np.mean([len(r['node_level']['vulnerable_nodes']) for r in localization_results])
        avg_exprs = np.mean([len(r['expression_level']['vulnerable_expressions']) for r in localization_results])
        avg_funcs = np.mean([len(r['function_level']['vulnerable_functions']) for r in localization_results])

        print(f"  Avg vulnerable nodes per path: {avg_nodes:.1f}")
        print(f"  Avg vulnerable expressions per path: {avg_exprs:.1f}")
        print(f"  Avg vulnerable functions per path: {avg_funcs:.1f}")

        # Save localization results
        import json
        with open('localization_results.json', 'w') as f:
            json.dump(localization_results, f, indent=2)
        print(f"\n  Localization results saved to localization_results.json")

    # Summary statistics
    print(f"\nTraining Summary:")
    print(f"  Avg Episode Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Avg Episode Length: {np.mean(episode_lengths):.1f}")
    print(f"  Final Buffer Size: {len(buffer)}")
    print(f"  Total Updates: {len(agent.losses)}")

    # Pattern statistics
    pattern_stats = get_pattern_bank_stats(env)
    print(f"\nPattern Bank Statistics:")
    print(f"  Unique patterns: {pattern_stats['unique_patterns']}")
    print(f"  Total patterns seen: {pattern_stats['total_patterns_seen']}")
    if pattern_stats['most_common_pattern']:
        pattern, count = pattern_stats['most_common_pattern']
        print(f"  Most common: {pattern} (seen {count} times)")

    # Analysis plots
    print("\n Generating Analysis Plots...")

    # Call analysis functions
    track_action_evolution(action_history_detailed)
    analyze_q_values(agent, env)
    analyze_performance_breakdown(metrics_collector)

    # Importance distribution analysis
    if importance_history['analyzed'] and importance_history['skipped']:
        plt.figure(figsize=(10, 5))
        plt.hist(importance_history['analyzed'], alpha=0.5, label='Analyzed', bins=30, color='red')
        plt.hist(importance_history['skipped'], alpha=0.5, label='Skipped', bins=30, color='blue')
        plt.xlabel('Importance Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Importance Scores by Action')
        plt.legend()
        plt.axvline(x=np.median(importance_history['analyzed']), color='red', linestyle='--', alpha=0.7)
        plt.axvline(x=np.median(importance_history['skipped']), color='blue', linestyle='--', alpha=0.7)
        plt.savefig('importance_distribution.png')
        plt.show()

    # Final metrics
    print("\n" + "=" * 60)
    print("FINAL METRICS ANALYSIS")
    final_metrics = metrics_collector.get_comprehensive_metrics() if use_enhanced else metrics_collector.get_metrics()

    if use_enhanced:
        print(f"\n Path-level Performance:")
        print(f"  Precision: {final_metrics['path_level']['precision']:.1%}")
        print(f"  Recall: {final_metrics['path_level']['recall']:.1%}")
        print(f"  Accuracy: {final_metrics['path_level']['accuracy']:.1%}")
        print(f"  F1-Score: {final_metrics['path_level']['f1_score']:.3f}")
        print(f"  Pruning: {final_metrics['path_level']['pruning_ratio']:.1%}")

        if 'contract_level' in final_metrics:
            print(f"\n Contract-level Performance:")
            print(f"  Detection Rate: {final_metrics['contract_level']['detection_rate']:.1%}")

    # Final visualization
    if use_enhanced and hasattr(visualizer, 'plot_comprehensive_analysis'):
        # Save model BEFORE visualization
        print("\n💾 Saving trained model...")
        torch.save({
            'q_network_state_dict': agent.q_network.state_dict(),
            'target_network_state_dict': agent.target_network.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'episode': 500
        }, 'trained_dqn_model.pth')
        print("✅ Model saved to trained_dqn_model.pth")
        visualizer.plot_comprehensive_analysis('final_comprehensive_analysis.png')

    return True
# # Save trained model
#     print("\nSaving trained model...")
#     torch.save({
#         'q_network_state_dict': agent.q_network.state_dict(),
#         'target_network_state_dict': agent.target_network.state_dict(),
#         'optimizer_state_dict': agent.optimizer.state_dict(),
#         'episode': 10000,
#         'final_metrics': final_metrics
#     }, 'trained_dqn_model.pth')
#     print("Model saved to trained_dqn_model.pth")
#     # Save model
#     print("\n💾 Saving trained model...")
#     try:
#         torch.save({
#             'q_network_state_dict': agent.q_network.state_dict(),
#             'target_network_state_dict': agent.target_network.state_dict(),
#             'optimizer_state_dict': agent.optimizer.state_dict(),
#             'episode': episode + 1,
#             'final_metrics': final_metrics if use_enhanced else metrics_collector.get_metrics()
#         }, 'trained_dqn_model.pth')
#         print("✅ Model saved successfully to trained_dqn_model.pth")
#     except Exception as e:
#         print(f"❌ Failed to save model: {e}")





def test_new_architecture():
    """تست معماری جدید per-path Q-learning"""
    print("\n" + "=" * 60)
    print("Testing New Per-Path Architecture")
    print("=" * 60)

    # Test 1: State Builder
    print("\n1. Testing StateBuilder...")
    from evn3 import StateBuilder
    state_builder = StateBuilder()
    print(f"   State shape: {state_builder.state_dim}")
    assert state_builder.state_dim == (20, 100), "State dimension should be (20, 100)"
    print("   ✓ StateBuilder OK")

    # Test 2: DQN Network
    print("\n2. Testing DQN Network...")
    dqn = DQN(path_features_dim=100)
    test_state = torch.randn(2, 20, 100)  # batch=2, paths=20, features=100
    q_values = dqn(test_state)
    print(f"   Input shape: {test_state.shape}")
    print(f"   Output shape: {q_values.shape}")
    assert q_values.shape == (2, 20, 2), "Q-values shape should be (batch, paths, 2)"
    print("   ✓ DQN forward pass OK")

    # Test 3: Environment compatibility
    print("\n3. Testing Environment...")
    env = BadRandomnessEnv(
        data_loader=ContractDataLoader(),
        pool_manager=PoolManager(),
        state_builder=StateBuilder(),
        debug_mode=False
    )

    state, info = env.reset()
    print(f"   State shape from env: {state.shape}")
    assert state.shape == (20, 100), "State from env should be (20, 100)"

    # Test action format
    test_action = (0, 'ANALYZE')  # path 0, ANALYZE
    next_state, reward, done, info = env.step(test_action)
    print(f"   Action format works: {test_action}")
    print(f"   Reward: {reward:.2f}")
    print("   ✓ Environment OK")

    # Test 4: Agent
    print("\n4. Testing Agent...")
    agent = DQNAgent(learning_rate=1e-5)
    action = agent.select_action(state, valid_paths=info.get('valid_paths', 10))
    print(f"   Selected action: {action}")
    assert isinstance(action, tuple), "Action should be tuple"
    assert action[1] in ['ANALYZE', 'SKIP'], "Action type should be ANALYZE or SKIP"
    print("   ✓ Agent selection OK")

    # Test 5: ReplayBuffer
    print("\n5. Testing ReplayBuffer...")
    buffer = ReplayBuffer(capacity=100)
    buffer.push(state, action, reward, next_state, done, info)
    if len(buffer) >= 1:
        batch = buffer.sample(1)
        assert 'path_indices' in batch, "Batch should have path_indices"
        assert 'action_types' in batch, "Batch should have action_types"
        print("   ✓ ReplayBuffer OK")

    print("\n" + "=" * 60)
    print("All components working correctly!")
    print("=" * 60)
    return True
def test_replay_buffer():
    """تست عملکرد Replay Buffer با محیط واقعی"""
    print("\n" + "=" * 60)
    print("Testing Replay Buffer")
    print("=" * 60)

    # Initialize environment and buffer
    env = BadRandomnessEnv(
        data_loader=ContractDataLoader(),
        pool_manager=PoolManager(),
        state_builder=StateBuilder(),
        debug_mode=False
    )

    buffer = ReplayBuffer(capacity=500)
    print(f"Buffer created with capacity {buffer.capacity}")

    # Collect experiences from 3 episodes
    for episode in range(3):
        state, info = env.reset()
        done = False
        step = 0
        episode_reward = 0

        while not done and step < 10:
            # تصحیح: استفاده از 2-action space

            action = random.randint(0, 9)  # بجای (0, 39)
            next_state, reward, done, info = env.step(action)

            # Store in buffer
            buffer.push(state, action, reward, next_state, done, info)

            state = next_state
            episode_reward += reward
            step += 1

        print(f"Episode {episode + 1}: {step} steps, total reward: {episode_reward:.2f}")

    # Test buffer statistics
    stats = buffer.get_statistics()
    print(f"\nBuffer Statistics:")
    for key, value in stats.items():
        if key != 'action_distribution':
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")

    # Test sampling
    if len(buffer) >= 32:
        batch = buffer.sample(32)
        print(f"\nBatch Sampling Test:")
        print(f"  States shape: {batch['states'].shape}")
        print(f"  Actions shape: {batch['actions'].shape}")
        print(f"  Rewards shape: {batch['rewards'].shape}")
        print(f"  Rewards in batch: mean={batch['rewards'].mean():.2f}, "
              f"std={batch['rewards'].std():.2f}")

    # Test overflow
    print(f"\nTesting buffer overflow...")
    initial_size = len(buffer)
    for _ in range(50):
        buffer.push(np.zeros(2085), 0, 0.0, np.zeros(2085), False)  # تصحیح: 2085 بجای 2080
    print(f"  Initial size: {initial_size}, After 50 pushes: {len(buffer)}")
    print(f"  Capacity maintained: {len(buffer) <= buffer.capacity}")

    print("\n✓ Replay Buffer tests passed!")
    return True


def test_network_compatibility():
    """تست سازگاری network با environment"""
    print("=" * 60)
    print("Testing DQN Network Compatibility")
    print("=" * 60)

    # Initialize environment
    env = BadRandomnessEnv(
        data_loader=ContractDataLoader(),
        pool_manager=PoolManager(),
        state_builder=StateBuilder(),
        debug_mode=False  # Disable debug prints
    )

    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"Environment state dimension: {state_dim}")
    print(f"Environment action dimension: {action_dim}")

    # Create network
    dqn = DQN(state_dim, action_dim)
    print(f"Network created successfully")

    # Test with random state
    random_state = np.random.randn(state_dim)
    debug_info = dqn.debug_forward(random_state)

    print("\nNetwork forward pass test:")
    for key, value in debug_info.items():
        if key != 'q_values':
            print(f"  {key}: {value}")

    # Test with real environment state
    state, info = env.reset()
    print(f"\nReal environment test:")
    print(f"  State shape: {state.shape}")

    debug_info = dqn.debug_forward(state)
    print(f"  Q-values shape: {debug_info['q_values_shape']}")
    print(f"  Q-values stats: {debug_info['q_stats']}")

    # Test batch processing
    batch_states = np.random.randn(32, state_dim)
    batch_tensor = torch.FloatTensor(batch_states)
    batch_q_values = dqn(batch_tensor)
    print(f"\nBatch processing test:")
    print(f"  Batch input shape: {batch_tensor.shape}")
    print(f"  Batch output shape: {batch_q_values.shape}")

    # Test action selection
    print("\nAction selection test:")
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = dqn(state_tensor)
        action = q_values.argmax(dim=1).item()
        print(f"  Selected action: {action}")
        print(f"  Q-value for selected action: {q_values[0, action].item():.4f}")

    # Save model


    print("\n✓ All compatibility tests passed!")

    return True

# if __name__ == "__main__":
#     # Run diagnostic
#     diagnostic_episode()
# if __name__ == "__main__":
#     # Test network
#     if test_network_compatibility():
#         # Test replay buffer
#         if test_replay_buffer():
#             # Test training loop
#             test_training_loop()
if __name__ == "__main__":
    if test_new_architecture():
        print("\nNew architecture tests passed! Ready for training.")
        test_training_loop()
