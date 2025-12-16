 

import gymnasium as gym  # تغییر از gym به gymnasium
from gymnasium import spaces  # تغییر از gym.spaces
import numpy as np
import logging
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
import random

from env3 import ContractDataLoader, ContractData
from env2 import PoolManager, PoolState
from env3 import StateBuilder

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EpisodeInfo:
    """اطلاعات episode برای tracking"""
    contract_address: str
    contract_label: str  # ground truth
    total_paths: int
    steps: int = 0
    total_reward: float = 0.0
    paths_analyzed: int = 0
    paths_skipped: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0


class BadRandomnessEnv(gym.Env):
    """
    محیط RL برای تشخیص Bad Randomness

    Action Space:
    - Discrete action (0 to 39):
      - 0-19: ANALYZE path at index
      - 20-39: SKIP path at index

    State Space:
    - Box(580,): state vector from StateBuilder

    Reward:
    - Based on action correctness and path risk level
    """

    def __init__(self,
                 data_loader: Optional[ContractDataLoader] = None,
                 pool_manager: Optional[PoolManager] = None,
                 state_builder: Optional[StateBuilder] = None,
                 max_steps: int = 50,
                 budget: float = 100.0,
                 reward_config: Optional[Dict] = None,
                 debug_mode: bool = True):

        self.action_space = spaces.Tuple((
            spaces.Discrete(20),  # path_index
            spaces.Discrete(2)  # action_type: 0=ANALYZE, 1=SKIP
        ))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(20, 100),  # structured state
            dtype=np.float32
        )

        # Initialize components
        self.data_loader = data_loader or ContractDataLoader()
        self.pool_manager = pool_manager or PoolManager()
        self.state_builder = state_builder or StateBuilder()

        # Environment parameters
        self.max_steps = max_steps
        self.initial_budget = budget
        self.budget = budget

        # Reward configuration
        self.reward_config = reward_config or {
            # Action costs
            'analyze_cost': 0.5,
            'skip_cost': 0.0,

            # Pattern-based rewards
            'true_positive_bonus': 2.0,
            'false_positive_penalty': 1.0,
            'true_negative_bonus': 0.5,
            'false_negative_penalty': 5.0,

            # Terminal rewards
            'episode_success_bonus': 10.0,
            'episode_failure_penalty': 10.0,

            # Other
            'invalid_action_penalty': 2.0,
            'efficiency_bonus': 0.3,
        }

        # Episode tracking
        self.current_episode = 0
        self.current_contract: Optional[ContractData] = None
        self.pool_state: Optional[PoolState] = None
        self.episode_info: Optional[EpisodeInfo] = None

        # Debug tracking
        self.debug_mode = debug_mode
        self.action_history = []
        self.reward_history = []

        # Pattern tracking
        self.pattern_bank = {}
        self.discovered_patterns = set()

        logger.info("Environment initialized with 2-action space")
        logger.debug(f"Action space: {self.action_space}")
        logger.debug(f"Observation space: {self.observation_space}")
    def _extract_pattern_signature(self, path: Dict) -> Tuple:
        """
        استخراج pattern signature از path برای pattern bank
        Returns: tuple (source_type, sink_type, has_strong_mitigation)
        """
        features = path.get('aggregate_features', {})
        basic_info = path.get('basic_info', {})

        # Get source type
        source_type = basic_info.get('source_type', 'unknown')

        # Get sink type
        sink_type = basic_info.get('sink_type', 'unknown')

        # Check for strong mitigation
        mitigation_score = features.get('mitigation_score', 0)
        require_density = features.get('require_density', 0)
        has_modifier = features.get('has_modifier_protection', 0)

        has_strong_mitigation = (
                mitigation_score > 0.7 or
                require_density > 0.5 or
                has_modifier == 1
        )

        pattern_signature = (source_type, sink_type, has_strong_mitigation)

        if self.debug_mode:
            logger.debug(f"Pattern signature: {pattern_signature}")

        return pattern_signature

    def _update_pattern_bank(self, path: Dict, action_type: str) -> Dict:
        """
        به‌روزرسانی pattern bank و محاسبه pattern statistics
        Returns: dict with pattern info for reward calculation
        """
        pattern_info = {
            'is_new': False,
            'repetition_count': 0,
            'uniqueness_score': 0.0
        }

        # فقط برای ANALYZE patterns را track می‌کنیم
        if action_type != 'ANALYZE':
            return pattern_info

        pattern = self._extract_pattern_signature(path)

        # Check if new pattern
        if pattern not in self.discovered_patterns:
            pattern_info['is_new'] = True
            pattern_info['uniqueness_score'] = 1.0
            self.discovered_patterns.add(pattern)

        # Update count
        if pattern in self.pattern_bank:
            self.pattern_bank[pattern] += 1
        else:
            self.pattern_bank[pattern] = 1

        pattern_info['repetition_count'] = self.pattern_bank[pattern]

        # Calculate uniqueness based on repetition
        if pattern_info['repetition_count'] > 1:
            pattern_info['uniqueness_score'] = 1.0 / pattern_info['repetition_count']

        if self.debug_mode:
            logger.info(
                f"Pattern bank update: {pattern} - Count: {pattern_info['repetition_count']}, New: {pattern_info['is_new']}")

        return pattern_info

    def _calculate_importance_score(self, path: Dict, pattern_info: Dict) -> float:
        """
        محاسبه importance score با تاکید قوی بر mitigation
        """
        features = path.get('aggregate_features', {})
        basic_info = path.get('basic_info', {})

        # Component 1: Risk Level - وزن 30% (کاهش از 40%)
        risk_level = basic_info.get('risk_level', 'UNKNOWN')
        if risk_level == 'HIGH':
            risk_level_score = 0.8
        elif risk_level == 'MEDIUM':
            risk_level_score = 0.5
        elif risk_level == 'LOW':
            risk_level_score = 0.2
        else:  # UNKNOWN
            risk_level_score = 0.4

        # Component 2: Source-Sink combination - وزن 25% (کاهش از 30%)
        source_encoded = features.get('source_type_encoded', [0, 0, 0, 0])
        sink_encoded = features.get('sink_type_encoded', [0, 0, 0, 0])

        # Source scoring
        source_risk = 0.0
        if source_encoded[0] == 1:  # timestamp
            source_risk = 0.9
        elif source_encoded[1] == 1:  # blocknumber
            source_risk = 0.8
        elif source_encoded[2] == 1:  # blockhash
            source_risk = 0.7
        elif source_encoded[3] == 1:  # sender
            source_risk = 0.5
        else:
            source_type = basic_info.get('source_type', 'unknown')
            if source_type == 'timestamp':
                source_risk = 0.9
            elif source_type == 'blocknumber':
                source_risk = 0.8
            elif source_type == 'blockhash':
                source_risk = 0.7
            elif source_type == 'sender':
                source_risk = 0.5
            else:
                source_risk = 0.3

        # Sink scoring
        sink_risk = 0.0
        if sink_encoded[0] == 1:  # valueTransfer
            sink_risk = 1.0
        elif sink_encoded[1] == 1:  # randomGeneration
            sink_risk = 0.8
        elif sink_encoded[2] == 1:  # stateModification
            sink_risk = 0.6
        else:
            sink_type = basic_info.get('sink_type', 'unknown')
            if sink_type == 'valueTransfer':
                sink_risk = 1.0
            elif sink_type == 'randomGeneration':
                sink_risk = 0.8
            elif sink_type == 'stateModification':
                sink_risk = 0.6
            elif sink_type == 'controlFlow':
                sink_risk = 0.4
            else:
                sink_risk = 0.3

        source_sink_score = (source_risk + sink_risk) / 2.0

        # Component 3: Pattern uniqueness - وزن 15% (کاهش از 20%)
        uniqueness_score = pattern_info.get('uniqueness_score', 0.5)

        # Component 4: Mitigation - وزن 30% (افزایش از 10%)
        mitigation_score = features.get('mitigation_score', 0)
        require_density = features.get('require_density', 0)
        has_modifier = features.get('has_modifier_protection', 0)
        has_restricted_visibility = features.get('has_restricted_visibility', 0)
        has_external_protection = features.get('has_external_protection', 0)

        # محاسبه mitigation قوی‌تر
        mitigation_factor = 0.0

        # هر نوع mitigation امتیاز مستقل دارد
        if has_modifier == 1:
            mitigation_factor += 0.3
        if require_density > 0.5:
            mitigation_factor += 0.4
        elif require_density > 0.3:
            mitigation_factor += 0.2
        if mitigation_score > 0.7:
            mitigation_factor += 0.3
        elif mitigation_score > 0.5:
            mitigation_factor += 0.2
        if has_restricted_visibility == 1:
            mitigation_factor += 0.2
        if has_external_protection == 1:
            mitigation_factor += 0.2

        # Cap mitigation در 1.0
        mitigation_factor = min(mitigation_factor, 1.0)

        # محاسبه base importance بدون mitigation
        base_importance = (risk_level_score * 0.3 +
                           source_sink_score * 0.25 +
                           uniqueness_score * 0.15)

        # اعمال mitigation به صورت multiplicative (نه additive)
        # این باعث می‌شود mitigation قوی واقعاً importance را کاهش دهد
        importance = base_importance * (1 - mitigation_factor * 0.7)

        # اضافه کردن noise کوچک
        import random
        importance += random.uniform(-0.02, 0.02)

        # حذف حد پایین برای HIGH risk با strong mitigation
        # فقط از 0.05 به عنوان حد پایین مطلق استفاده می‌کنیم
        importance = max(0.05, min(1.0, importance))

        if self.debug_mode:
            logger.debug(f"Importance: {importance:.3f} "
                         f"(risk={risk_level_score:.2f}, "
                         f"source_sink={source_sink_score:.2f}, "
                         f"unique={uniqueness_score:.2f}, "
                         f"mitigation_factor={mitigation_factor:.2f}, "
                         f"base={base_importance:.3f})")

        return importance


    def reset(self) -> Tuple[np.ndarray, Dict]:
        """
        شروع episode جدید
        Returns:
            tuple: (initial_state, info_dict)
        """
        self.current_episode += 1
        logger.info(f"=== Starting Episode {self.current_episode} ===")

        # انتخاب contract تصادفی
        contracts = self.data_loader.get_valid_contracts()
        if not contracts:
            raise ValueError("No valid contracts available")

        selected_addr, path_count, label = random.choice(contracts)
        self.current_contract = self.data_loader.load_contract(selected_addr)

        if not self.current_contract:
            raise ValueError(f"Failed to load contract {selected_addr}")

        # Initialize pool
        self.pool_state = self.pool_manager.initialize_pool(
            self.current_contract,
            self.current_episode
        )

        # Reset environment state
        self.budget = self.initial_budget
        self.state_builder.reset()
        self.action_history = []
        self.reward_history = []


        # Initialize episode info
        self.episode_info = EpisodeInfo(
            contract_address=selected_addr,
            contract_label=label,
            total_paths=path_count
        )

        # Build initial state
        state = self.state_builder.build_state(
            self.current_contract,
            self.pool_state,
            budget_used=(self.initial_budget - self.budget) / self.initial_budget,
            max_steps=self.max_steps
        )

        # Create info dict with action mask
        info = {
            'pool_size': len(self.pool_state.current_pool),
            'contract_address': selected_addr,
            'contract_label': label
        }

        logger.info(f"Episode {self.current_episode} reset complete")
        logger.info(f"Contract: {selected_addr[:10]}... ({label})")
        logger.info(f"Total paths: {path_count}, Pool size: {len(self.pool_state.current_pool)}")

        return state, info

    def step(self, action: Tuple[int, str]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        اجرای action با format جدید: (path_index, action_type)

        Args:
            action: tuple از (path_index, action_type) که action_type یکی از 'ANALYZE' یا 'SKIP' است

        Returns:
            tuple: (next_state, reward, done, info)
        """
        # Extract action components
        path_index, action_type = action

        # Validate path index
        pool_size = len(self.pool_state.current_pool)
        if pool_size == 0:
            logger.warning("Pool is empty")
            return self._handle_invalid_action()

        if path_index >= pool_size or path_index < 0:
            logger.warning(f"Invalid path index {path_index}, pool size: {pool_size}")
            return self._handle_invalid_action()

        # Get the selected path
        path = self.pool_state.current_pool[path_index]
        path_risk = path.get('basic_info', {}).get('risk_level', 'UNKNOWN')

        # Calculate importance for this path
        pattern_sig = self._extract_pattern_signature(path)
        pattern_info = {
            'is_new': pattern_sig not in self.discovered_patterns,
            'repetition_count': self.pattern_bank.get(pattern_sig, 0),
            'uniqueness_score': 1.0 if pattern_sig not in self.discovered_patterns else 1.0 / (
                    self.pattern_bank.get(pattern_sig, 0) + 1)
        }
        importance = self._calculate_importance_score(path, pattern_info)

        # Update pattern bank only for ANALYZE
        if action_type == 'ANALYZE':
            pattern_info = self._update_pattern_bank(path, action_type)
            path['pattern_info'] = pattern_info

        # Calculate reward
        reward = self._calculate_reward(action_type, path, path_risk, importance)




        if self.debug_mode:
            logger.info(f"Step {self.episode_info.steps}: {action_type} path {path_index}, "
                        f"importance={importance:.3f}, reward={reward:.2f}")

        # Update environment based on action
        if action_type == 'ANALYZE':
            self._handle_analyze(path_index, path_risk)
        else:  # SKIP
            self._handle_skip(path_index, path_risk)

        # Update episode info
        self.episode_info.steps += 1
        self.episode_info.total_reward += reward
        self.state_builder.update_step(action_type)

        # Check if episode is done
        done = self._is_done()

        # Build next state using structured format
        next_state = self.state_builder.build_state(
            self.current_contract,
            self.pool_state,
            budget_used=(self.initial_budget - self.budget) / self.initial_budget,
            max_steps=self.max_steps
        )

        # Create info dictionary
        info = {
            'episode': self.current_episode,
            'step': self.episode_info.steps,
            'action_type': action_type,
            'path_index': path_index,
            'path_importance': importance,
            'path_risk': path_risk,
            'pool_size': len(self.pool_state.current_pool),
            'budget_remaining': self.budget,
            'paths_analyzed': self.episode_info.paths_analyzed,
            'paths_skipped': self.episode_info.paths_skipped,
            'valid_paths': len(self.pool_state.current_pool),  # برای agent
        }

        # Track action history
        self.action_history.append((action_type, path_index))
        self.reward_history.append(reward)

        return next_state, reward, done, info

    def _calculate_calibrated_pattern_score(self, path: Dict) -> float:
        """
        محاسبه pattern score با weights کالیبره شده
        """
        import pickle
        import numpy as np

        # Load calibrated weights
        if not hasattr(self, 'calibrated_weights'):
            try:
                with open('calibrated_weights.pkl', 'rb') as f:
                    data = pickle.load(f)
                    self.calibrated_weights = data['weights']
                    self.calibrated_intercept = data['intercept']
            except:
                # Fallback to simple scoring
                return self._calculate_pattern_score(path)

        # Extract features (همان 25 features)
        features = []
        agg = path.get('aggregate_features', {})

        # 17 numerical
        for key in ['path_length_normalized', 'require_density', 'condition_density',
                    'keccak_density', 'mitigation_score', 'has_any_mitigation',
                    'has_strong_mitigation', 'has_modifier_protection',
                    'has_restricted_visibility', 'has_external_protection',
                    'function_require_density', 'unique_functions_ratio',
                    'node_diversity', 'distance_to_sink', 'distance_from_source',
                    'has_data_flow', 'contains_loop']:
            features.append(agg.get(key, 0))

        # 4 source + 4 sink
        features.extend(agg.get('source_type_encoded', [0,0,0,0]))
        features.extend(agg.get('sink_type_encoded', [0,0,0,0]))

        features = np.array(features)

        # Calculate score
        if len(self.calibrated_weights.shape) > 1:
            # Multi-class: use probability for HIGH class
            score = np.dot(features, self.calibrated_weights[2]) + self.calibrated_intercept[2]
        else:
            score = np.dot(features, self.calibrated_weights) + self.calibrated_intercept

        return float(score)

    def _calculate_pattern_score(self, path: Dict) -> float:
        """
        محاسبه pattern score با وزن‌های اصلاح شده
        """
        features = path.get('aggregate_features', {})

        vuln_score = 0.0
        safe_score = 0.0

        # Source scoring
        source_encoded = features.get('source_type_encoded', [0, 0, 0, 0])
        vuln_score += sum(source_encoded) * 2.0  # ساده: هر source = 2 امتیاز

        # Sink scoring
        sink_encoded = features.get('sink_type_encoded', [0, 0, 0, 0])
        if sink_encoded[0] == 1:  # transfer
            vuln_score += 3.0
        elif sink_encoded[1] == 1:  # randomGeneration
            vuln_score += 2.5
        else:
            vuln_score += 1.0

        # Protection - وزن کمتر
        if features.get('has_modifier_protection', 0) == 1:
            safe_score += 0.5  # از 3.0 به 0.5
        if features.get('require_density', 0) > 0.5:
            safe_score += 0.5

        net_score = vuln_score - safe_score

        return net_score

    

    def _calculate_reward(self, action_type: str, path: Dict, path_risk: str, importance: float) -> float:
        """
        Simplified hierarchical reward function
        """

        # Level 1: Critical Safety Rules (Absolute)
        if path_risk == 'HIGH' and action_type == 'SKIP':
            return -10.0  # Never skip HIGH risk
        if path_risk == 'LOW' and action_type == 'ANALYZE':
            return -3.0  # Avoid analyzing obvious LOW risk

        # Level 2: Importance-Based Core Reward
        median_importance = 0.27  # از داده واقعی شما
        if action_type == 'ANALYZE':
            # Positive reward if importance > median, negative if below
            reward = (importance - median_importance) * 25
        else:  # SKIP
            # Positive reward if importance < median, negative if above
            reward = (median_importance - importance) * 25

        # Level 3: MEDIUM Risk Handling (importance-dependent)
        if path_risk == 'MEDIUM':
            if importance > 0.3 and action_type == 'SKIP':
                reward -= 2.0  # Mild penalty for skipping important MEDIUM
            elif importance < 0.24 and action_type == 'ANALYZE':
                reward -= 1.0  # Mild penalty for analyzing unimportant MEDIUM

        # Level 4: Contract Context (subtle influence)
        if hasattr(self.episode_info, 'contract_label'):
            if self.episode_info.contract_label == 'SAFE':
                reward += 1.5 if action_type == 'SKIP' else -0.5
            elif self.episode_info.contract_label == 'VULNERABLE':
                reward += 0.5 if action_type == 'ANALYZE' else -0.5

        # Level 5: Pattern Discovery Bonus (optional, keep if useful)
        pattern_info = path.get('pattern_info', {})
        if action_type == 'ANALYZE' and pattern_info.get('is_new', False):
            reward += 2.0

        return np.clip(reward, -12.0, 12.0)

    def _handle_analyze(self, path_index: int, path_risk: str):
        """Handle ANALYZE action با refill خودکار"""
        self.episode_info.paths_analyzed += 1

        # Remove from pool
        if path_index < len(self.pool_state.current_pool):
            del self.pool_state.current_pool[path_index]
            logger.debug(f"Path {path_index} analyzed (risk={path_risk})")

        # همیشه pool را پر نگه دار
        min_pool_size = max(10, len(self.pool_state.current_pool))
        while len(self.pool_state.current_pool) < min_pool_size and len(self.pool_state.available_paths) > 0:
            new_path = self.pool_state.available_paths.pop(0)
            self.pool_state.current_pool.append(new_path)

        if self.debug_mode:
            logger.debug(
                f"Pool size: {len(self.pool_state.current_pool)}, Available: {len(self.pool_state.available_paths)}")

    def _handle_skip(self, path_index: int, path_risk: str):
        """Handle SKIP action با حذف path"""
        self.episode_info.paths_skipped += 1

        # حذف path که skip شده
        if path_index < len(self.pool_state.current_pool):
            del self.pool_state.current_pool[path_index]
            logger.debug(f"Path {path_index} skipped and removed (risk={path_risk})")

        # همیشه pool را پر نگه دار
        min_pool_size = max(10, len(self.pool_state.current_pool))
        while len(self.pool_state.current_pool) < min_pool_size and len(self.pool_state.available_paths) > 0:
            new_path = self.pool_state.available_paths.pop(0)
            self.pool_state.current_pool.append(new_path)

        if self.debug_mode:
            logger.debug(
                f"Pool size: {len(self.pool_state.current_pool)}, Available: {len(self.pool_state.available_paths)}")

    def _handle_invalid_action(self) -> Tuple[np.ndarray, float, bool, Dict]:
        """Handle invalid action"""
        reward = -self.reward_config['invalid_action_penalty']
        self.episode_info.steps += 1
        self.episode_info.total_reward += reward

        state = self.state_builder.build_state(
            self.current_contract,
            self.pool_state,
            budget_used=(self.initial_budget - self.budget) / self.initial_budget,
            max_steps=self.max_steps
        )

        info = {
            'error': 'Invalid action',
            'episode': self.current_episode,
            'step': self.episode_info.steps,
            # 'action_mask': self._get_valid_action_mask(),
        }

        done = self._is_done()

        logger.warning(f"Invalid action penalty: {reward}")

        return state, reward, done, info

    def _get_valid_action_mask(self) -> np.ndarray:
        """
        ساخت mask برای valid actions در 10-action space
        """
        mask = np.zeros(self.action_space.n, dtype=bool)
        pool_size = len(self.pool_state.current_pool) if self.pool_state else 0

        if pool_size > 0:
            # Valid ANALYZE actions (0 to min(4, pool_size-1))
            for i in range(min(5, pool_size)):
                mask[i] = True

            # Valid SKIP actions (5 to 5+min(4, pool_size-1))
            for i in range(min(5, pool_size)):
                mask[5 + i] = True

        return mask
    def _is_done(self) -> bool:
        """Check if episode should end"""
        # Max steps reached
        if self.episode_info.steps >= self.max_steps:
            logger.info(f"Episode done: max steps ({self.max_steps}) reached")
            return True

        # Budget exhausted
        if self.budget <= 0:
            logger.info(f"Episode done: budget exhausted")
            return True

        # Pool empty and no refill possible
        if len(self.pool_state.current_pool) == 0 and len(self.pool_state.available_paths) == 0:
            logger.info(f"Episode done: no paths available")
            return True

        # High confidence threshold (optional)
        if self.episode_info.true_positives >= 3:
            logger.info(f"Episode done: high confidence vulnerable")
            return True

        return False

    def get_episode_summary(self) -> Dict:
        """Get summary of current episode"""
        if not self.episode_info:
            return {}

        precision = (self.episode_info.true_positives /
                    max(self.episode_info.true_positives + self.episode_info.false_positives, 1))
        recall = (self.episode_info.true_positives /
                 max(self.episode_info.true_positives + self.episode_info.false_negatives, 1))

        return {
            'episode': self.current_episode,
            'contract': self.episode_info.contract_address[:10] + '...',
            'label': self.episode_info.contract_label,
            'steps': self.episode_info.steps,
            'total_reward': self.episode_info.total_reward,
            'paths_analyzed': self.episode_info.paths_analyzed,
            'paths_skipped': self.episode_info.paths_skipped,
            'true_positives': self.episode_info.true_positives,
            'false_positives': self.episode_info.false_positives,
            'true_negatives': self.episode_info.true_negatives,
            'false_negatives': self.episode_info.false_negatives,
            'precision': precision,
            'recall': recall,
            'f1': 2 * precision * recall / max(precision + recall, 1e-6)
        }

    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            summary = self.get_episode_summary()
            print("\n" + "="*50)
            print(f"Episode {summary.get('episode', 0)}")
            print(f"Contract: {summary.get('contract', 'N/A')} ({summary.get('label', 'N/A')})")
            print(f"Steps: {summary.get('steps', 0)}/{self.max_steps}")
            print(f"Reward: {summary.get('total_reward', 0):.2f}")
            print(f"Analyzed: {summary.get('paths_analyzed', 0)}, Skipped: {summary.get('paths_skipped', 0)}")
            print(f"Pool size: {len(self.pool_state.current_pool) if self.pool_state else 0}")
            print("="*50)
