# env2_pool_manager.py
# Pool of candidate paths shown to the agent during an episode.

import random
import logging
from typing import List, Dict, Set, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field
from env1_data_loader import ContractData

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PoolState:
    """Pool state within one episode."""
    current_pool: List[Dict] = field(default_factory=list)
    explored_paths: Set[int] = field(default_factory=set)  # path indices already decided
    available_paths: List[Dict] = field(default_factory=list)  # paths not drawn yet
    refill_count: int = 0
    total_analyzed: int = 0

    def __post_init__(self):
        logger.debug(f"PoolState initialized with {len(self.current_pool)} paths in pool")


class PoolManager:
    """
    Keeps the pool of paths during an episode: draws the initial pool,
    removes decided paths and refills the pool when it gets small.
    The priority score is used only to choose refill paths; it is not
    part of the agent's input.
    """

    def __init__(self,
                 min_pool_threshold: int = 10,
                 priority_start_episode: int = 200,
                 max_priority_ratio: float = 0.5):
        """
        Args:
            min_pool_threshold: refill when the pool drops below this size
            priority_start_episode: episode from which priority selection may be used
            max_priority_ratio: maximum share of priority-selected paths in the pool
        """
        self.min_pool_threshold = min_pool_threshold
        self.priority_start_episode = priority_start_episode
        self.max_priority_ratio = max_priority_ratio
        self.current_episode = 0
        self.current_pool = []
        self.discovered_patterns = set()
        self.analyze_count = 0
        self.skip_count = 0
        self.avg_importance_analyzed = 0.5
        self.total_importance = 0
        self.importance_count = 0

        # Debug counters
        self.debug_stats = {
            'total_paths_selected': 0,
            'total_refills': 0,
            'priority_selections': 0,
            'random_selections': 0
        }

        logger.info(f"PoolManager initialized with threshold={min_pool_threshold}, "
                    f"priority_start={priority_start_episode}")

    def determine_pool_size(self, total_paths: int) -> int:
        """
        Pool size from the number of paths in the contract.
        (Not used by initialize_pool, which has its own rule.)
        """
        if total_paths <= 20:
            size = total_paths
        elif total_paths <= 50:
            size = 20
        elif total_paths <= 100:
            size = 30
        else:
            size = 40

        logger.debug(f"Pool size determined: {size} for {total_paths} total paths")
        return size

    def calculate_priority_score(self, path: Dict) -> float:
        """
        Fixed priority of a path from its source, sink and missing protection.
        Used for refills only; the agent never sees this value.
        This is also the score of the Top-k baseline in the paper (Table 10).

        Returns:
            priority score (higher = more urgent)
        """
        score = 0.0

        # source / sink types
        basic_info = path.get('basic_info', {})
        source_type = basic_info.get('source_type', '')
        sink_type = basic_info.get('sink_type', '')

        # Source priority
        if source_type == 'timestamp':
            score += 2.5
        elif source_type == 'blockhash':
            score += 2.0
        elif source_type in ['blocknumber', 'difficulty']:
            score += 1.5
        elif source_type:  # Any other source
            score += 0.5

        # Sink priority
        if sink_type == 'transfer':
            score += 2.0
        elif sink_type in ['randomGeneration', 'stateModification']:
            score += 1.5
        elif sink_type:  # Any other sink
            score += 0.5

        # Missing protection raises the priority
        features = path.get('aggregate_features', {})
        if features.get('has_modifier_protection', 0) == 0:
            score += 1.5
        if features.get('require_density', 0) < 0.2:
            score += 1.0
        if features.get('has_external_protection', 0) == 0:
            score += 1.0

        # Lower mitigation = higher priority
        reliability = features.get('mitigation_score', 1.0)
        score *= (1 - reliability * 0.5)

        logger.debug(f"Path {path.get('path_index', -1)} priority score: {score:.2f} "
                     f"(source={source_type}, sink={sink_type})")

        return score

    def initialize_pool(self, contract: ContractData, episode: int) -> PoolState:
        """
        Draw the initial pool at random (the only random step of an episode).
        """
        self.current_episode = episode
        all_paths = contract.paths
        total_paths = len(all_paths)

        # pool size, as in Section 4.2.4 of the paper
        if total_paths <= 10:
            pool_size = min(total_paths, 10)
        elif total_paths <= 20:
            pool_size = 15
        elif total_paths <= 40:
            pool_size = 20
        else:
            pool_size = min(30, total_paths // 2)

        selected_indices = np.random.choice(
            total_paths,
            size=min(pool_size, total_paths),
            replace=False
        ).tolist()

        current_pool = [all_paths[i] for i in selected_indices]
        available_paths = [all_paths[i] for i in range(total_paths) if i not in selected_indices]

        pool_state = PoolState(
            current_pool=current_pool,
            available_paths=available_paths,
            explored_paths=set(),
            refill_count=0,
            total_analyzed=0
        )

        logger.info(f"Pool initialized: size={len(current_pool)}/{total_paths} paths")

        return pool_state
    def _select_random(self, paths: List[Dict], count: int) -> List[Dict]:
        """Pick paths at random."""
        count = min(count, len(paths))
        selected = random.sample(paths, count)

        # shuffle so the position in the pool carries no information
        random.shuffle(selected)

        self.debug_stats['random_selections'] += count
        logger.debug(f"Selected {count} paths randomly")
        return selected

    def _select_mixed(self,
                      paths: List[Dict],
                      pool_size: int,
                      priority_ratio: float) -> List[Dict]:
        """
        Mix of priority-based and random selection.

        Args:
            paths: candidate paths
            pool_size: number of paths to select
            priority_ratio: share selected by priority
        """
        pool_size = min(pool_size, len(paths))
        n_priority = int(pool_size * priority_ratio)
        n_random = pool_size - n_priority

        paths_with_priority = [(p, self.calculate_priority_score(p)) for p in paths]
        paths_with_priority.sort(key=lambda x: x[1], reverse=True)

        # highest-priority paths first
        selected = []
        if n_priority > 0:
            priority_paths = [p for p, _ in paths_with_priority[:n_priority]]
            selected.extend(priority_paths)
            self.debug_stats['priority_selections'] += n_priority
            logger.debug(f"Selected {n_priority} high-priority paths")

        # the rest at random
        if n_random > 0:
            remaining = [p for p, _ in paths_with_priority[n_priority:]]
            if len(remaining) >= n_random:
                random_paths = random.sample(remaining, n_random)
            else:
                random_paths = remaining
            selected.extend(random_paths)
            self.debug_stats['random_selections'] += len(random_paths)
            logger.debug(f"Selected {len(random_paths)} random paths")

        # hide the priority order
        random.shuffle(selected)
        return selected

    def handle_path_analyzed(self,
                             pool_state: PoolState,
                             path_index: int) -> bool:
        """
        Remove a decided path from the pool and refill if needed.

        Args:
            pool_state: current pool
            path_index: position of the path in the pool

        Returns:
            True on success, False for an invalid index
        """
        if path_index < 0 or path_index >= len(pool_state.current_pool):
            logger.error(f"Invalid path index: {path_index}")
            return False

        path = pool_state.current_pool[path_index]
        path_id = path.get('path_index', -1)

        pool_state.explored_paths.add(path_id)
        pool_state.total_analyzed += 1

        pool_state.current_pool.pop(path_index)

        logger.debug(f"Path {path_id} analyzed and removed from pool. "
                     f"Pool size now: {len(pool_state.current_pool)}")

        if self._needs_refill(pool_state):
            self.refill_pool(pool_state)

        return True

    def _needs_refill(self, pool_state: PoolState) -> bool:
        """True if the pool is below the threshold and paths are left."""
        needs = (len(pool_state.current_pool) < self.min_pool_threshold and
                 len(pool_state.available_paths) > 0)

        if needs:
            logger.debug(f"Pool needs refill: {len(pool_state.current_pool)} < {self.min_pool_threshold}")

        return needs

    def refill_pool(self, pool_state: PoolState) -> int:
        """
        Refill the pool up to min_pool_threshold.

        Returns:
            number of paths added
        """
        target_size = self.min_pool_threshold
        current_size = len(pool_state.current_pool)
        n_to_add = min(target_size - current_size, len(pool_state.available_paths))

        if n_to_add <= 0:
            logger.warning("No paths available for refill")
            return 0

        # half by priority, half at random
        n_priority = n_to_add // 2
        n_random = n_to_add - n_priority

        new_paths = []

        # Priority selection
        if n_priority > 0:
            paths_with_priority = [(p, self.calculate_priority_score(p))
                                   for p in pool_state.available_paths]
            paths_with_priority.sort(key=lambda x: x[1], reverse=True)

            for i in range(min(n_priority, len(paths_with_priority))):
                path = paths_with_priority[i][0]
                new_paths.append(path)
                pool_state.available_paths.remove(path)

        # Random selection from the remaining paths
        if n_random > 0 and pool_state.available_paths:
            random_count = min(n_random, len(pool_state.available_paths))
            random_paths = random.sample(pool_state.available_paths, random_count)
            new_paths.extend(random_paths)
            for p in random_paths:
                pool_state.available_paths.remove(p)

        pool_state.current_pool.extend(new_paths)
        pool_state.refill_count += 1

        # shuffle so refilled paths cannot be told apart by position
        random.shuffle(pool_state.current_pool)

        self.debug_stats['total_refills'] += 1
        logger.info(f"Refilled pool with {len(new_paths)} paths. "
                    f"New pool size: {len(pool_state.current_pool)}")

        return len(new_paths)

    def get_pool_statistics(self, pool_state: PoolState) -> Dict:
        """Pool statistics for logging."""
        stats = {
            'current_pool_size': len(pool_state.current_pool),
            'explored_count': len(pool_state.explored_paths),
            'available_count': len(pool_state.available_paths),
            'refill_count': pool_state.refill_count,
            'total_analyzed': pool_state.total_analyzed,
            'exploration_ratio': pool_state.total_analyzed /
                                 (pool_state.total_analyzed +
                                  len(pool_state.current_pool) +
                                  len(pool_state.available_paths))
            if pool_state.total_analyzed > 0 else 0
        }

        logger.debug(f"Pool stats: {stats}")
        return stats

    def reset_debug_stats(self):
        """Reset debug counters"""
        self.debug_stats = {
            'total_paths_selected': 0,
            'total_refills': 0,
            'priority_selections': 0,
            'random_selections': 0
        }
        logger.info("Debug stats reset")
