# SmartTaintRL

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-red.svg)](https://pytorch.org/)

**Efficient Detection and Localization of Bad Randomness Vulnerabilities in Ethereum Smart Contracts via Reinforcement Learning**

## Overview

SmartTaintRL combines taint analysis with deep reinforcement learning to detect and localize Bad Randomness
vulnerabilities in Ethereum smart contracts. A taint analysis extracts the paths from weak randomness sources
(`block.timestamp`, `block.number`, `blockhash`, ...) to sensitive operations. A Deep Q-Network agent then decides,
path by path, which paths to analyze and which to skip. The agent's decisions are used for the contract-level
verdict and for locating the vulnerable function and nodes.

### Key Features

- **Contract-level detection**: F1-score of 0.918 (balanced) and 0.701 (imbalanced), higher than Slither, RNVulDet,
  Mythril and TaintSentinel under the same evaluation protocol
- **Path pruning**: skips about 40% of the taint paths with a path-level recall of 0.938 (balanced) and 0.964 (imbalanced)
- **Localization**: ranks the vulnerable function or one of its callers first in 71.4% of the ground-truth
  contracts and among the top two in 85.7%

## Pipeline

```
Phase 1: taint analysis        Phase 2: path prioritization      Phase 3: localization
(TaintSentinel + embed*.py) -> (DQN agent, ANALYZE / SKIP)    -> (function and node level)

- source-sink paths            - state: 20 paths x 100 features  - gradient and attention of the Q-network
- 100 features per path        - hierarchical reward             - propagation over the semantic graph
                               - contract-level decision          - node centrality
```

## Installation

```bash
git clone https://github.com/HadisRe/SmartTaintRL.git
cd SmartTaintRL
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The paper used Python 3.12, PyTorch 2.6 (CPU), NetworkX 3.4 and scikit-learn 1.7 on a laptop without a GPU.
Slither, Mythril, RNVulDet and TaintSentinel are only needed to rerun the tool comparison.

## Repository Structure

```
SmartTaintRL/
├── src/
│   ├── preprocessing/       embed1_profile.py, embed2_path_db.py, embed3_modifiers.py
│   ├── rl_agent/            env1_data_loader.py, env2_pool_manager.py, env3_state_builder.py,
│   │                        env4_environment.py, dqn_agent.py
│   └── localization/        localizer.py, test_localization.py,
│                            data/ground_truth/final_dataset_14_contracts.pkl
├── models/                  dqn_<set>_fold<k>.pth (10 cross-validation models),
│                            trained_dqn_model_2500.pth (localization)
├── experiments/             scripts for every table and figure of the paper (see experiments/README.md)
│   └── folds/               the five folds used for all methods
├── requirements.txt
└── LICENSE
```

## Data

The pre-processed data is on Google Drive because of its size:
**[Download the dataset](https://drive.google.com/drive/folders/1st4LL5UqWfV2BA-eBALnXXLS_PuZw2Uv)**

Put it in a folder `data/` next to `src/` (or anywhere else, and set `SMARTTAINTRL_DATA` to that folder), with
these names:

```
data/
├── path_databases_balanced/       (path_databases_update_Blanced on Drive)
├── contract_profiles_balanced/    (contract_profiles_Balanced on Drive)
├── path_databases_imbalanced/     (path_databases_updated_Imbalanced on Drive)
├── contract_profiles_imbalanced/  (contract_profiles_Imbalanced on Drive)
├── contracts/vulnerable/*.sol     Solidity sources, one file per address
├── contracts/safe/*.sol
├── path_databases_node_level/     localization
└── contract_ast1_clean/           semantic graphs, localization
```

The experiments use the contracts that compile and have at least 5 taint paths:

| Dataset | Contracts | Vulnerable | Safe | Paths |
|---|---|---|---|---|
| Balanced | 284 | 157 | 127 | 5,160 |
| Imbalanced | 2,549 | 177 | 2,372 | 245,810 |

The fold of every contract is listed in `experiments/folds/contracts_<set>.csv`.

To build the features for new contracts, first run [TaintSentinel](https://github.com/HadisRe/TaintSentinel) on the
Solidity files (AST, semantic graph and taint paths), then:

```bash
python src/preprocessing/embed1_profile.py      # contract profiles
python src/preprocessing/embed2_path_db.py      # path databases, 100 features per path
python src/preprocessing/embed3_modifiers.py    # modifier information
```

## Usage

Cross-validation training (one model per fold, 2,500 episodes each):

```bash
python experiments/cv/kfold_eval.py balanced
```

Path-level results of the saved models (Table 7):

```bash
python experiments/cv/score_test_folds.py balanced
```

Contract-level results (Table 8):

```bash
for k in 1 2 3 4 5; do python experiments/contract_level/dump_agent.py balanced $k; done
python experiments/contract_level/contract_head.py balanced
```

Localization on the 14 ground-truth contracts (Tables 12 and 13):

```bash
PYTHONHASHSEED=14 python src/localization/test_localization.py --ablation
```

Replace `balanced` with `imbalanced` for the other dataset. `experiments/README.md` lists the script of every
other result (tool comparison, path-level baselines, statistics and the additional analyses).

## Results

Contract-level detection, same contracts and folds for every tool (5-fold cross-validation):

| Method | F1 (balanced) | F1 (imbalanced) |
|---|---|---|
| Mythril | 0.322 | – |
| RNVulDet | 0.639 | 0.395 |
| TaintSentinel | 0.735 | 0.531 |
| Slither (weak-prng) | 0.784 | 0.495 |
| **SmartTaintRL** | **0.918** | **0.701** |

Localization: the vulnerable function or a caller is ranked first in 71.4% of the contracts and among the top two
in 85.7%; node level P@5 0.68, R@5 0.80, F1@5 0.73.

## Related Work

- [TaintSentinel](https://github.com/HadisRe/TaintSentinel): the taint analysis used in Phase 1

## License

MIT License, see [LICENSE](LICENSE).
