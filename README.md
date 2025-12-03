# SmartTaintRL

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

**Efficient Detection and Localization of Bad Randomness Vulnerabilities in Ethereum Smart Contracts via Reinforcement Learning**

## Overview

SmartTaintRL is a framework that combines taint analysis with deep reinforcement learning to detect and localize Bad Randomness vulnerabilities in Ethereum smart contracts. Unlike traditional approaches that rely on pattern matching or exhaustive path exploration, our method uses a Deep Q-Network (DQN) agent to intelligently prioritize high-risk execution paths.

### Key Features

- **High Accuracy**: Achieves F1-score of 0.955 on balanced datasets and 0.950 on imbalanced ones
- **Efficient Path Pruning**: Reduces search space by 45% while maintaining 96% recall
- **Precise Localization**: 92.9% function-level accuracy in identifying vulnerable code
- **Robust to Class Imbalance**: Less than 1% F1 degradation under real-world conditions (95:5 ratio)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     SmartTaintRL Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Phase 1    │    │   Phase 2    │    │   Phase 3    │      │
│  │              │    │              │    │              │      │
│  │ Taint        │───▶│ RL-based    │───▶│ Vulnerability│      │
│  │ Analysis     │    │ Path        │    │ Localization │      │
│  │              │    │ Prioritization   │              │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                 │
│  • Extract paths     • DQN agent        • Function-level       │
│  • 100-dim features  • Pool management  • Node-level           │
│  • Source-sink pairs • Hierarchical     • Gradient-based       │
│                        rewards            attribution          │
└─────────────────────────────────────────────────────────────────┘
```

## Bad Randomness Vulnerability

Bad Randomness is ranked as the 4th most critical smart contract vulnerability by OWASP. Blockchain's deterministic nature prevents true random number generation, and developers often misuse predictable values like:

- `block.timestamp`
- `block.number`
- `blockhash`
- `block.difficulty`

These vulnerabilities have caused significant financial losses, including the SmartBillions hack (400 ETH) and Fomo3D exploit ($3M+).

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/HadisRe/SmartTaintRL.git
cd SmartTaintRL
```

### Step 2: Create Virtual Environment

We recommend using a virtual environment to avoid dependency conflicts.

**On Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (deep learning framework)
- NumPy (numerical computing)
- Gymnasium (RL environment)
- Matplotlib and Seaborn (visualization)

### Step 4: Verify Installation

```bash
python -c "import torch; import gymnasium; print('Installation successful!')"
```

## Project Structure

```
SmartTaintRL/
├── src/
│   ├── rl_agent/
│   │   ├── dqn_agent.py           # Deep Q-Network and training loop
│   │   ├── env1_data_loader.py    # Contract data loader
│   │   ├── env2_pool_manager.py   # Pool management for path selection
│   │   ├── env3_state_builder.py  # 100-dim state vector construction
│   │   └── env4_environment.py    # Main RL environment (Gymnasium)
│   │
│   └── preprocessing/
│       ├── embed1_profile.py      # Contract profile generation
│       ├── embed2_path_db.py      # Path database construction  
│       └── embed3_modifiers.py    # Modifier extraction and integration
│
├── requirements.txt
├── LICENSE
└── README.md

# Data available on Google Drive (see Data section)
```

## Data

### Option 1: Use Pre-processed Data (Recommended)

The pre-processed datasets are available on Google Drive due to file size limitations:

**[Download Dataset from Google Drive](https://drive.google.com/drive/folders/1st4LL5UqWfV2BA-eBALnXXLS_PuZw2Uv)**

The dataset contains:
- `contract_profiles_Balanced/`: Contract profiles for balanced dataset (200 vulnerable, 200 safe)
- `contract_profiles_Imbalanced/`: Contract profiles for imbalanced dataset (223 vulnerable, 4,083 safe)
- `path_databases_updated_Balanced/`: Path-level features for balanced dataset
- `path_databases_updated_Imbalanced/`: Path-level features for imbalanced dataset

Each path database contains 100-dimensional feature vectors extracted from taint analysis results.

### Option 2: Generate Data from Scratch

If you want to process new contracts or regenerate the features, you need to:

**Step 1:** Run [TaintSentinel](https://github.com/HadisRe/TaintSentinel) on your Solidity contracts to generate:
- AST files (`*_ast.json`)
- Semantic graphs (`*_semantic_graph.json`)
- Taint analysis results (`*_taint_analysis_filtered.json`)

**Step 2:** Run the preprocessing scripts in order:

```bash
# Generate contract profiles from TaintSentinel outputs
python src/preprocessing/embed1_profile.py

# Build path databases with 100-dimensional features
python src/preprocessing/embed2_path_db.py

# Extract and integrate modifier information
python src/preprocessing/embed3_modifiers.py
```

These scripts read the TaintSentinel outputs and create the feature files needed for the RL agent.

## Methodology

### Phase 1: Taint Analysis (Preprocessing)

This phase uses [TaintSentinel](https://github.com/HadisRe/TaintSentinel) for:
- Transforming Solidity code into semantic graphs (CFG + DFG)
- Identifying taint sources (weak entropy) and sinks (sensitive operations)
- Extracting all possible taint paths

The preprocessing scripts (`embed*.py`) then process these outputs to create 100-dimensional feature vectors for each path, including:
- Structural features (path length, node diversity, branch complexity)
- Security features (require density, modifier protection, mitigation score)
- Semantic features (keccak operations, arithmetic intensity)
- Source-sink interaction features

### Phase 2: RL-based Path Prioritization

The DQN agent learns to make ANALYZE or SKIP decisions for each path:
- **State**: 20 × 100 matrix (20 paths in pool, 100 features each)
- **Action**: ANALYZE or SKIP for selected path
- **Reward**: Hierarchical reward based on path importance and risk level

Key components:
- Dynamic pool management with priority-based refilling
- Pattern registry for tracking discovered vulnerability signatures
- Hierarchical reward engineering with safety constraints

### Phase 3: Vulnerability Localization

After detection, the system localizes vulnerabilities using:
- Gradient-based attribution from Q-network
- Graph propagation with centrality analysis
- Function-level and node-level ranking

## Comparison with Existing Tools

| Method | F1 (Balanced) | F1 (Imbalanced) | Time |
|--------|---------------|-----------------|------|
| Slither | 0.450 | - | 22min |
| Mythril | 0.372 | - | >24h |
| TaintSentinel | 0.892 | 0.611 | 8.3h |
| RNVulDet | 0.662 | 0.360 | 5.5h |
| **SmartTaintRL** | **0.955** | **0.950** | **3h** |

## Dataset

We used a filtered subset of contracts for RL training:

**Balanced Dataset:** 400 contracts
- 200 vulnerable contracts
- 200 safe contracts

**Imbalanced Dataset:** 4,306 contracts  
- 223 vulnerable contracts
- 4,083 safe contracts

The imbalanced dataset reflects real-world conditions where vulnerable contracts are significantly fewer than safe ones (approximately 1:18 ratio).

## Related Work

This project builds upon:
- [TaintSentinel](https://github.com/HadisRe/TaintSentinel) - Taint analysis framework used for preprocessing

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
