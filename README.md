# SmartTaintRL

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

**Efficient Detection and Localization of Bad Randomness Vulnerabilities in Ethereum Smart Contracts via Reinforcement Learning**

## Overview

SmartTaintRL is a novel framework that combines **taint analysis** with **deep reinforcement learning** to detect and localize Bad Randomness vulnerabilities in Ethereum smart contracts. Unlike traditional approaches that rely on pattern matching or exhaustive path exploration, our method uses a Deep Q-Network (DQN) agent to intelligently prioritize high-risk execution paths.

### Key Features

- 🎯 **High Accuracy**: Achieves F1-score of 0.955 on balanced datasets and 0.950 on imbalanced ones
- ⚡ **Efficient Path Pruning**: Reduces search space by 45% while maintaining 96% recall
- 🔍 **Precise Localization**: 92.9% function-level accuracy in identifying vulnerable code
- 📊 **Robust to Class Imbalance**: Less than 1% F1 degradation under real-world conditions (95:5 ratio)

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

Bad Randomness is ranked as the **4th most critical** smart contract vulnerability by OWASP. Blockchain's deterministic nature prevents true random number generation, and developers often misuse predictable values like:

- `block.timestamp`
- `block.number`
- `blockhash`
- `block.difficulty`

These vulnerabilities have caused significant financial losses, including the SmartBillions hack (400 ETH) and Fomo3D exploit ($3M+).

## Installation

```bash
# Clone the repository
git clone https://github.com/YourUsername/SmartTaintRL.git
cd SmartTaintRL

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy
- NetworkX
- Solidity compiler (solc)

## Project Structure

```
SmartTaintRL/
├── src/
│   ├── taint_analyzer.py      # Taint analysis module
│   ├── feature_extractor.py   # 100-dimensional feature extraction
│   ├── dqn_agent.py           # Deep Q-Network implementation
│   ├── localizer.py           # Vulnerability localization
│   └── utils.py               # Utility functions
├── models/
│   └── trained_dqn_model.pth  # Pre-trained model weights
├── data/
│   └── sample_contracts/      # Example vulnerable contracts
├── evaluation/
│   └── benchmark.py           # Evaluation scripts
└── requirements.txt
```

## Methodology

### Phase 1: Taint Analysis
- Transforms Solidity code into semantic graphs (CFG + DFG)
- Identifies taint sources (weak entropy) and sinks (sensitive operations)
- Extracts all possible taint paths with 100-dimensional feature vectors

### Phase 2: RL-based Path Prioritization
- DQN agent learns to ANALYZE or SKIP paths based on risk assessment
- Dynamic pool management with priority scoring
- Hierarchical reward engineering with safety constraints
- Pattern registry for exploration guidance

### Phase 3: Vulnerability Localization
- Gradient-based attribution for node importance
- Graph propagation with centrality analysis
- Function-level and node-level identification

## Comparison with Existing Tools

| Method | F1 (Balanced) | F1 (Imbalanced) | Time |
|--------|---------------|-----------------|------|
| Slither | 0.450 | - | 22min |
| Mythril | 0.372 | - | >24h |
| TaintSentinel | 0.892 | 0.611 | 8.3h |
| RNVulDet | 0.662 | 0.360 | 5.5h |
| **SmartTaintRL** | **0.955** | **0.950** | **3h** |

## Dataset

Evaluated on **4,706 real-world contracts** from Ethereum mainnet:
- 423 vulnerable contracts
- 4,283 safe contracts
- 252,844 execution paths

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

⭐ **If you find this project useful, please consider giving it a star!**
