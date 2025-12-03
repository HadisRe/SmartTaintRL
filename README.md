# SmartTaintRL

**Efficient Detection and Localization of Bad Randomness Vulnerabilities in Ethereum Smart Contracts via Reinforcement Learning**

## Overview

SmartTaintRL is a framework that combines taint analysis with deep reinforcement learning to detect and localize Bad Randomness vulnerabilities in Ethereum smart contracts. Unlike traditional approaches that rely on pattern matching or exhaustive path exploration, our method uses a Deep Q-Network agent to intelligently prioritize high-risk execution paths.

### Key Features

- Achieves F1-score of 0.955 on balanced datasets and 0.950 on imbalanced ones
- Reduces search space by 45% while maintaining 96% recall
- Provides 92.9% function-level accuracy in identifying vulnerable code
- Shows less than 1% F1 degradation under real-world class imbalance (95:5 ratio)

## Architecture

The framework consists of three phases:

**Phase 1 - Taint Analysis:** Transforms Solidity code into semantic graphs combining CFG and DFG, identifies taint sources (weak entropy) and sinks (sensitive operations), and extracts all possible taint paths with 100-dimensional feature vectors.

**Phase 2 - RL-based Path Prioritization:** A DQN agent learns to ANALYZE or SKIP paths based on risk assessment. It uses dynamic pool management with priority scoring, hierarchical reward engineering with safety constraints, and a pattern registry for exploration guidance.

**Phase 3 - Vulnerability Localization:** Uses gradient-based attribution for node importance combined with graph propagation and centrality analysis to identify vulnerabilities at both function and node levels.

## Bad Randomness Vulnerability

Bad Randomness is ranked as the 4th most critical smart contract vulnerability by OWASP. Blockchain's deterministic nature prevents true random number generation, and developers often misuse predictable values like `block.timestamp`, `block.number`, `blockhash`, and `block.difficulty`.

These vulnerabilities have caused significant financial losses, including the SmartBillions hack (400 ETH) and Fomo3D exploit (over $3M).

## Installation

```bash
git clone https://github.com/YourUsername/SmartTaintRL.git
cd SmartTaintRL

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

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

## Comparison with Existing Tools

| Method | F1 (Balanced) | F1 (Imbalanced) | Time |
|--------|---------------|-----------------|------|
| Slither | 0.450 | - | 22min |
| Mythril | 0.372 | - | >24h |
| TaintSentinel | 0.892 | 0.611 | 8.3h |
| RNVulDet | 0.662 | 0.360 | 5.5h |
| SmartTaintRL | 0.955 | 0.950 | 3h |

## Dataset

Evaluated on 4,706 real-world contracts from Ethereum mainnet:
- 423 vulnerable contracts
- 4,283 safe contracts
- 252,844 execution paths

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome. Feel free to submit a Pull Request.
