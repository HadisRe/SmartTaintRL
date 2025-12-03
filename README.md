# SmartTaintRL

A framework for detecting Bad Randomness vulnerabilities in Ethereum smart contracts using reinforcement learning and taint analysis.

## What is this?

This project addresses a critical security issue in smart contracts: **Bad Randomness**. Many Ethereum contracts (especially gambling, lottery, and gaming apps) use predictable blockchain values like `block.timestamp` or `blockhash` for random number generation. Attackers can exploit this to predict outcomes and steal funds.

We combine taint analysis with a DQN-based agent that learns which execution paths are worth analyzing. Instead of checking every single path (which gets expensive fast), the agent prioritizes suspicious ones.

## Main idea

The system works in three stages:

1. **Taint analysis** - We build a semantic graph from the contract, mark entropy sources and sensitive sinks, then extract all taint paths with their features.

2. **Path prioritization** - A Deep Q-Network agent decides for each path: analyze it or skip it? The agent is trained with a reward function that considers source risk, sink criticality, and protection mechanisms.

3. **Localization** - Once we find a vulnerable contract, we pinpoint the exact function and nodes responsible using gradient-based attribution.

## Results

We tested on 4,706 contracts from Ethereum mainnet:

| Method | F1 (balanced) | F1 (imbalanced) |
|--------|--------------|-----------------|
| Slither | 0.450 | - |
| Mythril | 0.372 | - |
| TaintSentinel | 0.892 | 0.611 |
| RNVulDet | 0.662 | 0.360 |
| Ours | 0.955 | 0.950 |

The agent prunes about 45% of paths while keeping 96% recall. Localization accuracy is 92.9% at function level.

## Repository structure

```
SmartTaintRL/
├── src/                    # main code
│   ├── taint_analyzer.py
│   ├── feature_extractor.py
│   ├── dqn_agent.py
│   └── localizer.py
├── models/                 # trained model weights
├── data/                   # datasets
└── evaluation/             # evaluation scripts
```

## Setup

```bash
git clone https://github.com/YourUsername/SmartTaintRL.git
cd SmartTaintRL
pip install -r requirements.txt
```

Requires Python 3.8+, PyTorch, NetworkX.

## Why Bad Randomness matters

This vulnerability is ranked 4th in OWASP Smart Contract Top 10. Real attacks include:
- SmartBillions: 400 ETH stolen
- Fomo3D: over $3M lost

## License

MIT
