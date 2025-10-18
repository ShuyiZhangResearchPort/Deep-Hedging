# Deep-Hedging

A reinforcement learning framework for computing optimal dynamic hedging strategies on option portfolios. The agent learns to rebalance a portfolio of stocks, bonds, and options at each time step to minimize Greeks (delta, gamma, vega) subject to transaction costs.

## Quick Start
```bash
python train.py --config cfgs/<config_name>
```

## Installation
```bash
pip install torch numpy numba
```

## File Structure
```
Deep-Hedging/
├── cfgs/                          # Configuration files
│   ├── configDGTC.yaml            # Delta-Gamma with transaction costs
│   └── configDGVTC.yaml           # Delta-Gamma-Vega with transaction costs
│
├── models/                        # Pre-trained model weights
│   ├── non-uniform/
│   │   └── GBMLSTM_T.pth
│   └── uniform/
│       ├── GARCHLSTMD.pth
│       ├── GARCHLSTMKAGGLEDG.pth
│       └── GARCHLSTMKAGGLEDGV.pth
│
├── notebooks/                     # Jupyter notebooks and scripts
│   ├── RLHNDGV.ipynb
│   └── deep_training.py
│
├── src/                           # Core source code
│   ├── agents/
│   │   ├── __init__.py
│   │   └── policy_net_garch.py    # RL policy network
│   │
│   ├── option_greek/
│   │   ├── __init__.py
│   │   ├── precompute.py          # Heston-Nandi coefficient precomputation
│   │   └── pricing.py             # Option pricing and Greeks
│   │
│   ├── simulation/
│   │   └── hedging_sim.py         # Hedging environment and dynamics
│   │
│   └── visualization/
│       ├── __init__.py
│       └── plot_results.py        # Result visualization
│
├── train.py                       # Main training script
├── LICENSE
└── README.md
```

## Configuration

Two configuration templates are provided:
- **configDGTC.yaml** — Delta-Gamma hedging with transaction costs
- **configDGVTC.yaml** — Delta-Gamma-Vega hedging with transaction costs

## Models

Pre-trained weights are available in `models/`:
- **uniform/** — Models trained on uniform time grids with Heston-Nandi dynamics
- **non-uniform/** — Models trained on non-uniform grids with GBM dynamics
