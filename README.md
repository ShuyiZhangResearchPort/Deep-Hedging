Deep Hedging
A reinforcement learning framework for computing optimal dynamic hedging strategies on option portfolios. The agent learns to rebalance a portfolio of stocks, bonds, and options at each time step to minimize Greeks (delta, gamma, vega) subject to transaction costs.
Training
bashpython train.py --config cfgs/<config_name>
See cfgs/ for configuration options.
