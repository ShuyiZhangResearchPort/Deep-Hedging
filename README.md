<!DOCTYPE html>
<html>
<head>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        h2 {
            font-size: 1.8em;
            margin-top: 30px;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }
        .description {
            font-size: 1.1em;
            color: #555;
            margin-bottom: 20px;
        }
        pre {
            background: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            border-left: 3px solid #007acc;
        }
        code {
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.95em;
        }
        .file-tree {
            background: #f9f9f9;
            padding: 20px;
            border-radius: 5px;
            font-family: 'Monaco', monospace;
            font-size: 0.9em;
            line-height: 1.5;
            overflow-x: auto;
        }
        .file-comment {
            color: #666;
            margin-left: 20px;
        }
    </style>
</head>
<body>

<h1>Deep-Hedging</h1>

<p class="description">
    A reinforcement learning framework for computing optimal dynamic hedging strategies on option portfolios. 
    The agent learns to rebalance a portfolio of stocks, bonds, and options at each time step to minimize Greeks 
    (delta, gamma, vega) subject to transaction costs.
</p>

<h2>Quick Start</h2>
<pre><code>python train.py --config cfgs/&lt;config_name&gt;</code></pre>

<h2>Installation</h2>
<pre><code>pip install torch numpy numba</code></pre>

<h2>File Structure</h2>

<div class="file-tree">
Deep-Hedging/<br>
├── cfgs/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="file-comment"># Configuration files</span><br>
│&nbsp;&nbsp;&nbsp; ├── configDGTC.yaml&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="file-comment"># Delta-Gamma with transaction costs</span><br>
│&nbsp;&nbsp;&nbsp; └── configDGVTC.yaml&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="file-comment"># Delta-Gamma-Vega with transaction costs</span><br>
│<br>
├── models/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="file-comment"># Pre-trained model weights</span><br>
│&nbsp;&nbsp;&nbsp; ├── non-uniform/<br>
│&nbsp;&nbsp;&nbsp; │&nbsp;&nbsp;&nbsp; └── GBMLSTM_T.pth<br>
│&nbsp;&nbsp;&nbsp; └── uniform/<br>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ├── GARCHLSTMD.pth<br>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ├── GARCHLSTMKAGGLEDG.pth<br>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └── GARCHLSTMKAGGLEDGV.pth<br>
│<br>
├── notebooks/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="file-comment"># Jupyter notebooks and scripts</span><br>
│&nbsp;&nbsp;&nbsp; ├── RLHNDGV.ipynb<br>
│&nbsp;&nbsp;&nbsp; └── deep_training.py<br>
│<br>
├── src/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="file-comment"># Core source code</span><br>
│&nbsp;&nbsp;&nbsp; ├── agents/<br>
│&nbsp;&nbsp;&nbsp; │&nbsp;&nbsp;&nbsp; ├── __init__.py<br>
│&nbsp;&nbsp;&nbsp; │&nbsp;&nbsp;&nbsp; └── policy_net_garch.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="file-comment"># RL policy network</span><br>
│&nbsp;&nbsp;&nbsp; │<br>
│&nbsp;&nbsp;&nbsp; ├── option_greek/<br>
│&nbsp;&nbsp;&nbsp; │&nbsp;&nbsp;&nbsp; ├── __init__.py<br>
│&nbsp;&nbsp;&nbsp; │&nbsp;&nbsp;&nbsp; ├── precompute.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="file-comment"># Heston-Nandi coefficient precomputation</span><br>
│&nbsp;&nbsp;&nbsp; │&nbsp;&nbsp;&nbsp; └── pricing.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="file-comment"># Option pricing and Greeks</span><br>
│&nbsp;&nbsp;&nbsp; │<br>
│&nbsp;&nbsp;&nbsp; ├── simulation/<br>
│&nbsp;&nbsp;&nbsp; │&nbsp;&nbsp;&nbsp; └── hedging_sim.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="file-comment"># Hedging environment and dynamics</span><br>
│&nbsp;&nbsp;&nbsp; │<br>
│&nbsp;&nbsp;&nbsp; └── visualization/<br>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ├── __init__.py<br>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └── plot_results.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="file-comment"># Result visualization</span><br>
│<br>
├── train.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="file-comment"># Main training script</span><br>
├── LICENSE<br>
└── README.md
</div>

<h2>Configuration</h2>
<p>Two configuration templates are provided:</p>
<ul>
    <li><strong>configDGTC.yaml</strong> — Delta-Gamma hedging with transaction costs</li>
    <li><strong>configDGVTC.yaml</strong> — Delta-Gamma-Vega hedging with transaction costs</li>
</ul>

<h2>Models</h2>
<p>Pre-trained weights are available in <code>models/</code>:</p>
<ul>
    <li><strong>uniform/</strong> — Models trained on uniform time grids with Heston-Nandi dynamics</li>
    <li><strong>non-uniform/</strong> — Models trained on non-uniform grids with GBM dynamics</li>
</ul>

</body>
</html>
