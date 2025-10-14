import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, Any, Optional

# Your option pricing functions (adjust path as needed)
from src.option_greek.pricing import (
    price_option_precomputed,
    delta_precomputed_analytical,
    gamma_precomputed_analytical,
    vega_precomputed_analytical
)
class PolicyNetGARCH(nn.Module):
    def __init__(self, obs_dim=5, hidden_size=128, n_hedging_instruments=2, num_layers=2):
        super().__init__()

        self.n_hedging_instruments = n_hedging_instruments

        # LSTM to process the observation sequence
        self.lstm = nn.LSTM(obs_dim, hidden_size, num_layers=2, batch_first=True)

        # Create multiple FC layers dynamically
        self.fc_layers = nn.ModuleList()
        in_dim = hidden_size
        for _ in range(num_layers):
            self.fc_layers.append(nn.Linear(in_dim, hidden_size))
            in_dim = hidden_size

        # Create output heads dynamically for each instrument
        self.instrument_heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(n_hedging_instruments)
        ])

        # Initialize weights
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, obs_sequence):
        lstm_out, _ = self.lstm(obs_sequence)
        x = lstm_out

        # Pass through all FC layers
        for fc in self.fc_layers:
            x = F.relu(fc(x))

        # Get positions for all instruments
        outputs = [head(x).squeeze(-1) for head in self.instrument_heads]

        return outputs

# --------------------------------------------------------------------------
# Hedging Environment (Batched Trajectory Simulation)
# --------------------------------------------------------------------------
class HedgingEnvGARCH:
    def __init__(self, sim, garch_params=None, precomputed_data_1yr=None,
                 precomputed_data_1_5yr=None, precomputed_data_2yr=None,
                 n_hedging_instruments=2, dt_min=1e-10, device="cpu",
                 instrument_strikes=None, instrument_types=None):
        self.sim = sim
        self.M = sim.M
        self.N = sim.N
        self.dt_min = dt_min
        self.device = torch.device(device)

        self.garch_params = garch_params or {
            "omega": 1.593749e-07,
            "alpha": 2.308475e-06,
            "beta": 0.689984,
            "gamma": 342.870019,
            "lambda": 0.420499,
            "sigma0": sim.sigma,
        }

        self.n_hedging_instruments = n_hedging_instruments
        self.precomputed_data_1yr = precomputed_data_1yr
        self.precomputed_data_1_5yr = precomputed_data_1_5yr
        self.precomputed_data_2yr = precomputed_data_2yr

        # Define instrument maturities based on n_hedging_instruments
        if n_hedging_instruments == 1:
            self.instrument_maturities = [252]  # Stock only
            self.instrument_strikes = [None]
        elif n_hedging_instruments == 2:
            self.instrument_maturities = [252, 504]  # Stock + 2yr
            if instrument_strikes is None:
                self.instrument_strikes = [None, self.sim.K]  # Default ATM
            else:
                self.instrument_strikes = [None, instrument_strikes[0]]
        elif n_hedging_instruments == 3:
            self.instrument_maturities = [252, 378, 504]  # Stock + 1.5yr + 2yr
            if instrument_strikes is None:
                self.instrument_strikes = [None, self.sim.K, self.sim.K]  # Default ATM
            else:
                self.instrument_strikes = [None, instrument_strikes[0], instrument_strikes[1]]
        else:
            raise ValueError(f"n_hedging_instruments must be 1, 2, or 3, got {n_hedging_instruments}")

        # Define instrument types (call/put) based on n_hedging_instruments
        if instrument_types is None:
            # Default: all hedging instruments are same type as the option we're hedging
            self.instrument_types = [None] + [self.sim.option_type] * (n_hedging_instruments - 1)
        else:
            # Validate input
            if len(instrument_types) != n_hedging_instruments - 1:
                raise ValueError(f"instrument_types must have length {n_hedging_instruments - 1}, got {len(instrument_types)}")
            self.instrument_types = [None] + instrument_types  # None for stock, then user-specified types

        self.omega = float(self.garch_params["omega"])
        self.alpha = float(self.garch_params["alpha"])
        self.beta = float(self.garch_params["beta"])
        self.gamma = float(self.garch_params["gamma"])
        self.lambda_ = float(self.garch_params["lambda"])

        self.S0 = self.sim.S0
        self.K = self.sim.K
        self.T = self.sim.T
        self.r = self.sim.r / 252.0
        self.option_type = self.sim.option_type
        self.side = self.sim.side
        self.contract_size = self.sim.contract_size
        self.TCP = getattr(self.sim, "TCP", 0.0)

        self.sigma0 = float(self.garch_params["sigma0"])
        self.sigma_t = torch.full((self.M,), self.sigma0, dtype=torch.float32, device=self.device)
        self.h_t = (self.sigma_t ** 2 / 252)

    def reset(self):
        self.Z = torch.randn((self.M, self.N), dtype=torch.float32, device=self.device)
        sigma0_annual = float(self.garch_params["sigma0"])
        self.sigma_t = torch.full((self.M,), sigma0_annual, dtype=torch.float32, device=self.device)
        self.h_t = (self.sigma_t ** 2 / 252)

    def compute_all_paths_hn_delta(self, S_trajectory):
        """Compute Heston-Nandi delta for ALL paths."""
        M, N_plus_1 = S_trajectory.shape
        HN_delta_trajectory = torch.zeros((M, N_plus_1), dtype=torch.float32, device=self.device)

        for t in range(N_plus_1):
            S_t = S_trajectory[:, t]
            D_t = delta_precomputed_analytical(
                S=S_t, K=self.K, step_idx=t, r_daily=self.r, N=self.N,
                option_type=self.option_type, precomputed_data=self.precomputed_data_1yr
            )
            HN_delta_trajectory[:, t] = D_t

        return HN_delta_trajectory

    def compute_all_paths_hn_gamma(self, S_trajectory):
        """Compute Heston-Nandi gamma for ALL paths."""
        M, N_plus_1 = S_trajectory.shape
        HN_gamma_trajectory = torch.zeros((M, N_plus_1), dtype=torch.float32, device=self.device)

        for t in range(N_plus_1):
            S_t = S_trajectory[:, t]
            G_t = gamma_precomputed_analytical(
                S=S_t, K=self.K, step_idx=t, r_daily=self.r, N=self.N,
                option_type=self.option_type, precomputed_data=self.precomputed_data_1yr
            )
            HN_gamma_trajectory[:, t] = G_t

        return HN_gamma_trajectory

    def compute_all_paths_hn_vega(self, S_trajectory):
        """Compute Heston-Nandi vega for ALL paths."""
        M, N_plus_1 = S_trajectory.shape
        HN_vega_trajectory = torch.zeros((M, N_plus_1), dtype=torch.float32, device=self.device)

        for t in range(N_plus_1):
            S_t = S_trajectory[:, t]
            V_t = vega_precomputed_analytical(
                S=S_t, K=self.K, step_idx=t, r_daily=self.r, N=self.N, omega=self.omega,
                alpha=self.alpha, beta=self.beta, gamma=self.gamma, lambda_=self.lambda_, sigma0=self.sigma0,
                option_type=self.option_type, precomputed_data=self.precomputed_data_1yr
            )
            HN_vega_trajectory[:, t] = V_t

        return HN_vega_trajectory

    def compute_hn_option_positions(self, S_trajectory, portfolio_greeks):
        """
        Compute optimal hedge positions using linear algebra for n Greeks with n instruments.

        Solves the system:
            A @ x = b
        where A[i,j] = greek_i of instrument_j

        Args:
            S_trajectory: [M, N+1] stock prices
            portfolio_greeks: dict with keys like 'delta', 'gamma', 'vega' containing [M, N+1] tensors

        Returns:
            positions: [M, N+1, n_instruments] positions for each instrument
        """
        M, N_plus_1 = S_trajectory.shape
        n = self.n_hedging_instruments
        epsilon = 1e-10

        # Define which greeks to hedge based on n
        if n == 1:
            greek_names = ['delta']
        elif n == 2:
            greek_names = ['delta', 'gamma']
        elif n == 3:
            greek_names = ['delta', 'gamma', 'vega']

        # Compute greeks for all hedging instruments
        instrument_greeks = []

        for j, maturity_days in enumerate(self.instrument_maturities):
            if maturity_days == 252:  # Stock
                greeks = {
                    'delta': torch.ones((M, N_plus_1), device=self.device),
                    'gamma': torch.zeros((M, N_plus_1), device=self.device),
                    'vega': torch.zeros((M, N_plus_1), device=self.device),
                }
            else:  # Options
                if maturity_days == 378:
                    precomputed = self.precomputed_data_1_5yr
                elif maturity_days == 504:
                    precomputed = self.precomputed_data_2yr
                else:
                    raise ValueError(f"No precomputed data for maturity {maturity_days}")

                # GET THE OPTION TYPE FOR THIS INSTRUMENT
                inst_option_type = self.instrument_types[j]

                delta_inst = torch.zeros((M, N_plus_1), device=self.device)
                gamma_inst = torch.zeros((M, N_plus_1), device=self.device)
                vega_inst = torch.zeros((M, N_plus_1), device=self.device)

                for t in range(N_plus_1):
                    S_t = S_trajectory[:, t]

                    delta_inst[:, t] = delta_precomputed_analytical(
                        S=S_t, K=self.instrument_strikes[j], step_idx=t, r_daily=self.r, N=maturity_days,
                        option_type=inst_option_type,  # USE INSTRUMENT TYPE
                        precomputed_data=precomputed
                    )

                    gamma_inst[:, t] = gamma_precomputed_analytical(
                        S=S_t, K=self.instrument_strikes[j], step_idx=t, r_daily=self.r, N=maturity_days,
                        option_type=inst_option_type,  # USE INSTRUMENT TYPE
                        precomputed_data=precomputed
                    )

                    if n >= 3:
                        vega_inst[:, t] = vega_precomputed_analytical(
                            S=S_t, K=self.instrument_strikes[j], step_idx=t, r_daily=self.r, N=maturity_days, omega=self.omega,
                            alpha=self.alpha, beta=self.beta, gamma=self.gamma, lambda_=self.lambda_, sigma0=self.sigma0,
                            option_type=inst_option_type,  # USE INSTRUMENT TYPE
                            precomputed_data=precomputed
                        )

                greeks = {
                    'delta': delta_inst,
                    'gamma': gamma_inst,
                    'vega': vega_inst
                }

            instrument_greeks.append(greeks)

        # Build A matrix: [M, N+1, n, n]
        # A[i,j] = greek_i of instrument_j
        A = torch.zeros((M, N_plus_1, n, n), device=self.device)

        for i, greek_name in enumerate(greek_names):
            for j, inst_greeks in enumerate(instrument_greeks):
                A[:, :, i, j] = inst_greeks[greek_name]

        # Build b vector: [M, N+1, n]
        b = torch.stack([-portfolio_greeks[g] for g in greek_names], dim=-1)
        print(f"b: {b[0, 0, :]}")
        # Compute condition number for monitoring
        condition_numbers = torch.linalg.cond(A)
        max_cond = condition_numbers.max().item()
        mean_cond = condition_numbers.mean().item()
        pct_singular = (condition_numbers > 1e6).float().mean().item() * 100

        if max_cond > 1e3:
            logger.warning(
                f"High condition number: max={max_cond:.2e}, mean={mean_cond:.2e}, "
                f"{pct_singular:.1f}% effectively singular"
            )

        # Solve A @ x = b
        epsilon = 1e-12  # small number to avoid division by zero
        lambda_ = 1e-6    # ridge regularization

        # 1️⃣ Row-normalize each Greek across instruments
        # Compute norm along instrument axis for each Greek
        row_norm = A.norm(dim=-1, keepdim=True)  # [M, N+1, n_greeks, 1]
        A_scaled = A / (row_norm + epsilon)
        b_scaled = b / (row_norm[..., 0] + epsilon)  # scale b the same way

        # 2️⃣ Compute ridge least-squares solution per batch
        # x = (A^T A + λI)^(-1) A^T b
        # Batch dimensions: [M, N+1]
        M_size, N_size, n_greeks, n_instr = A.shape
        I = torch.eye(n_instr, device=A.device).reshape(1, 1, n_instr, n_instr)

        # Compute A^T A and A^T b
        ATA = torch.matmul(A_scaled.transpose(-2, -1), A_scaled)  # [M, N+1, n_instr, n_instr]
        ATb = torch.matmul(A_scaled.transpose(-2, -1), b_scaled.unsqueeze(-1))  # [M, N+1, n_instr, 1]

        # Solve (ATA + λI) x = ATb
        x = torch.linalg.solve(ATA + lambda_ * I, ATb)  # [M, N+1, n_instr, 1]
        x = x.squeeze(-1)  # [M, N+1, n_instr]
        try:
            x = torch.linalg.solve(A, b)  # [M, N+1, n]
            print(f"first path: {x[0]}")
        except RuntimeError as e:
            logger.error(f"Matrix inversion failed: {e}. Using delta-only fallback.")
            x = torch.zeros((M, N_plus_1, n), device=self.device)
            x[:, :, 0] = -portfolio_greeks['delta']
            return x

        # Handle ill-conditioned matrices
        singular_mask = condition_numbers > 1e6
        if singular_mask.any():
            x_fallback = torch.zeros_like(x)
            x_fallback[:, :, 0] = -portfolio_greeks['delta']
            x = torch.where(singular_mask.unsqueeze(-1), x_fallback, x)

        return x  # [M, N+1, n_instruments]

    def simulate_trajectory_and_get_observations(self, policy_net):
        """
        Simulate trajectory using LSTM with proper hidden state management.

        Returns:
            S_trajectory: [M, N+1] stock prices
            V_trajectory: [M, N+1] 1-year option values
            O_trajectories: dict mapping maturity -> [M, N+1] option prices
            obs_sequence: [M, N+1, 5] observations (for logging)
            all_positions: [M, N+1, n_instruments] positions from policy network
        """
        S_trajectory = []
        V_trajectory = []
        O_trajectories = {mat: [] for mat in self.instrument_maturities[1:]}  # Exclude stock
        obs_list = []

        S_t = torch.full((self.M,), self.S0, dtype=torch.float32, device=self.device)
        h_t = self.h_t.clone()

        S_trajectory.append(S_t)

        # Price 1-year option
        V0 = price_option_precomputed(
            S=S_t, K=self.K, step_idx=0, r_daily=self.r, N=self.N,
            option_type=self.option_type, precomputed_data=self.precomputed_data_1yr
        )
        V_trajectory.append(V0)

        # Price all hedge instruments
        for i, maturity_days in enumerate(self.instrument_maturities[1:], start=1):
            if maturity_days == 378:
                precomputed = self.precomputed_data_1_5yr
            else:
                precomputed = self.precomputed_data_2yr

            # GET THE OPTION TYPE FOR THIS INSTRUMENT
            inst_option_type = self.instrument_types[i]

            O0 = price_option_precomputed(
                S=S_t, K=self.instrument_strikes[i], step_idx=0, r_daily=self.r, N=maturity_days,
                option_type=inst_option_type,  # USE INSTRUMENT TYPE
                precomputed_data=precomputed
            )
            O_trajectories[maturity_days].append(O0)

        # Initial observation
        obs_t = torch.zeros((self.M, 1, 5), dtype=torch.float32, device=self.device)
        obs_t[:, 0, 0] = 0.0
        obs_t[:, 0, 1] = S_t / self.K
        obs_t[:, 0, 2] = 0.5
        obs_t[:, 0, 3] = V0 / S_t
        obs_t[:, 0, 4] = self.side * V0
        obs_list.append(obs_t)

        # Get initial positions from policy
        lstm_out, hidden_state = policy_net.lstm(obs_t)
        x = F.relu(policy_net.fc1(lstm_out))

        outputs = []
        for i, head in enumerate(policy_net.instrument_heads):
            if i == 0:
                output = torch.sigmoid(head(x)).squeeze(-1)[:, 0]
            else:
                output = head(x).squeeze(-1)[:, 0]
            outputs.append(output)

        positions_t = torch.stack(outputs, dim=-1)  # [M, n_instruments]
        all_positions = [positions_t]

        for t in range(self.N):
            sqrt_h = torch.sqrt(h_t)
            h_t = self.omega + self.beta * h_t + self.alpha * (self.Z[:, t] - self.gamma * sqrt_h) ** 2
            h_t = torch.clamp(h_t, min=1e-12)

            r_t = (self.r + self.lambda_ * h_t - 0.5 * h_t) + torch.sqrt(h_t) * self.Z[:, t]
            S_t = S_t * torch.exp(r_t)

            # Price 1-year option
            V_t = price_option_precomputed(
                S=S_t, K=self.K, step_idx=t+1, r_daily=self.r, N=self.N,
                option_type=self.option_type, precomputed_data=self.precomputed_data_1yr
            )

            # Price all hedge instruments
            for i, maturity_days in enumerate(self.instrument_maturities[1:], start=1):
                if maturity_days == 378:
                    precomputed = self.precomputed_data_1_5yr
                else:
                    precomputed = self.precomputed_data_2yr

                # GET THE OPTION TYPE FOR THIS INSTRUMENT
                inst_option_type = self.instrument_types[i]

                O_t = price_option_precomputed(
                    S=S_t, K=self.instrument_strikes[i], step_idx=t+1, r_daily=self.r, N=maturity_days,
                    option_type=inst_option_type,  # USE INSTRUMENT TYPE
                    precomputed_data=precomputed
                )
                O_trajectories[maturity_days].append(O_t)

            S_trajectory.append(S_t)
            V_trajectory.append(V_t)

            time_val = (t + 1) / self.N
            obs_new = torch.zeros((self.M, 1, 5), dtype=torch.float32, device=self.device)
            obs_new[:, 0, 0] = time_val
            obs_new[:, 0, 1] = S_t / self.K
            obs_new[:, 0, 2] = positions_t[:, 0].detach()  # Previous stock delta
            obs_new[:, 0, 3] = V_t / S_t
            obs_new[:, 0, 4] = self.side * V_t
            obs_list.append(obs_new)

            lstm_out, hidden_state = policy_net.lstm(obs_new, hidden_state)
            x = F.relu(policy_net.fc1(lstm_out))

            outputs = []
            for i, head in enumerate(policy_net.instrument_heads):
                if i == 0:
                    output = torch.sigmoid(head(x)).squeeze(-1)[:, 0]
                else:
                    output = head(x).squeeze(-1)[:, 0]
                outputs.append(output)

            positions_t = torch.stack(outputs, dim=-1)
            all_positions.append(positions_t)

        S_trajectory = torch.stack(S_trajectory, dim=1)
        V_trajectory = torch.stack(V_trajectory, dim=1)

        # Stack option trajectories
        for maturity_days in O_trajectories:
            O_trajectories[maturity_days] = torch.stack(O_trajectories[maturity_days], dim=1)

        all_positions = torch.stack(all_positions, dim=1)  # [M, N+1, n_instruments]
        obs_sequence = torch.cat(obs_list, dim=1)

        return S_trajectory, V_trajectory, O_trajectories, obs_sequence, all_positions

    def simulate_full_trajectory(self, all_positions, O_trajectories):
        """
        Simulate full hedging trajectory with n instruments.

        Args:
            all_positions: [M, N+1, n_instruments] positions for all instruments
            O_trajectories: dict mapping maturity -> [M, N+1] option prices
        """
        S_t = torch.full((self.M,), self.S0, dtype=torch.float32, device=self.device)
        positions_t = all_positions[:, 0]  # [M, n_instruments]

        V0 = price_option_precomputed(
            S=S_t, K=self.K, step_idx=0, r_daily=self.r, N=self.N,
            option_type=self.option_type, precomputed_data=self.precomputed_data_1yr
        )

        # Initialize bank account: short 1-year option - all hedging positions
        B_t = self.side * V0 - positions_t[:, 0] * S_t
        for i, maturity_days in enumerate(self.instrument_maturities[1:], start=1):
            B_t -= positions_t[:, i] * O_trajectories[maturity_days][:, 0]

        h_t = self.h_t.clone()

        S_traj, B_traj = [S_t], [B_t]
        position_trajs = {i: [positions_t[:, i]] for i in range(self.n_hedging_instruments)}

        for t in range(self.N):
            sqrt_h = torch.sqrt(h_t)
            h_t = self.omega + self.beta * h_t + self.alpha * (self.Z[:, t] - self.gamma * sqrt_h) ** 2
            h_t = torch.clamp(h_t, min=1e-12)

            r_t = (self.r + self.lambda_ * h_t - 0.5 * h_t) + torch.sqrt(h_t) * self.Z[:, t]
            S_t = S_t * torch.exp(r_t)

            positions_new = all_positions[:, t+1]

            # Compute trades and transaction costs for all instruments
            dt = 1.0 / self.N
            B_t = B_t * torch.exp(torch.tensor(self.r * 252.0, device=self.device) * dt)

            for i in range(self.n_hedging_instruments):
                trade = positions_new[:, i] - positions_t[:, i]

                if i == 0:  # Stock
                    price = S_t
                    multiplier = 1
                else:  # Options
                    maturity = self.instrument_maturities[i]
                    price = O_trajectories[maturity][:, t+1]
                    multiplier = 100

                # Transaction costs
                cost = self.TCP * multiplier * torch.abs(trade) * price

                # Update bank account
                B_t = B_t - trade * price - cost

            positions_t = positions_new

            S_traj.append(S_t)
            B_traj.append(B_t)
            for i in range(self.n_hedging_instruments):
                position_trajs[i].append(positions_t[:, i])

        # Terminal payoff of 1-year option
        payoff = torch.clamp(S_t - self.K, min=0.0) if self.option_type.lower() == "call" \
                 else torch.clamp(self.K - S_t, min=0.0)
        payoff = payoff * self.contract_size

        # Terminal value includes: bank account + all instrument positions
        terminal_value = B_t + positions_t[:, 0] * S_t
        for i, maturity_days in enumerate(self.instrument_maturities[1:], start=1):
            terminal_value += positions_t[:, i] * O_trajectories[maturity_days][:, -1]

        terminal_error = terminal_value - self.side * payoff

        trajectories = {
            'S': torch.stack(S_traj, dim=1),
            'B': torch.stack(B_traj, dim=1),
            'positions': all_positions,
            'O': O_trajectories,
        }

        return terminal_error, trajectories


# --------------------------------------------------------------------------
# Training Loop
# --------------------------------------------------------------------------
def train_garch(
    HedgingSim,
    PolicyNetClass,
    HedgingEnvClass,
    episodes=100,
    gamma=0.9999,
    actor_lr=1e-4,
    weight_decay=1e-6,
    ent_coef=0.5,
    seed=5,
    device="cpu",
    n_hedging_instruments=2,
    perturb_scale=0.0,
    instrument_strikes=None,
    instrument_types=None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device_t = torch.device(device)

    garch_params = {
        "omega": 1.593749e-07,
        "alpha": 2.308475e-06,
        "beta": 0.689984,
        "gamma": 342.870019,
        "lambda": 0.420499,
        "sigma0": 0.127037,
    }

    logger.info("Precomputing characteristic functions for all required maturities...")

    # Precompute for 1-year option (252 days)
    precomputed_data_1yr = precompute_hn_coefficients(
        N=252, r_daily=0.04 / 252,
        omega=garch_params["omega"], alpha=garch_params["alpha"],
        beta=garch_params["beta"], gamma=garch_params["gamma"],
        lambda_=garch_params["lambda"],
        N_quad=128, u_max=100.0, device=device
    )

    # Precompute for 1.5-year option (378 days) if needed
    precomputed_data_1_5yr = None
    if n_hedging_instruments >= 3:
        precomputed_data_1_5yr = precompute_hn_coefficients(
            N=378, r_daily=0.04 / 252,
            omega=garch_params["omega"], alpha=garch_params["alpha"],
            beta=garch_params["beta"], gamma=garch_params["gamma"],
            lambda_=garch_params["lambda"],
            N_quad=128, u_max=100.0, device=device
        )

    # Precompute for 2-year option (504 days)
    precomputed_data_2yr = precompute_hn_coefficients(
        N=504, r_daily=0.04 / 252,
        omega=garch_params["omega"], alpha=garch_params["alpha"],
        beta=garch_params["beta"], gamma=garch_params["gamma"],
        lambda_=garch_params["lambda"],
        N_quad=128, u_max=100.0, device=device
    )

    logger.info("Precomputation complete.")

    policy_net = PolicyNetClass(
        obs_dim=5, hidden_size=128,
        n_hedging_instruments=n_hedging_instruments
    ).to(device_t)
    opt = torch.optim.AdamW(policy_net.parameters(), lr=actor_lr, weight_decay=weight_decay)

    logger.info("Starting training_garch: episodes=%d, n_hedging_instruments=%d, device=%s",
                episodes, n_hedging_instruments, device)

    for episode in range(1, episodes + 1):
        print(f"episode: {episode}")
        sim = HedgingSim(
            S0=100, K=100, m=0.1, r=0.04, sigma=0.127037, T=1.0,
            option_type="call", position="short", M=1000, N=252, TCP=0, seed=episode
        )
        env = HedgingEnvClass(
            sim, garch_params=garch_params, device=device,
            precomputed_data_1yr=precomputed_data_1yr,
            precomputed_data_1_5yr=precomputed_data_1_5yr,
            precomputed_data_2yr=precomputed_data_2yr,
            n_hedging_instruments=n_hedging_instruments,
            instrument_strikes=instrument_strikes,
            instrument_types=instrument_types
        )
        env.reset()

        try:
            S_traj, V_traj, O_traj, obs_sequence, RL_positions = env.simulate_trajectory_and_get_observations(policy_net)
            terminal_errors, trajectories = env.simulate_full_trajectory(RL_positions, O_traj)

            # --- FULL GRADIENT DESCENT ---
            opt.zero_grad()

            # Compute loss on all terminal errors at once
            loss = torch.abs(terminal_errors).mean()

            # Backpropagate
            loss.backward()

            # Gradient clipping + optimizer step
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
            opt.step()

            if torch.isnan(loss) or torch.isinf(loss):
                logger.error("Loss became NaN/Inf")
                raise RuntimeError("Loss became NaN/Inf")

            if episode % 10 == 0:
                checkpoint_name = f"policy_net_mc_garch_lstm_{n_hedging_instruments}inst.pth"
                torch.save(policy_net.state_dict(), checkpoint_name)
                logger.info("Checkpoint overwritten at episode %d: %s", episode, checkpoint_name)

            final_reward = -float(loss.item())

            logger.info(
                "Episode %d | Final Reward: %.6f | Total Loss: %.6f",
                episode, final_reward, loss.item()
            )

            if episode % 1 == 0:
                try:
                    path_idx = 4
                    S_final = trajectories['S'][:, -1]
                    payoff = torch.clamp(S_final - env.K, min=0.0) if env.option_type.lower() == "call" \
                            else torch.clamp(env.K - S_final, min=0.0)
                    payoff = payoff * env.contract_size

                    # RL performance
                    terminal_value_rl = trajectories['B'][:, -1] + RL_positions[:, -1, 0] * S_final
                    for i, maturity in enumerate(env.instrument_maturities[1:], start=1):
                        terminal_value_rl += RL_positions[:, -1, i] * O_traj[maturity][:, -1]

                    terminal_hedge_error_rl = (terminal_value_rl - env.side * payoff).cpu().detach().numpy()

                    mse_rl = float(np.mean(terminal_hedge_error_rl ** 2))
                    smse_rl = mse_rl / (env.S0 ** 2)
                    cvar_95_rl = float(np.mean(np.sort(terminal_hedge_error_rl ** 2)[-int(0.05 * env.M):]))

                    # Compute HN benchmark
                    logger.info("Computing analytical HN hedge for all %d paths", env.M)

                    # Build portfolio greeks based on n_hedging_instruments
                    if n_hedging_instruments == 1:
                        HN_delta_all = env.compute_all_paths_hn_delta(S_traj)
                        portfolio_greeks = {
                            'delta': -env.side * HN_delta_all
                        }
                    elif n_hedging_instruments == 2:
                        HN_delta_all = env.compute_all_paths_hn_delta(S_traj)
                        HN_gamma_1yr = env.compute_all_paths_hn_gamma(S_traj)
                        portfolio_greeks = {
                            'delta': -env.side * HN_delta_all,
                            'gamma': -env.side * HN_gamma_1yr
                        }
                    elif n_hedging_instruments == 3:
                        HN_delta_all = env.compute_all_paths_hn_delta(S_traj)
                        HN_gamma_1yr = env.compute_all_paths_hn_gamma(S_traj)
                        HN_vega_1yr = env.compute_all_paths_hn_vega(S_traj)
                        portfolio_greeks = {
                            'delta': -env.side * HN_delta_all,
                            'gamma': -env.side * HN_gamma_1yr,
                            'vega': -env.side * HN_vega_1yr
                        }

                    # Compute HN positions
                    HN_positions_all = env.compute_hn_option_positions(S_traj, portfolio_greeks)

                    # Simulate HN strategy
                    _, trajectories_hn = env.simulate_full_trajectory(HN_positions_all, O_traj)

                    terminal_value_hn = trajectories_hn['B'][:, -1] + HN_positions_all[:, -1, 0] * S_final
                    for i, maturity in enumerate(env.instrument_maturities[1:], start=1):
                        terminal_value_hn += HN_positions_all[:, -1, i] * O_traj[maturity][:, -1]

                    terminal_hedge_error_hn = (terminal_value_hn - env.side * payoff).cpu().detach().numpy()

                    mse_hn = float(np.mean(terminal_hedge_error_hn ** 2))
                    smse_hn = mse_hn / (env.S0 ** 2)
                    cvar_95_hn = float(np.mean(np.sort(terminal_hedge_error_hn ** 2)[-int(0.05 * env.M):]))

                    # Plotting
                    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
                    time_steps = np.arange(env.N + 1)

                    # Extract sample paths for plotting
                    rl_positions_sample = RL_positions[path_idx].cpu().detach().numpy()
                    hn_positions_sample = HN_positions_all[path_idx].cpu().detach().numpy()

                    # Plot 1: Stock Delta Comparison
                    axes[0, 0].plot(time_steps, rl_positions_sample[:, 0], label='RL Delta',
                                  linewidth=2, color='tab:blue')
                    axes[0, 0].plot(time_steps, hn_positions_sample[:, 0], label='HN Delta (Practitioner)',
                                  linewidth=2, linestyle='--', alpha=0.8, color='tab:orange')
                    axes[0, 0].set_xlabel("Time Step", fontsize=11)
                    axes[0, 0].set_ylabel("Delta", fontsize=11)
                    axes[0, 0].set_title(f"Stock Delta: Practitioner vs RL (Path {path_idx})", fontsize=12)
                    axes[0, 0].legend(fontsize=10)
                    axes[0, 0].grid(True, alpha=0.3)

                    # Plot 2: Option Positions Comparison
                    if n_hedging_instruments >= 2:
                        for i in range(1, n_hedging_instruments):
                            maturity = env.instrument_maturities[i]
                            opt_type = env.instrument_types[i]
                            strike = env.instrument_strikes[i]
                            label_suffix = f'{maturity}d {opt_type.upper()} K={strike}'

                            axes[0, 1].plot(time_steps, rl_positions_sample[:, i],
                                          label=f'RL {label_suffix}', linewidth=2)
                            axes[0, 1].plot(time_steps, hn_positions_sample[:, i],
                                          label=f'HN {label_suffix}', linewidth=2,
                                          linestyle='--', alpha=0.8)
                        axes[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
                        axes[0, 1].set_xlabel("Time Step", fontsize=11)
                        axes[0, 1].set_ylabel("Option Contracts", fontsize=11)
                        axes[0, 1].set_title(f"Option Positions: Practitioner vs RL (Path {path_idx})", fontsize=12)
                        axes[0, 1].legend(fontsize=9)
                        axes[0, 1].grid(True, alpha=0.3)
                    else:
                        axes[0, 1].text(0.5, 0.5, 'No option positions\n(Delta hedge only)',
                                      ha='center', va='center', transform=axes[0, 1].transAxes)
                        axes[0, 1].set_title("Option Positions", fontsize=12)

                    # Plot 3: Stock Price Trajectory
                    axes[1, 0].plot(time_steps, S_traj[path_idx].cpu().detach().numpy(),
                                  label='Stock Price', color='tab:green', linewidth=2)
                    axes[1, 0].axhline(y=env.K, color='r', linestyle='--', label='Strike', alpha=0.7)
                    axes[1, 0].set_xlabel("Time Step", fontsize=11)
                    axes[1, 0].set_ylabel("Stock Price", fontsize=11)
                    axes[1, 0].set_title(f"Stock Price Trajectory (Path {path_idx})", fontsize=12)
                    axes[1, 0].legend(fontsize=10)
                    axes[1, 0].grid(True, alpha=0.3)

                    # Plot 4: Hedging Instrument Prices
                    for i, maturity in enumerate(env.instrument_maturities[1:], start=1):
                        opt_type = env.instrument_types[i]
                        strike = env.instrument_strikes[i]
                        axes[1, 1].plot(time_steps, O_traj[maturity][path_idx].cpu().detach().numpy(),
                                      label=f'{maturity}d {opt_type.upper()} K={strike}', linewidth=2)
                    axes[1, 1].set_xlabel("Time Step", fontsize=11)
                    axes[1, 1].set_ylabel("Option Price", fontsize=11)
                    axes[1, 1].set_title(f"Hedging Instrument Prices (Path {path_idx})", fontsize=12)
                    axes[1, 1].legend(fontsize=10)
                    axes[1, 1].grid(True, alpha=0.3)

                    # Plot 5: Position Difference (RL - Practitioner)
                    delta_diff = rl_positions_sample[:, 0] - hn_positions_sample[:, 0]
                    axes[2, 0].plot(time_steps, delta_diff, color='tab:red', linewidth=2)
                    axes[2, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
                    axes[2, 0].set_xlabel("Time Step", fontsize=11)
                    axes[2, 0].set_ylabel("Delta Difference", fontsize=11)
                    axes[2, 0].set_title(f"RL Delta - HN Delta (Path {path_idx})", fontsize=12)
                    axes[2, 0].grid(True, alpha=0.3)

                    # Plot 6: Terminal Error Distribution
                    axes[2, 1].hist(terminal_hedge_error_rl, bins=50, color="tab:blue", alpha=0.7,
                                  edgecolor='black', label='RL')
                    axes[2, 1].hist(terminal_hedge_error_hn, bins=50, color="tab:orange", alpha=0.7,
                                  edgecolor='black', label='HN (Practitioner)')
                    axes[2, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
                    axes[2, 1].set_xlabel("Terminal Hedge Error", fontsize=11)
                    axes[2, 1].set_ylabel("Frequency", fontsize=11)

                    greek_labels = {1: 'Delta', 2: 'Delta-Gamma', 3: 'Delta-Gamma-Vega'}
                    title_text = (f"Episode {episode} - {n_hedging_instruments} Instruments ({greek_labels[n_hedging_instruments]})\n"
                                f"RL: MSE={mse_rl:.4f} | SMSE={smse_rl:.6f} | CVaR95={cvar_95_rl:.4f}\n"
                                f"HN: MSE={mse_hn:.4f} | SMSE={smse_hn:.6f} | CVaR95={cvar_95_hn:.4f}")
                    axes[2, 1].set_title(title_text, fontsize=10)
                    axes[2, 1].legend(fontsize=10)
                    axes[2, 1].grid(True, alpha=0.3)

                    fig.tight_layout()
                    plt.savefig(f"hedge_comparison_{n_hedging_instruments}inst_ep{episode}.png",
                              dpi=150, bbox_inches='tight')
                    plt.show()

                    # Log statistics
                    delta_mae = np.mean(np.abs(delta_diff))
                    delta_rmse = np.sqrt(np.mean(delta_diff ** 2))

                    logger.info(
                        "Path %d Delta Statistics - MAE: %.6f | RMSE: %.6f",
                        path_idx, delta_mae, delta_rmse
                    )

                    if n_hedging_instruments >= 2:
                        for i in range(1, n_hedging_instruments):
                            position_diff = rl_positions_sample[:, i] - hn_positions_sample[:, i]
                            position_mae = np.mean(np.abs(position_diff))
                            position_rmse = np.sqrt(np.mean(position_diff ** 2))
                            logger.info(
                                "Path %d Instrument %d Position Statistics - MAE: %.6f | RMSE: %.6f",
                                path_idx, i, position_mae, position_rmse
                            )

                    logger.info(
                        "RL Performance: MSE=%.6f | SMSE=%.6f | CVaR95=%.6f",
                        mse_rl, smse_rl, cvar_95_rl
                    )
                    logger.info(
                        "HN Performance: MSE=%.6f | SMSE=%.6f | CVaR95=%.6f",
                        mse_hn, smse_hn, cvar_95_hn
                    )

                except Exception as e:
                    logger.warning("Plotting skipped due to %s", e)
        except Exception as exc:
            logger.exception("Error during episode %d: %s", episode, exc)
            raise

    out_name = f"policy_net_mc_garch_lstm_{n_hedging_instruments}inst.pth"
    torch.save(policy_net.state_dict(), out_name)
    logger.info("Training finished. Model saved to %s", out_name)

    return policy_net

