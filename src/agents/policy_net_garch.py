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
    vega_precomputed_analytical,
    theta_precomputed_analytical
)

logger = logging.getLogger(__name__)


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
                 precomputed_data_2_5yr=None, n_hedging_instruments=2, 
                 dt_min=1e-10, device="cpu", 
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
        self.precomputed_data_2_5yr = precomputed_data_2_5yr

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
        elif n_hedging_instruments == 4:
            self.instruments_maturities = [252, 378, 504, 630]
            if instrument_strikes is None:
                self.instrument_strikes = [None, self.sim.K, self.sim.K, self.sim.K]  # Default ATM
            else:
                self.instrument_strikes = [None, instrument_strikes[0], instrument_strikes[1], instrument_strikes[2]
        else:
            raise ValueError(f"n_hedging_instruments must be 1, 2, 3, or 4, got {n_hedging_instruments}")
        

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
                option_type=self.option_type, precomputed_data=self.precomputed_data_1yr,
                omega=self.omega, alpha=self.alpha, beta=self.beta, 
                gamma_param=self.gamma, lambda_=self.lambda_
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
                S=S_t, K=self.K, step_idx=t, r_daily=self.r, N=self.N, 
                omega=self.omega, alpha=self.alpha, beta=self.beta, 
                gamma_param=self.gamma, lambda_=self.lambda_, sigma0=self.sigma0,
                option_type=self.option_type, precomputed_data=self.precomputed_data_1yr
            )
            HN_vega_trajectory[:, t] = V_t

        return HN_vega_trajectory
    def compute_all_paths_hn_theta(self, S_trajecty):
        """Compute Heston-Nandi vega for ALL paths."""
        M, N_plus_1 = S_trajectory.shape
        HN_theta_trajectory = torch.zeros((M, N_plus_1), dtype=torch.float32, device=self.device)

        for t in range(N_plus_1):
            S_t = S_trajectory[:, t]
            V_t = theta_precomputed_analytical(
                S=S_t, K=self.K, step_idx=t, r_daily=self.r, N=self.N, 
                omega=self.omega, alpha=self.alpha, beta=self.beta, 
                gamma_param=self.gamma, lambda_=self.lambda_, sigma0=self.sigma0,
                option_type=self.option_type, precomputed_data=self.precomputed_data_1yr
            )
            HN_theta_trajectory[:, t] = V_t

        return HN_theta_trajectory

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
        elif n == 4:
            greek_names = ['delta', 'gamma', 'vega', 'theta']

        # Compute greeks for all hedging instruments
        instrument_greeks = []

        for j, maturity_days in enumerate(self.instrument_maturities):
            if maturity_days == 252:  # Stock
                greeks = {
                    'delta': torch.ones((M, N_plus_1), device=self.device),
                    'gamma': torch.zeros((M, N_plus_1), device=self.device),
                    'vega': torch.zeros((M, N_plus_1), device=self.device),
                    'theta': torch.zeros((M, N_plus_1), device=self.device)
                }
            else:  # Options
                if maturity_days == 378:
                    precomputed = self.precomputed_data_1_5yr
                elif maturity_days == 504:
                    precomputed = self.precomputed_data_2yr
                elif maturity_days = 630:
                    precomputed = self.precomputed_data_2_5yr
                else:
                    raise ValueError(f"No precomputed data for maturity {maturity_days}")

                # GET THE OPTION TYPE FOR THIS INSTRUMENT
                inst_option_type = self.instrument_types[j]

                delta_inst = torch.zeros((M, N_plus_1), device=self.device)
                gamma_inst = torch.zeros((M, N_plus_1), device=self.device)
                vega_inst = torch.zeros((M, N_plus_1), device=self.device)
                theta_inst = torch.zeros((M, N_plus_1), device=self.device)

                for t in range(N_plus_1):
                    S_t = S_trajectory[:, t]

                    delta_inst[:, t] = delta_precomputed_analytical(
                        S=S_t, K=self.instrument_strikes[j], step_idx=t, r_daily=self.r, N=maturity_days,
                        option_type=inst_option_type,
                        precomputed_data=precomputed
                    )

                    gamma_inst[:, t] = gamma_precomputed_analytical(
                        S=S_t, K=self.instrument_strikes[j], step_idx=t, r_daily=self.r, N=maturity_days,
                        option_type=inst_option_type,
                        precomputed_data=precomputed,
                        omega=self.omega, alpha=self.alpha, beta=self.beta,
                        gamma_param=self.gamma, lambda_=self.lambda_
                    )

                    
                    vega_inst[:, t] = vega_precomputed_analytical(
                        S=S_t, K=self.instrument_strikes[j], step_idx=t, r_daily=self.r, N=maturity_days, 
                        omega=self.omega, alpha=self.alpha, beta=self.beta, 
                        gamma_param=self.gamma, lambda_=self.lambda_, sigma0=self.sigma0,
                        option_type=inst_option_type,
                        precomputed_data=precomputed
                    )
                    if n >= 4:
                    theta_inst[:, t] = theta_precomputed_analytical(
                        S=S_t, K=self.instrument_strikes[j], step_idx=t, r_daily=self.r, N=maturity_days, 
                        omega=self.omega, alpha=self.alpha, beta=self.beta, 
                        gamma_param=self.gamma, lambda_=self.lambda_, sigma0=self.sigma0,
                        option_type=inst_option_type,
                        precomputed_data=precomputed
                    )
                greeks = {
                    'delta': delta_inst,
                    'gamma': gamma_inst,
                    'vega': vega_inst
                    'theta': theta_inst
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

        # Compute condition number for monitoring
        condition_numbers = torch.linalg.cond(A)
        max_cond = condition_numbers.max().item()
        mean_cond = condition_numbers.mean().item()
        pct_singular = (condition_numbers > 1e6).float().mean().item() * 100



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

        # Get initial positions from policy - FIXED
        lstm_out, hidden_state = policy_net.lstm(obs_t)
        x = lstm_out
        for fc in policy_net.fc_layers:
            x = F.relu(fc(x))

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

            # Get new positions - FIXED
            lstm_out, hidden_state = policy_net.lstm(obs_new, hidden_state)
            x = lstm_out
            for fc in policy_net.fc_layers:
                x = F.relu(fc(x))

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
                    multiplier = 10

                # Transaction costs, inportant for an option contract.
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
