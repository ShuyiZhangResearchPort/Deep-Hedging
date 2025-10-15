import numpy as np
from scipy.stats import norm, skew, kurtosis
import matplotlib.pyplot as plt
from scipy.special import softmax
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from math import lcm
import numpy as np
import warnings
from scipy.integrate import quad
import numpy as np
from numpy import exp, log, real
from numpy.polynomial.legendre import leggauss
from typing import Optional
from numba import njit



import numpy as np
from numba import jit, prange
from numpy.polynomial.legendre import leggauss
from scipy.stats import norm
import sys
sys.path.append('/content')  # add current folder to Python path just in case

import importlib
importlib.reload(hn_utils)  # reload if you edited it

# Step 3: Import the functions
from src.option_greek.pricing import _fstar_hn_scalar, _fstar_hn_vectorized, _f_hn_scalar, _f_hn_vectorized

# Test

class HedgingSim:
    """
    Delta hedging for European call/put under Black-Scholes with proportional transaction costs.
    side = +1 short option, -1 long option.
    Stored D = unsigned delta * contract_size.
    Signed share position each step = side * D.
    Re-hedge trade (shares) = side * (newD - oldD).
    Final P&L = -side * payoff + B_T + side * D_T * S_T
    """

    def __init__(self, S0, K, m, r, sigma, T, option_type="call", position="long",
                M=5000, N=252, TCP=0.0001, contract_size=1, seed=42):
        self.S0 = S0
        self.K = K
        self.m = m
        self.r = r
        self.sigma = sigma        # initial annualized volatility
        self.T = T
        self.option_type = option_type.lower()
        self.position = position.lower()
        self.M = M
        self.N = N
        self.TCP = TCP
        self.contract_size = contract_size
        self.seed = seed
        self.dt = T / N
        self.side = 1 if self.position == "short" else -1
        if seed is not None:
            np.random.seed(seed)

        # -----------------------
        # GARCH(1,1) parameters
        # -----------------------
        self.omega = 1.593749e-07    # long-run variance component
        self.alpha = 2.308475e-06     # reaction to new shocks
        self.beta  =0.689984      # persistence
        self.gamma = 342.870019      # leverage effect (set 0 if unused)
        self.lambda_ = .420499
        self.sigma0 = sigma   # starting vol level

    def bs_price(self, S, K, T, r, sigma, option_type="call",  # basic black scholes function
                 q=0.0, intrinsic_at_expiry=True, min_time=1e-12, min_vol=1e-12):
        S = np.asarray(S, float); K = np.asarray(K, float)
        T = np.asarray(T, float); sigma = np.asarray(sigma, float)
        S, K, T, sigma = np.broadcast_arrays(S, K, T, sigma)
        out = np.empty_like(S)
        exp_mask = T <= min_time
        live_mask = ~exp_mask
        if np.any(exp_mask):
            call_intr = np.maximum(S[exp_mask] - K[exp_mask], 0.0)
            put_intr = np.maximum(K[exp_mask] - S[exp_mask], 0.0)
            if intrinsic_at_expiry:
                if option_type == "call":
                    out[exp_mask] = call_intr
                elif option_type == "put":
                    out[exp_mask] = put_intr
                else:
                    raise ValueError("option_type must be 'call' or 'put'")
            else:
                live_mask = np.ones_like(T, bool)
                exp_mask[:] = False
        if np.any(live_mask):
            T_eff = np.maximum(T[live_mask], min_time)
            sig_eff = np.maximum(sigma[live_mask], min_vol)
            sqrtT = np.sqrt(T_eff)
            d1 = (np.log(S[live_mask] / K[live_mask]) + (r - q + 0.5*sig_eff**2)*T_eff) / (sig_eff*sqrtT)
            d2 = d1 - sig_eff*sqrtT
            disc_r = np.exp(-r*T_eff); disc_q = np.exp(-q*T_eff)
            if option_type == "call":
                val = S[live_mask]*disc_q*norm.cdf(d1) - K[live_mask]*disc_r*norm.cdf(d2)
            else:
                val = K[live_mask]*disc_r*norm.cdf(-d2) - S[live_mask]*disc_q*norm.cdf(-d1)
            out[live_mask] = val
        return float(out) if out.shape == () else out


    def bs_delta(self, S, K, T, r, sigma, option_type="call", q=0.0, min_time=1e-12, min_vol=1e-12):
        """
        Vectorized Black-Scholes delta calculation.

        Parameters:
        -----------
        S : array_like
            Spot price(s)
        K : array_like
            Strike price(s)
        T : array_like
            Time to expiration (years)
        r : float
            Risk-free rate
        sigma : array_like
            Volatility
        option_type : str
            'call' or 'put'
        q : float
            Dividend yield (default 0.0)
        min_time : float
            Minimum time threshold (default 1e-12)
        min_vol : float
            Minimum volatility threshold (default 1e-12)

        Returns:
        --------
        float or ndarray
            Delta value(s)
        """
        # Convert to arrays and broadcast
        S, K, T, sigma = np.broadcast_arrays(
            np.asarray(S, float),
            np.asarray(K, float),
            np.asarray(T, float),
            np.asarray(sigma, float)
        )

        # Initialize output
        delta = np.zeros_like(S)

        # Handle expired options
        exp_mask = T <= min_time
        if np.any(exp_mask):
            if option_type == "call":
                delta[exp_mask] = (S[exp_mask] > K[exp_mask]).astype(float)
            else:
                delta[exp_mask] = -(S[exp_mask] < K[exp_mask]).astype(float)

        # Handle live options
        live_mask = ~exp_mask
        if np.any(live_mask):
            T_live = np.maximum(T[live_mask], min_time)
            sig_live = np.maximum(sigma[live_mask], min_vol)

            # Calculate d1
            d1 = (np.log(S[live_mask] / K[live_mask]) +
                  (r - q + 0.5 * sig_live**2) * T_live) / (sig_live * np.sqrt(T_live))

            # Calculate delta based on option type
            discount = np.exp(-q * T_live)
            if option_type == "call":
                delta[live_mask] = discount * norm.cdf(d1)
            else:
                delta[live_mask] = discount * (norm.cdf(d1) - 1.0)

        # Return scalar if input was scalar
        return float(delta) if delta.shape == () else delta


    def fstar_hn(self, phi, const, S, X, Time_inDays, r_daily, omega, alpha, beta, gamma, lambda_):
        S = np.atleast_1d(S)
        X = np.atleast_1d(X)
        Time_inDays = np.atleast_1d(Time_inDays)
        r_daily = np.atleast_1d(r_daily)

        shape = np.broadcast_shapes(S.shape, X.shape, Time_inDays.shape, r_daily.shape)
        S_bc = np.broadcast_to(S, shape)
        X_bc = np.broadcast_to(X, shape)
        Time_bc = np.broadcast_to(Time_inDays, shape)
        r_bc = np.broadcast_to(r_daily, shape)

        n_elements = int(np.prod(shape))  # CAST TO PYTHON INT
        S_flat = S_bc.flatten()
        X_flat = X_bc.flatten()
        Time_flat = Time_bc.flatten()
        r_flat = r_bc.flatten()

        result_flat = _fstar_hn_vectorized(phi, const, S_flat, X_flat, Time_flat, r_flat, omega, alpha, beta, gamma, lambda_, n_elements)
        result = result_flat.reshape(shape)

        return float(result) if result.shape == () else result

    def f_hn(self, phi, const, S, X, Time_inDays, r_daily, omega, alpha, beta, gamma, lambda_):
        S = np.atleast_1d(S)
        X = np.atleast_1d(X)
        Time_inDays = np.atleast_1d(Time_inDays)
        r_daily = np.atleast_1d(r_daily)

        shape = np.broadcast_shapes(S.shape, X.shape, Time_inDays.shape, r_daily.shape)
        S_bc = np.broadcast_to(S, shape)
        X_bc = np.broadcast_to(X, shape)
        Time_bc = np.broadcast_to(Time_inDays, shape)
        r_bc = np.broadcast_to(r_daily, shape)

        n_elements = int(np.prod(shape))  # CAST TO PYTHON INT
        S_flat = S_bc.flatten()
        X_flat = X_bc.flatten()
        Time_flat = Time_bc.flatten()
        r_flat = r_bc.flatten()

        result_flat = _f_hn_vectorized(phi, const, S_flat, X_flat, Time_flat, r_flat, omega, alpha, beta, gamma, lambda_, n_elements)
        result = result_flat.reshape(shape)

        return result if result.shape != () else result.item()

    def hn_price(self, S, X, Time_inDays, r_daily, omega, alpha, beta, gamma, lambda_, option_type="call", N_quad=128, u_max=100.0):
        u_nodes, w_nodes = leggauss(N_quad)
        u_nodes = 0.5 * (u_nodes + 1) * u_max
        w_nodes = 0.5 * u_max * w_nodes

        S = np.atleast_1d(S)
        X = np.atleast_1d(X)
        Time_inDays = np.atleast_1d(Time_inDays)
        r_daily = np.atleast_1d(r_daily)

        shape = np.broadcast_shapes(S.shape, X.shape, Time_inDays.shape, r_daily.shape)

        integrand1 = np.zeros((len(u_nodes),) + shape)
        integrand2 = np.zeros((len(u_nodes),) + shape)

        for i, u in enumerate(u_nodes):
            integrand1[i] = self.fstar_hn(u, 1.0, S, X, Time_inDays, r_daily, omega, alpha, beta, gamma, lambda_)
            integrand2[i] = self.fstar_hn(u, 0.0, S, X, Time_inDays, r_daily, omega, alpha, beta, gamma, lambda_)

        call1 = np.tensordot(w_nodes, integrand1, axes=([0], [0]))
        call2 = np.tensordot(w_nodes, integrand2, axes=([0], [0]))

        S_bc = np.broadcast_to(S, shape)
        X_bc = np.broadcast_to(X, shape)
        Time_bc = np.broadcast_to(Time_inDays, shape)
        r_bc = np.broadcast_to(r_daily, shape)

        disc = np.exp(-r_bc * Time_bc)
        call_price = S_bc / 2.0 + disc * call1 - X_bc * disc * (0.5 + call2)

        if option_type == "call":
            result = call_price
        elif option_type == "put":
            result = call_price - S_bc + X_bc * disc
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        return float(result) if result.shape == () else result


    def hn_delta(self, S, X, Time_inDays, r_daily, omega, alpha, beta, gamma, lambda_, option_type="call", N_quad=128, u_max=100.0):
        u_nodes, w_nodes = leggauss(N_quad)
        u_nodes = 0.5 * (u_nodes + 1) * u_max
        w_nodes = 0.5 * u_max * w_nodes

        S = np.atleast_1d(S)
        X = np.atleast_1d(X)
        Time_inDays = np.atleast_1d(Time_inDays)
        r_daily = np.atleast_1d(r_daily)

        shape = np.broadcast_shapes(S.shape, X.shape, Time_inDays.shape, r_daily.shape)
        S_bc = np.broadcast_to(S, shape)

        integrand1 = np.zeros((len(u_nodes),) + shape, dtype=np.complex128)
        integrand2 = np.zeros((len(u_nodes),) + shape, dtype=np.complex128)

        for i, u in enumerate(u_nodes):
            cphi1 = 1j * u + 1.0
            cphi0 = 1j * u
            integrand1[i] = cphi1 * self.f_hn(u, 1.0, S, X, Time_inDays, r_daily, omega, alpha, beta, gamma, lambda_) / S_bc
            integrand2[i] = cphi0 * self.f_hn(u, 0.0, S, X, Time_inDays, r_daily, omega, alpha, beta, gamma, lambda_) / S_bc

        delta1 = np.real(np.tensordot(w_nodes, integrand1, axes=([0], [0])))
        delta2 = np.real(np.tensordot(w_nodes, integrand2, axes=([0], [0])))

        X_bc = np.broadcast_to(X, shape)
        Time_bc = np.broadcast_to(Time_inDays, shape)
        r_bc = np.broadcast_to(r_daily, shape)
        disc = np.exp(-r_bc * Time_bc)

        delta_call = 0.5 + disc * delta1 - X_bc * disc * delta2

        result = delta_call if option_type == "call" else delta_call - 1.0
        return float(result) if result.shape == () else result
    def bs_gamma(self, S, K, T, r, sigma, option_type="call", q=0.0, min_time=1e-12, min_vol=1e-12):
        """
        Black-Scholes gamma: sensitivity of delta to changes in underlying price.
        Returns gamma per share (unsigned), compatible with array inputs.
        """
        S = np.asarray(S, float)
        K = np.asarray(K, float)
        T = np.asarray(T, float)
        sigma = np.asarray(sigma, float)
        S, K, T, sigma = np.broadcast_arrays(S, K, T, sigma)

        gamma = np.zeros_like(S)
        live_mask = T > min_time
        exp_mask = ~live_mask

        # At expiry: gamma is zero
        gamma[exp_mask] = 0.0

        if np.any(live_mask):
            T_eff = np.maximum(T[live_mask], min_time)
            sigma_eff = np.maximum(sigma[live_mask], min_vol)
            sqrtT = np.sqrt(T_eff)
            d1 = (np.log(S[live_mask]/K[live_mask]) + (r - q + 0.5*sigma_eff**2)*T_eff) / (sigma_eff*sqrtT)
            gamma[live_mask] = norm.pdf(d1) / (S[live_mask] * sigma_eff * sqrtT)

        return float(gamma) if gamma.shape == () else gamma


    def run(self):
        """Run the hedging strategy with M paths and N timesteps, using GARCH(1,1) dynamics and hn_delta"""
        M, N = self.M, self.N
        S0, K, T, r, dt = self.S0, self.K, self.T, self.r, self.dt
        opt_type = self.option_type
        mult = self.contract_size

        Z = np.random.standard_normal((M, N))
        P = np.zeros((M, N+1))
        S = np.zeros((M, N+1))
        S[:, 0] = S0
        D = np.zeros((M, N+1))
        VAR = np.zeros(N+1)
        B = np.zeros((M, N+1))
        C = np.zeros((M, N+1))

        # Initial GARCH variance (daily scale)
        h_t = np.full(M, (self.sigma0 ** 2) / 252.0)

        # Convert T (years) to days and r to daily rate
        Time_inDays = T * 252
        r_daily = r / 252

        # Initial delta
        D[:, 0] = self.side * self.hn_delta(S[:, 0], K, Time_inDays, r_daily,
                                            self.omega, self.alpha, self.beta,
                                            self.gamma, self.lambda_, opt_type) * mult

        # Initial price
        V0 = self.hn_price(S[:, 0], K, Time_inDays, r_daily,
                          self.omega, self.alpha, self.beta,
                          self.gamma, self.lambda_, opt_type) * mult

        B[:, 0] = self.side * V0 - D[:, 0] * S[:, 0]

        for t in range(1, N+1):
            tau = max(T - t*dt, 1e-6)
            tau_days = tau * 252

            # CORRECTED GARCH update - matches paper equation (1b)
            h_t = (self.omega
                  + self.beta * h_t
                  + self.alpha * (Z[:, t-1] - self.gamma * np.sqrt(h_t))**2)
            h_t = np.maximum(h_t, 1e-12)

            # Return increment - using physical measure
            # For risk-neutral simulation, use: r_daily + lambda_*h_t - 0.5*h_t
            r_t = (r_daily + self.lambda_ * h_t - 0.5 * h_t) + np.sqrt(h_t) * Z[:, t-1]
            S[:, t] = S[:, t-1] * np.exp(r_t)

            # Hedge update with hn_delta
            newD = self.hn_delta(S[:, t], K, tau_days, r_daily,
                                self.omega, self.alpha, self.beta,
                                self.gamma, self.lambda_, opt_type) * mult * self.side

            trade = newD - D[:, t-1]
            C[:, t] = self.TCP * np.abs(trade) * S[:, t]
            B[:, t] = B[:, t-1] * np.exp(r*dt) - trade * S[:, t] - C[:, t]
            P[:, t] = B[:, t] - B[:, t-1]
            VAR[t] = np.var(P[:, t])
            D[:, t] = newD

        self.S, self.D, self.B, self.P, self.C, self.VAR = S, D, B, P, C, VAR


    def unequal_run(self, proportions, chunks):
        """Run hedging with uneven hedge schedule under GARCH(1,1) dynamics, using hn_delta"""
        M, N = self.M, self.N
        S0, K, T, m, r = self.S0, self.K, self.T, self.m, self.r
        opt_type = self.option_type
        mult = self.contract_size

        # Integerize hedge allocations
        chunks_hedge = [int(w * N) for w in proportions]
        diff = N - sum(chunks_hedge)
        if diff > 0:
            for i in np.argsort(proportions)[-diff:]:
                chunks_hedge[i] += 1

        total_hedges = int(np.sum(chunks_hedge))
        if total_hedges <= 0:
            self.S = np.full((M, 1), S0, float)
            self.D = np.zeros((M, 1))
            self.B = np.zeros((M, 1))
            self.P = np.zeros((M, 1))
            self.C = np.zeros((M, 1))
            self.dt_schedule = {
                'dt_values': np.array([]),
                'time_points': np.array([]),
                'proportions': proportions,
                'chunks': chunks
            }
            return

        Z = np.random.standard_normal((M, total_hedges))
        P = np.zeros((M, total_hedges + 1))
        S = np.zeros((M, total_hedges + 1))
        D = np.zeros((M, total_hedges + 1))
        B = np.zeros((M, total_hedges + 1))
        C = np.zeros((M, total_hedges + 1))

        # Initial GARCH variance (daily)
        h_t = np.full(M, (self.sigma0 ** 2) / 252.0)

        # Convert to days
        Time_inDays = T * 252
        r_daily = r / 252

        # t=0 setup with hn_delta
        S[:, 0] = S0
        D[:, 0] = self.side * self.hn_delta(S[:, 0], K, Time_inDays, r_daily,
                                            self.omega, self.alpha, self.beta,
                                            self.gamma, self.lambda_, opt_type) * mult

        V0 = self.hn_price(S[:, 0], K, Time_inDays, r_daily, self.omega, self.alpha,
                          self.beta, self.gamma, self.lambda_, opt_type) * mult

        B[:, 0] = self.side * V0 - D[:, 0] * S[:, 0]

        dt = self.T / self.N
        hedge_idx, current_time = 1, 0.0
        dt_values, time_points = [], []

        for chunk_idx, chunk in enumerate(chunks):
            if not chunk:
                continue

            n_hedges = int(chunks_hedge[chunk_idx])
            if n_hedges <= 0:
                continue

            chunk_start, chunk_end = int(chunk[0]), int(chunk[-1])
            start_time, end_time = (chunk_start - 1) * dt, chunk_end * dt
            chunk_duration = max(end_time - start_time, dt)
            dt_hedge = chunk_duration / n_hedges
            current_time = start_time

            for j in range(1, n_hedges + 1):
                dt_gap = dt_hedge

                # GARCH update
                h_t = self.omega + self.alpha * (np.sqrt(h_t) * Z[:, hedge_idx-1])**2 + self.beta * h_t
                h_t = np.maximum(h_t, 1e-12)

                # Evolve stock
                r_t = (m - 0.5 * h_t) * dt_gap + np.sqrt(h_t * dt_gap) * Z[:, hedge_idx-1]
                S[:, hedge_idx] = S[:, hedge_idx - 1] * np.exp(r_t)

                current_time += dt_gap
                dt_values.append(dt_gap)
                time_points.append(current_time)

                tau = T - current_time
                tau_days = tau * 252
                newD = self.hn_delta(S[:, hedge_idx], K, tau_days, r_daily,
                                    self.omega, self.alpha, self.beta,
                                    self.gamma, self.lambda_, opt_type) * mult * self.side
                trade = newD - D[:, hedge_idx - 1]
                C[:, hedge_idx] = self.TCP * np.abs(trade) * S[:, hedge_idx]
                B[:, hedge_idx] = B[:, hedge_idx - 1] * np.exp(r * dt_gap) - trade * S[:, hedge_idx] - C[:, hedge_idx]
                P[:, hedge_idx] = B[:, hedge_idx] - B[:, hedge_idx - 1]
                D[:, hedge_idx] = newD
                hedge_idx += 1

        self.S, self.D, self.B, self.P, self.C = S, D, B, P, C
        self.dt_schedule = {
            'dt_values': np.array(dt_values),
            'time_points': np.array(time_points),
            'proportions': proportions,
            'chunks': chunks
        }

    def stochastic_unequal_run(self, total_equal_var):
        """Run hedging with stochastic time steps based on gamma"""
        M, N = self.M, self.N
        S0, K, T, r, sigma = self.S0, self.K, self.T, self.r, self.sigma
        mult = self.contract_size
        opt_type = self.option_type

        # Allocate arrays indexed by hedge step
        P = np.zeros((M, N+1))
        S = np.zeros((M, N+1))
        D = np.zeros((M, N+1))
        B = np.zeros((M, N+1))
        C = np.zeros((M, N+1))

        # t=0
        S[:, 0] = S0
        D[:, 0] = self.side * self.bs_delta(S[:, 0], K, T, r, sigma, opt_type) * mult

        # CHANGED: Use hn_price instead of bs_price
        Time_inDays = T * 252
        r_daily = r / 252
        V0 = self.hn_price(S[:, 0], K, Time_inDays, r_daily, self.omega, self.alpha,
                          self.beta, self.gamma, self.lambda_, opt_type) * mult

        B[:, 0] = self.side * V0 - D[:, 0]*S[:, 0]

        current_time = 0.0
        hedge_idx = 1
        dt_values = []
        time_points = []

        while current_time < T:
            tau = max(T - current_time, 1e-12)

            # Compute gamma at current step
            gamma = self.bs_gamma(S[:, hedge_idx - 1], K, tau, r, sigma, opt_type) * mult * self.side

            # Compute stochastic dt
            S_abs = np.abs(S[:, hedge_idx - 1])
            dt_vec = 1 / ((gamma * sigma * S_abs)**2 + 1e-12)
            dt = np.sum(dt_vec) / total_equal_var
            dt = min(dt, T - current_time)

            # Advance stock
            Z = np.random.standard_normal(M)
            S[:, hedge_idx] = S[:, hedge_idx - 1] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)

            # Delta hedge
            newD = self.bs_delta(S[:, hedge_idx], K, tau - dt, r, sigma, opt_type) * mult * self.side
            trade = newD - D[:, hedge_idx - 1]

            # Transaction costs
            C[:, hedge_idx] = self.TCP * np.abs(trade) * S[:, hedge_idx]

            # Bond update
            B[:, hedge_idx] = B[:, hedge_idx - 1] * np.exp(r*dt) - trade*S[:, hedge_idx] - C[:, hedge_idx]

            # P&L update
            P[:, hedge_idx] = B[:, hedge_idx] - B[:, hedge_idx - 1]

            # Update delta
            D[:, hedge_idx] = newD

            # Save dt and time for schedule
            dt_values.append(dt)
            current_time += dt
            time_points.append(current_time)

            hedge_idx += 1
            if hedge_idx > N:
                break

        # Store arrays
        self.S, self.D, self.B, self.P, self.C = S[:, :hedge_idx], D[:, :hedge_idx], B[:, :hedge_idx], P[:, :hedge_idx], C[:, :hedge_idx]

        # Store dt schedule
        self.dt_schedule = {
            'dt_values': np.array(dt_values),
            'time_points': np.array(time_points),
            'proportions': None,
            'chunks': None
        }

    def unequal_run_constant_evolution(self, proportions, chunks):
        """
        Run hedging with non-uniform hedge frequency per time bucket.
        - proportions: array-like of length len(chunks), summing roughly to 1.0
        - chunks: list of lists of coarse time step indices (1-based)
        """
        M, N = self.M, self.N
        S0, K, T, m, r, sigma = self.S0, self.K, self.T, self.m, self.r, self.sigma
        opt_type = self.option_type
        mult = self.contract_size

        # Number of hedges in each chunk
        chunks_hedge = [int(w * N) for w in proportions]
        diff = N - sum(chunks_hedge)
        if diff > 0:
            for i in np.argsort(proportions)[-diff:]:
                chunks_hedge[i] += 1

        print(chunks_hedge)
        total_hedges = int(np.sum(chunks_hedge))

        if total_hedges == 0:
            self.S = np.full((M, 1), S0, dtype=float)
            self.D = np.zeros((M, 1))
            self.B = np.zeros((M, 1))
            self.P = np.zeros((M, 1))
            self.C = np.zeros((M, 1))
            return

        # Fine grid resolution
        N_fine = lcm(*chunks_hedge)
        dt_fine = T / N_fine

        # Simulate S on the fine grid
        Z = np.random.standard_normal((M, N_fine))
        S = np.empty((M, N_fine + 1), dtype=float)
        S[:, 0] = S0

        for t in range(1, N_fine + 1):
            S[:, t] = S[:, t-1] * np.exp((m - 0.5 * sigma**2) * dt_fine + sigma * np.sqrt(dt_fine) * Z[:, t-1])

        # Build hedge schedule
        hedge_indices = []
        for chunk_idx, coarse_idxs in enumerate(chunks):
            n_hedges = chunks_hedge[chunk_idx]
            if n_hedges <= 0 or len(coarse_idxs) == 0:
                continue

            start_coarse = coarse_idxs[0] - 1
            end_coarse = coarse_idxs[-1]
            f_start = int(np.floor(start_coarse * N_fine / N))
            f_end = int(np.floor(end_coarse * N_fine / N))

            if f_end <= f_start:
                f_end = min(f_start + 1, N_fine)

            idxs = np.linspace(f_start + 1, f_end, n_hedges, endpoint=False)
            hedge_indices.extend(idxs.astype(int))

        hedge_indices = np.clip(np.sort(np.array(hedge_indices, dtype=int)), 1, N_fine)
        hedge_indices = np.unique(hedge_indices)

        # State arrays
        D = np.zeros((M, total_hedges + 1), dtype=float)
        B = np.zeros((M, total_hedges + 1), dtype=float)
        P = np.zeros((M, total_hedges + 1), dtype=float)
        C = np.zeros((M, total_hedges + 1), dtype=float)

        # Initial setup at t=0
        D[:, 0] = self.side * self.bs_delta(S[:, 0], K, T, r, sigma, opt_type) * mult

        # CHANGED: Use hn_price instead of bs_price
        Time_inDays = T * 252
        r_daily = r / 252
        V0 = self.hn_price(S[:, 0], K, Time_inDays, r_daily, self.omega, self.alpha,
                          self.beta, self.gamma, self.lambda_, opt_type) * mult

        B[:, 0] = self.side * V0 - D[:, 0] * S[:, 0]

        # Hedge through the schedule
        prev_f = 0
        for h_idx, f_idx in enumerate(hedge_indices, start=1):
            dt_gap = (f_idx - prev_f) * dt_fine
            tau = max(T - f_idx * dt_fine, 1e-12)
            S_h = S[:, f_idx]

            newD = self.bs_delta(S_h, K, tau, r, sigma, opt_type) * mult * self.side
            trade = newD - D[:, h_idx - 1]
            C[:, h_idx] = self.TCP * np.abs(trade) * S_h
            B[:, h_idx] = B[:, h_idx - 1] * np.exp(r * dt_gap) - trade * S_h - C[:, h_idx]
            P[:, h_idx] = B[:, h_idx] - B[:, h_idx - 1]
            D[:, h_idx] = newD
            prev_f = f_idx

        self.S, self.D, self.B, self.P, self.C = S, D, B, P, C

    def retrieve_distribution(self):
        """Calculate final P&L distribution"""
        # CHANGED: Use hn_price instead of bs_price for final payoff
        # At expiry (T=0), we can use intrinsic value or hn_price with Time_inDays=0
        # For consistency, let's use intrinsic value at expiry
        if self.option_type == "call":
            payoff = np.maximum(self.S[:, -1] - self.K, 0.0) * self.contract_size
        elif self.option_type == "put":
            payoff = np.maximum(self.K - self.S[:, -1], 0.0) * self.contract_size
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        return -self.side * payoff + self.B[:, -1] + self.D[:, -1] * self.S[:, -1]
    def retrieve_variance(self):
      return self.VAR

    def plot_pnl(self, label_prefix="Hedge", num_paths_to_plot=6, random_seed=None):
        fig, ax = plt.subplots(figsize=(12, 6))
        # Optionally set a seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
        # Randomly choose paths
        paths = np.random.choice(self.P.shape[0], size=min(num_paths_to_plot, self.P.shape[0]), replace=False)
        print(self.P)
        # Plot each sampled path
        for i in paths:
            ax.plot(self.P[i, :], label=f"{label_prefix} path {i}")
        # Optional: text on the side (only if var_text is defined)
        # ax.text(1.02, 0.5, var_text, transform=ax.transAxes, va='center', ha='left',
        # bbox=dict(facecolor='white', alpha=0.85, edgecolor='black'))
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Incremental P&L")
        ax.set_title(f"Hedging Strategy P&L Paths ({label_prefix})")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def calculated_bucketed_variances(self, label_prefix="Hedge", bucket_size=8, random_seed=None):
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots(figsize=(12, 6))
        paths = self.N
        avg_variance = np.zeros((paths + (bucket_size)-1) // bucket_size)  # number of chunks
        chunks = []
        for chunk_idx, start_i in enumerate(range(0, paths, bucket_size)):
            end_i = min(start_i + bucket_size, paths) - 1
            chunk = list(range(start_i + 1, end_i + 2))  # shift indices +1
            chunks.append(chunk)
            var_chunk = np.var(self.P[:, start_i:end_i+1], axis=0)
            print(var_chunk)
            avg_variance[chunk_idx] = np.mean(var_chunk)
        value = avg_variance / avg_variance.sum()
        return value, chunks
        # Optionally set a seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
        # Compute variance across all paths at each timestep
        variance_per_timestep = np.var(self.P, axis=0)  # variance across paths for each timestep
        # Plot variance
        ax.plot(range(self.P.shape[1]), variance_per_timestep, color='blue', lw=2,
                label=f"{label_prefix} Variance at Each Timestep")
        # Labels and title
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Variance of P&L Across Paths")
        ax.set_title(f"Variance of Hedging P&L Paths Over Time ({label_prefix})")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def payoff_distribution(self):
        """
        Returns the distribution of the derivative payoff at expiry, scaled by contract size.
        This is the raw PnL relative to initial option price.
        """
        # simulate terminal stock prices
        if self.seed is not None:
            np.random.seed(self.seed)
        Z = np.random.standard_normal(self.M)
        ST = self.S0 * np.exp((self.r - 0.5*self.sigma**2)*self.T + self.sigma*np.sqrt(self.T)*Z)
        # option payoff
        if self.option_type == "call":
            payoff = np.maximum(ST - self.K, 0)
        else:
            payoff = np.maximum(self.K - ST, 0)
        # initial option price
        V0 = self.bs_price(self.S0, self.K, self.T, self.r, self.sigma, self.option_type)
        # side: -1 for long, +1 for short
        side = -1 if self.position == "long" else 1
        return (-side * V0 + payoff) * self.contract_size

    def compute_dt_schedule(self, proportions, chunks):
        """
        Compute the dt schedule for given proportions and chunks.
        Returns dt_values and time_points arrays.
        """
        # Flatten chunks safely
        flat_chunks = []
        for ch in chunks:
            if isinstance(ch, (list, np.ndarray)):
                flat_chunks.extend(ch)
            else:
                flat_chunks.append(ch)

        # Make sure they are sorted and unique
        flat_chunks = sorted(list(set(flat_chunks)))

        # Compute dt_values
        # dt = time interval lengths in terms of T
        # Include start=0 and end=N
        indices = [0] + flat_chunks + [self.N]
        dt_values = np.diff(indices) * (self.T / self.N)
        time_points = np.cumsum(dt_values)

        return dt_values, time_points


def distribution_stats(dist, alpha=0.95):
    mean_v = dist.mean()
    std_v = dist.std()
    skew_v = skew(dist)
    kurt_v = kurtosis(dist)
    p1, p5, p50, p95, p99 = np.percentile(dist, [1, 5, 50, 95, 99])
    var_level = np.percentile(dist, (1 - alpha) * 100)
    cvar = dist[dist <= var_level].mean()
    downside = dist[dist < 0]
    semivar = (downside**2).mean() if downside.size else 0.0
    return dict(mean=mean_v, std=std_v, skew=skew_v, kurt=kurt_v,
                p1=p1, p5=p5, p50=p50, p95=p95, p99=p99,
                VaR=var_level, CVaR=cvar, semivar=semivar)


def plot_final_distribution(dist, title):
    stats = distribution_stats(dist)
    mn, sd = stats["mean"], stats["std"]
    xmin, xmax = dist.min(), dist.max()
    x = np.linspace(xmin, xmax, 1000)
    pdf = norm.pdf(x, mn, sd) if sd > 0 else np.zeros_like(x)
    plt.figure(figsize=(9,6))
    plt.hist(dist, bins=140, density=True, alpha=0.55, edgecolor='black')
    plt.plot(x, pdf, 'r--', lw=2, label='Normal fit')
    stat_text = "\n".join([
        f"Mean {stats['mean']:.2f}",
        f"Std {stats['std']:.2f}",
        f"Skew {stats['skew']:.2f}",
        f"Kurt {stats['kurt']:.2f}",
        f"P1 {stats['p1']:.2f} P5 {stats['p5']:.2f}",
        f"Median {stats['p50']:.2f}",
        f"P95 {stats['p95']:.2f} P99 {stats['p99']:.2f}",
        f"VaR95 {stats['VaR']:.2f}",
        f"CVaR {stats['CVaR']:.2f}",
        f"SemiVar {stats['semivar']:.2f}"
    ])
    plt.text(0.02, 0.98, stat_text, transform=plt.gca().transAxes, va='top', ha='left',
             bbox=dict(facecolor='white', alpha=0.85, edgecolor='black'))
    plt.title(f"Final P&L Distribution: {title}")
    plt.xlabel("P&L")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_all_distributions(S0=100, K=100, r=0.00, sigma=0.25, T=1.0,
                          M=1000, N=252, TCP=0.00005, contract_size=1, seed=9):
    cfgs = [
        ("Short Call", "call", "short")
    ]
    for label, opt_type, pos in cfgs:
        print(label, opt_type, pos)
        sim = HedgingSim(S0=S0, K=K, r=r, sigma=sigma, T=T,
                         option_type=opt_type, position=pos,
                         M=M, N=N, TCP=TCP, contract_size=contract_size, seed=seed)
        sim.run()
        sim.plot_pnl()


def run_payoff_distributions(S0=100, K=100, r=0.00, sigma=0.25, T=1.0,
                             M=1000, N=252, TCP=0.00005, contract_size=1, seed=9):
    cfgs = [
        ("Short Call", "call", "short")
    ]
    for label, opt_type, pos in cfgs:
        print(label, opt_type, pos)
        sim = HedgingSim(S0=S0, K=K, r=r, sigma=sigma, T=T,
                         option_type=opt_type, position=pos,
                         M=M, N=N, TCP=TCP, contract_size=contract_size, seed=seed)
        payoffs = sim.payoff_distribution()
        return payoffs


def plot_stats_vs_frequency(S0=100, K=100, r=0.02, sigma=0.25, T=1.0,
                            M=600, TCP=0.00005, contract_size=1,
                            N_list=None, seed=11, alpha=0.95):
    if N_list is None:
        N_list = list(range(40, 241, 20))

    cfgs = [
        ("Long Call", "call", "long"),
        ("Short Call", "call", "short"),
        ("Long Put", "put", "long"),
        ("Short Put", "put", "short"),
    ]

    keys = ["mean", "std", "VaR", "CVaR", "semivar"]
    results = {lbl: {k: [] for k in keys} for lbl, _, _ in cfgs}

    for lbl, opt_type, pos in cfgs:
        for N in N_list:
            sim = HedgingSim(S0=S0, K=K, r=r, sigma=sigma, T=T,
                             option_type=opt_type, position=pos,
                             M=M, N=N, TCP=TCP, contract_size=contract_size, seed=seed)
            sim.run()
            dist = sim.retrieve_distribution()
            st = distribution_stats(dist, alpha=alpha)
            for k in keys:
                results[lbl][k].append(st[k])

    colors = {
        "Long Call": "tab:blue",
        "Short Call": "tab:orange",
        "Long Put": "tab:green",
        "Short Put": "tab:red"
    }

    plt.figure(figsize=(15, 11))
    plt.subplot(2, 2, 1)
    for lbl in results:
        plt.plot(N_list, results[lbl]["mean"], lw=2, label=lbl, color=colors[lbl])
    plt.title("Mean P&L")
    plt.xlabel("Hedge Steps (N)")
    plt.ylabel("Mean")
    plt.legend()

    plt.subplot(2, 2, 2)
    for lbl in results:
        plt.plot(N_list, results[lbl]["std"], lw=2, label=lbl, color=colors[lbl])
    plt.title("Std Dev")
    plt.xlabel("Hedge Steps (N)")
    plt.ylabel("Std")

    plt.subplot(2, 2, 3)
    for lbl in results:
        plt.plot(N_list, results[lbl]["VaR"], lw=2, label=lbl, color=colors[lbl])
    plt.title(f"VaR ({alpha*100:.0f}%)")
    plt.xlabel("Hedge Steps (N)")
    plt.ylabel("VaR")

    plt.subplot(2, 2, 4)
    for lbl in results:
        plt.plot(N_list, results[lbl]["CVaR"], lw=2, label=lbl, color=colors[lbl])
    plt.title(f"CVaR ({alpha*100:.0f}%)")
    plt.xlabel("Hedge Steps (N)")
    plt.ylabel("CVaR")

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    for lbl in results:
        plt.plot(N_list, results[lbl]["semivar"], lw=2, label=lbl, color=colors[lbl])
    plt.title("Downside Semivariance")
    plt.xlabel("Hedge Steps (N)")
    plt.ylabel("Semivariance")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return results

def plot_kde_distribution(dist, title="P&L Distribution", bw_method='scott'):
    """
    Plot KDE of a distribution instead of a histogram.

    Parameters:
    - dist: array-like, the data to plot
    - title: str, plot title
    - bw_method: str or float, bandwidth method for gaussian_kde ('scott', 'silverman', or float)
    """
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    import numpy as np

    kde = gaussian_kde(dist, bw_method=bw_method)
    x_min, x_max = dist.min(), dist.max()
    x = np.linspace(-10, 15, 1000)
    y = kde(x)

    plt.figure(figsize=(9, 6))
    plt.plot(x, y, lw=2, label='KDE')
    plt.fill_between(x, 0, y, alpha=0.3)
    plt.title(title)
    plt.xlabel("P&L")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()
