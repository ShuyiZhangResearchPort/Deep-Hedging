import torch
import numpy as np
import numpy as np
from numba import jit, prange
def gamma_precomputed_analytical(
    S,
    K,
    step_idx,
    r_daily,
    N,
    option_type,
    precomputed_data,
    omega,
    alpha,
    beta,
    gamma_param,
    lambda_
):
    """
    Compute gamma analytically using precomputed Heston–Nandi coefficients.
    Gamma is the second derivative of option price with respect to S.

    Args:
        S: Spot price(s), float or torch.Tensor
        K: Strike price(s), float or torch.Tensor
        step_idx: Current step index (0..N)
        r_daily: Daily risk-free rate
        N: Total number of days to expiry used for precomputation
        option_type: 'call' or 'put'
        precomputed_data: Output of precompute_hn_coefficients(...)
        omega, alpha, beta, gamma_param, lambda_: HN parameters (not used directly here)
        sigma0: Annualized initial volatility (not used in gamma; included per requested signature)

    Returns:
        torch.FloatTensor of gamma with the broadcasted shape of (S, K).
    """
    device = torch.device(precomputed_data["device"])
    coefficients = precomputed_data["coefficients"]
    u_nodes = precomputed_data["u_nodes"]
    w_nodes = precomputed_data["w_nodes"]

    S_t = torch.as_tensor(S, dtype=torch.float64, device=device)
    K_t = torch.as_tensor(K, dtype=torch.float64, device=device)

    shape = torch.broadcast_shapes(S_t.shape, K_t.shape)
    S_bc = S_t.expand(shape) if S_t.shape != shape else S_t
    K_bc = K_t.expand(shape) if K_t.shape != shape else K_t

    log_S = torch.log(S_bc)
    log_K = torch.log(K_bc)

    coeff_step = coefficients[step_idx]

    # First integral (const = 1)
    coeff_K_1 = coeff_step[:, 0, 0]
    coeff_S_1 = coeff_step[:, 0, 1]
    const_term_1 = coeff_step[:, 0, 2]

    exponent_1 = (
        coeff_K_1.unsqueeze(-1) * log_K.unsqueeze(0)
        + coeff_S_1.unsqueeze(-1) * log_S.unsqueeze(0)
        + const_term_1.unsqueeze(-1)
    )

    cphi0_1 = 1j * u_nodes
    f1 = torch.exp(exponent_1) / cphi0_1.unsqueeze(-1) / np.pi

    # For gamma: integrand is cphi * (cphi - 1) * f / S^2, with cphi = 1j*u + const
    cphi_1 = cphi0_1 + 1.0
    gamma_integrand_1 = (
        cphi_1.unsqueeze(-1)
        * (cphi_1.unsqueeze(-1) - 1.0)
        * f1
        / (S_bc.unsqueeze(0) ** 2)
    )

    # Second integral (const = 0)
    coeff_K_0 = coeff_step[:, 1, 0]
    coeff_S_0 = coeff_step[:, 1, 1]
    const_term_0 = coeff_step[:, 1, 2]

    exponent_0 = (
        coeff_K_0.unsqueeze(-1) * log_K.unsqueeze(0)
        + coeff_S_0.unsqueeze(-1) * log_S.unsqueeze(0)
        + const_term_0.unsqueeze(-1)
    )

    cphi0_0 = 1j * u_nodes
    f0 = torch.exp(exponent_0) / cphi0_0.unsqueeze(-1) / np.pi

    cphi_0 = cphi0_0
    gamma_integrand_0 = (
        cphi_0.unsqueeze(-1)
        * (cphi_0.unsqueeze(-1) - 1.0)
        * f0
        / (S_bc.unsqueeze(0) ** 2)
    )

    # Integrate (real part)
    gamma1 = torch.sum(w_nodes.unsqueeze(-1) * torch.real(gamma_integrand_1), dim=0)
    gamma2 = torch.sum(w_nodes.unsqueeze(-1) * torch.real(gamma_integrand_0), dim=0)

    Time_inDays = float(N - step_idx)
    disc = torch.exp(torch.tensor(-r_daily * Time_inDays, dtype=torch.float64, device=device))

    # Gamma is same for calls and puts
    gamma_val = disc * gamma1 - K_bc * disc * gamma2
    return gamma_val.to(torch.float32)


def vega_precomputed_analytical(
    S,
    K,
    step_idx,
    r_daily,
    N,
    option_type,
    precomputed_data,
    omega,
    alpha,
    beta,
    gamma_param,
    lambda_,
    sigma0,
):
    """
    Compute vega analytically using precomputed Heston–Nandi coefficients.
    This returns standard vega with respect to volatility by scaling the
    variance-sensitivity by 2 * sigma0, as requested.

    Mathematically:
        If price integrand uses exp(a + b * sigma2), then:
            ∂Price/∂sigma2 = integral( b * f )
        And vega (∂Price/∂sigma) = 2 * sigma0 * ∂Price/∂sigma2

    Args:
        S: Spot price(s), float or torch.Tensor
        K: Strike price(s), float or torch.Tensor
        step_idx: Current step index (0..N)
        r_daily: Daily risk-free rate
        N: Total number of days to expiry used for precomputation
        option_type: 'call' or 'put' (vega is same for calls and puts)
        precomputed_data: Output of precompute_hn_coefficients(...)
        omega, alpha, beta, gamma_param, lambda_: HN parameters
        sigma0: Annualized initial volatility (used for scaling: 2 * sigma0)

    Returns:
        torch.FloatTensor of vega (scaled by 2 * sigma0) with the broadcasted shape of (S, K).
    """
    device = torch.device(precomputed_data["device"])
    coefficients = precomputed_data["coefficients"]
    u_nodes = precomputed_data["u_nodes"]
    w_nodes = precomputed_data["w_nodes"]

    S_t = torch.as_tensor(S, dtype=torch.float64, device=device)
    K_t = torch.as_tensor(K, dtype=torch.float64, device=device)

    shape = torch.broadcast_shapes(S_t.shape, K_t.shape)
    S_bc = S_t.expand(shape) if S_t.shape != shape else S_t
    K_bc = K_t.expand(shape) if K_t.shape != shape else K_t

    log_S = torch.log(S_bc)
    log_K = torch.log(K_bc)

    # Complex parameters for recursion of b
    omega_c = torch.tensor(omega, dtype=torch.complex128, device=device)
    alpha_c = torch.tensor(alpha, dtype=torch.complex128, device=device)
    beta_c = torch.tensor(beta, dtype=torch.complex128, device=device)
    gamma_c = torch.tensor(gamma_param, dtype=torch.complex128, device=device)
    lambda_c = torch.tensor(lambda_, dtype=torch.complex128, device=device)
    r_daily_c = torch.tensor(r_daily, dtype=torch.complex128, device=device)

    lambda_r = torch.tensor(-0.5, dtype=torch.complex128, device=device)
    gamma_r = gamma_c + lambda_c + 0.5

    coeff_step = coefficients[step_idx]
    Time_inDays = N - step_idx

    iu_nodes = 1j * u_nodes

    # Accumulators for the two integrals (const=1 and const=0)
    vega1 = None
    vega2 = None

    for const_idx, const_val in enumerate([1.0, 0.0]):
        const_c = torch.tensor(const_val, dtype=torch.complex128, device=device)
        cphi_vec = iu_nodes + const_c

        # Recurrence for b
        b_vec = lambda_r * cphi_vec + 0.5 * cphi_vec**2
        for _ in range(1, Time_inDays):
            denom_vec = 1.0 - 2.0 * alpha_c * b_vec
            b_vec = (
                cphi_vec * (lambda_r + gamma_r)
                - 0.5 * gamma_r**2
                + beta_c * b_vec
                + 0.5 * (cphi_vec - gamma_r) ** 2 / denom_vec
            )

        # Precomputed exponent parts
        coeff_K = coeff_step[:, const_idx, 0]
        coeff_S = coeff_step[:, const_idx, 1]
        const_term = coeff_step[:, const_idx, 2]

        exponent = (
            coeff_K.unsqueeze(-1) * log_K.unsqueeze(0)
            + coeff_S.unsqueeze(-1) * log_S.unsqueeze(0)
            + const_term.unsqueeze(-1)
        )

        cphi0 = 1j * u_nodes
        f = torch.exp(exponent) / cphi0.unsqueeze(-1) / np.pi

        # Vega integrand: b * f
        vega_integrand = b_vec.unsqueeze(-1) * f
        contrib = torch.sum(w_nodes.unsqueeze(-1) * torch.real(vega_integrand), dim=0)

        if const_idx == 0:
            vega1 = contrib
        else:
            vega2 = contrib

    Time_inDays_f = float(N - step_idx)
    disc = torch.exp(torch.tensor(-r_daily * Time_inDays_f, dtype=torch.float64, device=device))

    # Raw sensitivity to variance (sigma^2)
    vega_variance = disc * vega1 - K_bc * disc * vega2

    # Scale to volatility sensitivity: vega_sigma = 2 * sigma0 * dV/d(sigma^2)
    vega_sigma = (2.0 * float(sigma0)) * vega_variance

    # Vega is the same for calls and puts
    return vega_sigma.to(torch.float32)
def delta_precomputed_analytical(S, K, step_idx, r_daily, N, option_type, precomputed_data):
    """
    Compute delta analytically using precomputed coefficients.
    Delta = d(Price)/dS using the characteristic function formula directly.
    """
    device = torch.device(precomputed_data["device"])
    coefficients = precomputed_data["coefficients"]
    u_nodes = precomputed_data["u_nodes"]
    w_nodes = precomputed_data["w_nodes"]

    S_t = torch.as_tensor(S, dtype=torch.float64, device=device)
    K_t = torch.as_tensor(K, dtype=torch.float64, device=device)

    shape = torch.broadcast_shapes(S_t.shape, K_t.shape)
    S_bc = S_t.expand(shape) if S_t.shape != shape else S_t
    K_bc = K_t.expand(shape) if K_t.shape != shape else K_t

    log_S = torch.log(S_bc)
    log_K = torch.log(K_bc)

    coeff_step = coefficients[step_idx]

    # For const=1
    coeff_K_1 = coeff_step[:, 0, 0]
    coeff_S_1 = coeff_step[:, 0, 1]
    const_term_1 = coeff_step[:, 0, 2]

    exponent_1 = (coeff_K_1.unsqueeze(-1) * log_K.unsqueeze(0) +
                  coeff_S_1.unsqueeze(-1) * log_S.unsqueeze(0) +
                  const_term_1.unsqueeze(-1))

    cphi0_1 = 1j * u_nodes
    f1 = torch.exp(exponent_1) / cphi0_1.unsqueeze(-1) / np.pi

    # Derivative: d/dS of exp(...) = exp(...) * coeff_S / S
    df1_dS = f1 * coeff_S_1.unsqueeze(-1) / S_bc.unsqueeze(0)

    # For const=0
    coeff_K_0 = coeff_step[:, 1, 0]
    coeff_S_0 = coeff_step[:, 1, 1]
    const_term_0 = coeff_step[:, 1, 2]

    exponent_0 = (coeff_K_0.unsqueeze(-1) * log_K.unsqueeze(0) +
                  coeff_S_0.unsqueeze(-1) * log_S.unsqueeze(0) +
                  const_term_0.unsqueeze(-1))

    cphi0_0 = 1j * u_nodes
    f0 = torch.exp(exponent_0) / cphi0_0.unsqueeze(-1) / np.pi

    df0_dS = f0 * coeff_S_0.unsqueeze(-1) / S_bc.unsqueeze(0)

    # Integrate derivatives
    delta1 = torch.sum(w_nodes.unsqueeze(-1) * torch.real(df1_dS), dim=0)
    delta2 = torch.sum(w_nodes.unsqueeze(-1) * torch.real(df0_dS), dim=0)

    Time_inDays = float(N - step_idx)
    disc = torch.exp(torch.tensor(-r_daily * Time_inDays, dtype=torch.float64, device=device))

    # Delta formula: dC/dS = 0.5 + disc * delta1 - K * disc * delta2
    delta_call = 0.5 + disc * delta1 - K_bc * disc * delta2

    if option_type == "call":
        return delta_call.to(torch.float32)
    elif option_type == "put":
        return (delta_call - 1.0).to(torch.float32)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
@jit(nopython=True, cache=True)
def _fstar_hn_scalar(phi, const, S, X, Time_inDays, r_daily, omega, alpha, beta, gamma, lambda_):
    cphi0 = 1j * phi
    cphi = cphi0 + const
    lambda_r = -0.5
    gamma_r = gamma + lambda_ + 0.5
    sigma2 = (omega + alpha) / (1 - beta - alpha * gamma_r**2)

    a = cphi * r_daily
    b = lambda_r * cphi + 0.5 * cphi**2

    for i in range(1, int(Time_inDays)):
        denom = 1 - 2 * alpha * b
        a = a + cphi * r_daily + b * omega - 0.5 * np.log(denom)
        b = cphi * (lambda_r + gamma_r) - 0.5 * gamma_r**2 + beta * b + 0.5 * (cphi - gamma_r)**2 / denom

    result = np.exp(-cphi0 * np.log(X) + cphi * np.log(S) + a + b * sigma2) / cphi0 / np.pi
    return result.real

# -------------------------------
# Vectorized version
@jit(nopython=True, parallel=True, cache=True)
def _fstar_hn_vectorized(phi, const, S_flat, X_flat, Time_flat, r_flat, omega, alpha, beta, gamma, lambda_, n_elements):
    result = np.empty(n_elements, dtype=np.float64)
    for idx in prange(n_elements):
        result[idx] = _fstar_hn_scalar(phi, const, S_flat[idx], X_flat[idx], Time_flat[idx], r_flat[idx], omega, alpha, beta, gamma, lambda_)
    return result

# -------------------------------
# Complex scalar
@jit(nopython=True, cache=True)
def _f_hn_scalar(phi, const, S, X, Time_inDays, r_daily, omega, alpha, beta, gamma, lambda_):
    cphi0 = 1j * phi
    cphi = cphi0 + const
    lambda_r = -0.5
    gamma_r = gamma + lambda_ + 0.5
    sigma2 = (omega + alpha) / (1 - beta - alpha * gamma_r**2)

    a = cphi * r_daily
    b = lambda_r * cphi + 0.5 * cphi**2

    for i in range(1, int(Time_inDays)):
        denom = 1 - 2 * alpha * b
        a = a + cphi * r_daily + b * omega - 0.5 * np.log(denom)
        b = cphi * (lambda_r + gamma_r) - 0.5 * gamma_r**2 + beta * b + 0.5 * (cphi - gamma_r)**2 / denom

    return np.exp(-cphi0 * np.log(X) + cphi * np.log(S) + a + b * sigma2) / cphi0 / np.pi

# -------------------------------
# Vectorized complex
@jit(nopython=True, parallel=True, cache=True)
def _f_hn_vectorized(phi, const, S_flat, X_flat, Time_flat, r_flat, omega, alpha, beta, gamma, lambda_, n_elements):
    result = np.empty(n_elements, dtype=np.complex128)
    for idx in prange(n_elements):
        result[idx] = _f_hn_scalar(phi, const, S_flat[idx], X_flat[idx], Time_flat[idx], r_flat[idx], omega, alpha, beta, gamma, lambda_)
    return result
