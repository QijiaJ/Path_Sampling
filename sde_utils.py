"""
SDE simulation utilities for path measure sampling experiments.
Provides reference processes (Brownian motion, OU, double-well),
likelihood functions, and Euler-Maruyama integration.
"""

import torch
import numpy as np


# =============================================================================
# Reference processes and their properties
# =============================================================================

class BrownianMotion:
    """Standard Brownian motion: dX_t = sqrt(2) dW_t."""
    def __init__(self, d=1):
        self.d = d
        self.name = "brownian"

    def drift(self, x, t):
        return torch.zeros_like(x)

    def diffusion_coeff(self):
        return np.sqrt(2.0)

    def stationary_density(self):
        return None  # No stationary distribution

    def grad_log_stationary(self, x):
        return torch.zeros_like(x)


class OrnsteinUhlenbeck:
    """OU process: dX_t = -beta * X_t dt + sqrt(2) dW_t.
    Stationary distribution: N(0, 1/beta).
    """
    def __init__(self, d=1, beta=0.25):
        self.d = d
        self.beta = beta
        self.name = "ou"

    def drift(self, x, t):
        return -self.beta * x

    def diffusion_coeff(self):
        return np.sqrt(2.0)

    def grad_log_stationary(self, x):
        """grad log nu(x) = -beta * x for nu = N(0, 1/beta * I)."""
        return -self.beta * x

    def potential(self, x):
        """V(x) such that nu propto exp(-V(x)). V(x) = beta/2 ||x||^2."""
        return 0.5 * self.beta * (x ** 2).sum(dim=-1)

    def grad_potential(self, x):
        return self.beta * x

    def sample_stationary(self, n):
        return torch.randn(n, self.d) / np.sqrt(self.beta)


class DoubleWell:
    """Double-well potential: V(x) = 5(||x||^2 - 1)^2.
    dX_t = -grad V(X_t) dt + sqrt(2) dW_t.
    For TPS with start A=-1, end B=+1 in 1D.
    For higher d, V(x) = 5(x_1^2 - 1)^2 (acts on first coordinate).
    """
    def __init__(self, d=1, scale=5.0):
        self.d = d
        self.scale = scale
        self.name = "double_well"

    def potential(self, x):
        """V(x) = scale * (x_1^2 - 1)^2 for TPS."""
        if x.dim() == 1:
            return self.scale * (x ** 2 - 1) ** 2
        return self.scale * (x[:, 0] ** 2 - 1) ** 2

    def grad_potential(self, x):
        """grad V(x)."""
        g = torch.zeros_like(x)
        if x.dim() == 1:
            g = 4 * self.scale * x * (x ** 2 - 1)
        else:
            g[:, 0] = 4 * self.scale * x[:, 0] * (x[:, 0] ** 2 - 1)
        return g

    def drift(self, x, t):
        d = -self.grad_potential(x)
        return d.clamp(-10.0, 10.0)

    def diffusion_coeff(self):
        return np.sqrt(2.0)


# =============================================================================
# Likelihood / observation models
# =============================================================================

def gaussian_likelihood(x_t, y_obs, t, t_obs, sigma_obs):
    """
    Gaussian likelihood: J_t(x_t) = sum_k 1/(2*sigma^2) ||y_k - x_{t_k}||^2 * delta(t - t_k).
    For soft constraint (no delta), returns the penalty at time t.
    x_t: (batch, d)
    """
    total = torch.zeros(x_t.shape[0], device=x_t.device)
    for y, t_o in zip(y_obs, t_obs):
        # Soft constraint with width gamma around observation time
        y_tensor = torch.tensor(y, device=x_t.device, dtype=x_t.dtype)
        if y_tensor.dim() == 0:
            y_tensor = y_tensor.unsqueeze(0)
        if y_tensor.dim() == 1 and x_t.dim() == 2:
            y_tensor = y_tensor.unsqueeze(0).expand(x_t.shape[0], -1)
        diff = x_t - y_tensor
        total += (1.0 / (2 * sigma_obs ** 2)) * (diff ** 2).sum(dim=-1)
    return total


def soft_constraint_J(x, x_fixed, gamma, t, t_obs):
    """
    Soft constraint likelihood: J_t(x) = 1/(2*gamma^2) ||x - x_fixed||^2
    applied at observation time t_obs.
    Returns per-sample J values.
    """
    if abs(t - t_obs) < 1e-6:
        diff = x - x_fixed
        return (1.0 / (2 * gamma ** 2)) * (diff ** 2).sum(dim=-1)
    return torch.zeros(x.shape[0], device=x.device)


# =============================================================================
# SDE integrators
# =============================================================================

def euler_maruyama_step(x, drift_fn, t, dt, diffusion=np.sqrt(2.0)):
    """Single Euler-Maruyama step: X_{t+dt} = X_t + drift(X_t, t)*dt + sigma*sqrt(dt)*Z."""
    noise = torch.randn_like(x) * np.sqrt(dt) * diffusion
    return x + drift_fn(x, t) * dt + noise


def simulate_sde(x0, drift_fn, T, dt, diffusion=np.sqrt(2.0), store_path=True):
    """
    Simulate SDE from t=0 to t=T.
    x0: (n, d) initial conditions
    drift_fn: callable (x, t) -> drift
    Returns: final x_T, and optionally full trajectory (n, T/dt+1, d).
    """
    n_steps = int(T / dt)
    x = x0.clone()
    if store_path:
        path = [x.clone()]
    for i in range(n_steps):
        t = i * dt
        x = euler_maruyama_step(x, drift_fn, t, dt, diffusion)
        if store_path:
            path.append(x.clone())
    if store_path:
        return x, torch.stack(path, dim=1)  # (n, n_steps+1, d)
    return x, None


def simulate_brownian_bridge(n, d, T, dt, A, B):
    """
    Sample Brownian bridge paths from X(0)=A to X(T)=B.
    Uses the SDE: dX_t = (B - X_t)/(T - t) dt + sqrt(2) dW_t.
    """
    n_steps = int(T / dt)
    A_tensor = torch.tensor(A, dtype=torch.float32)
    B_tensor = torch.tensor(B, dtype=torch.float32)

    if A_tensor.dim() == 0:
        x = A_tensor.expand(n, d).clone()
        B_expanded = B_tensor.expand(n, d)
    else:
        x = A_tensor.unsqueeze(0).expand(n, -1).clone()
        B_expanded = B_tensor.unsqueeze(0).expand(n, -1)

    path = [x.clone()]
    for i in range(n_steps):
        t = i * dt
        remaining = max(T - t, 1e-6)
        drift = (B_expanded - x) / remaining
        noise = torch.randn_like(x) * np.sqrt(2.0 * dt)
        x = x + drift * dt + noise
        path.append(x.clone())

    # Force endpoint
    path[-1] = B_expanded.clone()
    return torch.stack(path, dim=1)  # (n, n_steps+1, d)


# =============================================================================
# Onsager-Machlup action and path probability
# =============================================================================

def onsager_machlup_action(path, drift_fn, dt):
    """
    Compute the OM action I(x) = 1/2 int_0^T (1/2 ||dx/dt - u^ref||^2 + 1/2 div u^ref) dt
    for a discretized path.
    path: (n_steps+1, d) single path or (batch, n_steps+1, d)
    """
    if path.dim() == 2:
        path = path.unsqueeze(0)
    batch, n_steps_plus_1, d = path.shape
    n_steps = n_steps_plus_1 - 1

    action = torch.zeros(batch, device=path.device)
    for i in range(n_steps):
        t = i * dt
        x = path[:, i]
        x_next = path[:, i + 1]
        vel = (x_next - x) / dt
        drift = drift_fn(x, t)
        action += 0.5 * ((vel - drift) ** 2).sum(dim=-1) * dt

    return 0.5 * action
