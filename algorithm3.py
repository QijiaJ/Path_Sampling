"""
Algorithm 3: JKO with pushforward map for solving the Wasserstein dynamics (17).
Implements the Lagrangian method from Section 4.3.1 using both ICNN and standard MLP.

The algorithm evolves marginal densities q_t via a sequence of pushforward maps,
stepping from t=0 to t=T with step size h.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import tracemalloc

from networks import ICNN, PushforwardMLP


class WassersteinJKO:
    """
    Algorithm 3: JKO scheme with pushforward maps.

    Solves the Wasserstein dynamics by sequentially optimizing:
      q_{t+h} = argmin_{q} (1/2h) W_2^2(q, q_t) + F(q)

    where F involves the Fisher information I(q;nu) and the likelihood J.
    The pushforward is parameterized by either an ICNN or standard MLP.
    """

    def __init__(self, d, grad_V_fn, J_fn_at_t, T, h,
                 sigma_fi=0.4, m_perturbations=30,
                 use_icnn=True, hidden_dims=(64, 64, 64, 64),
                 lr=1e-3, device='cpu'):
        """
        d: spatial dimension
        grad_V_fn: callable(x) -> gradient of potential V (reference drift = -grad V)
        J_fn_at_t: callable(x, t) -> J_t(x) likelihood at specific time t
        T: time horizon
        h: JKO step size
        sigma_fi: bandwidth for Fisher information estimator
        m_perturbations: number of perturbations for FI estimator
        use_icnn: if True, use ICNN; otherwise use standard MLP
        hidden_dims: hidden layer sizes
        lr: learning rate
        """
        self.d = d
        self.grad_V = grad_V_fn
        self.J_fn = J_fn_at_t
        self.T = T
        self.h = h
        self.sigma_fi = sigma_fi
        self.m_pert = m_perturbations
        self.use_icnn = use_icnn
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.device = device

        # Store pushforward maps for each time step
        self.pushforward_maps = {}

    def _create_network(self):
        """Create a fresh pushforward network."""
        if self.use_icnn:
            return ICNN(self.d, self.hidden_dims, context_dim=1).to(self.device)
        else:
            return PushforwardMLP(self.d, self.hidden_dims, context_dim=1).to(self.device)

    def fisher_information_estimator(self, samples, sigma=None):
        """
        Lemma 4.2: Randomized FI estimator.
        R_hat(q_t) = 1/n sum_i (1/m sum_j ||y_j - x_i||^2 / (2*sigma^4))
        """
        if sigma is None:
            sigma = self.sigma_fi
        n, d = samples.shape
        # Generate perturbations for all samples at once
        # y_j^i ~ N(x_i, sigma^2 I)
        perturbations = samples.unsqueeze(1) + sigma * torch.randn(
            n, self.m_pert, d, device=self.device
        )  # (n, m, d)
        diffs_sq = ((perturbations - samples.unsqueeze(1)) ** 2).sum(dim=2)  # (n, m)
        fi_per_sample = diffs_sq.mean(dim=1) / (2 * sigma ** 4)  # (n,)
        return fi_per_sample  # Returns per-sample, caller decides how to aggregate

    def log_density_score(self, samples):
        """
        Estimate grad log q_t / nu using the stationary reference.
        For equilibrium reference nu propto exp(-V):
        grad log (q_t/nu)(x) ≈ grad log q_t(x) + grad V(x)
        We estimate grad log q_t via score estimation.
        """
        # Simple KDE-based score estimate for low dimensions
        n, d = samples.shape
        sigma = self.sigma_fi

        # Score of KDE: grad log q_hat(x) = -1/sigma^2 * sum_j w_j (x - x_j)
        # where w_j = K(x, x_j) / sum_k K(x, x_k)
        # This is expensive but works for moderate n
        diffs = samples.unsqueeze(0) - samples.unsqueeze(1)  # (n, n, d)
        dists_sq = (diffs ** 2).sum(dim=2)  # (n, n)
        weights = torch.exp(-dists_sq / (2 * sigma ** 2))  # (n, n)
        weights = weights / weights.sum(dim=1, keepdim=True)  # Normalize

        # grad log q(x_i) ≈ -1/sigma^2 * sum_j w_ij (x_i - x_j)
        score = -(1.0 / sigma ** 2) * (weights.unsqueeze(2) * diffs).sum(dim=1)
        return score

    def jko_step(self, particles, t, n_opt_steps=200):
        """
        Perform one JKO step: transport particles from q_t to q_{t+h}.

        Solves (Eq. 51):
        phi^theta <- argmin 1/(2h^2*n) sum (X_{t+h}^i - X_t^i)^2
                     + 1/n sum [R^theta(X_{t+h}^i) + 2/h * J_{t+h}(X_{t+h}^i)]
                     + L^theta({X_{t+h}^i})
        """
        n = particles.shape[0]
        phi = self._create_network()
        optimizer = optim.Adam(phi.parameters(), lr=self.lr)

        t_tensor = torch.full((n, 1), t, device=self.device)

        for step in range(n_opt_steps):
            # Pushforward: X_{t+h} = grad phi(X_t, t) for ICNN, or phi(X_t, t) for MLP
            if self.use_icnn:
                x_new = phi.gradient(particles.detach(), t)
            else:
                x_new = phi(particles.detach(), t)

            # 1. Transport cost: 1/(2h^2) * ||X_{t+h} - X_t||^2
            transport_cost = (1.0 / (2 * self.h ** 2 * n)) * \
                ((x_new - particles.detach()) ** 2).sum()

            # 2. Fisher information estimator R^theta
            fi_per_sample = self.fisher_information_estimator(x_new)
            fi_cost = fi_per_sample.mean()

            # 3. Likelihood term: (2/h) * J_{t+h}(X_{t+h})
            J_vals = self.J_fn(x_new, t + self.h)
            likelihood_cost = (2.0 / self.h) * J_vals.mean()

            # 4. Log-density penalty L^theta (from stationary reference)
            # L(x) = Delta log nu(x) + 1/2 ||grad V(x)||^2
            # For OU: this simplifies nicely
            grad_V_vals = self.grad_V(x_new)
            log_density_cost = (0.5 * (grad_V_vals ** 2).sum(dim=1)).mean()

            # 5. Terminal KL term if t+h == T
            terminal_cost = torch.tensor(0.0, device=self.device)
            if abs(t + self.h - self.T) < 1e-6:
                # 1/(nh) sum log(1/n / nu(X_T))
                # Approximated as log-density of particles vs reference
                V_vals = 0.5 * (particles.detach() ** 2).sum(dim=1)  # For Gaussian ref
                terminal_cost = (1.0 / (n * self.h)) * V_vals.sum()

            loss = transport_cost + fi_cost + likelihood_cost + log_density_cost + terminal_cost

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Get final transported particles
        if self.use_icnn:
            new_particles = phi.gradient(particles.detach().requires_grad_(True), t)
            new_particles = new_particles.detach()
        else:
            with torch.no_grad():
                new_particles = phi(particles, t)

        self.pushforward_maps[t] = phi
        return new_particles, loss.item()

    def run(self, x0, n_opt_steps=200, callback=None):
        """
        Run Algorithm 3 from t=0 to t=T.

        Args:
            x0: (n, d) initial particles ~ q_0 = rho_0
            n_opt_steps: optimization steps per JKO step
            callback: optional fn(t, particles, metrics)

        Returns:
            particles_history: dict mapping t -> particles (n, d)
            timing: dict with wall-clock time and memory
        """
        n_jko_steps = int(self.T / self.h)
        particles = x0.clone().to(self.device)
        particles_history = {0.0: particles.clone()}

        # Timing
        start_time = time.time()
        tracemalloc.start()
        timing = {'per_step': [], 'total': 0, 'peak_memory_mb': 0}

        for step in range(n_jko_steps):
            step_start = time.time()
            t = step * self.h

            particles, loss = self.jko_step(particles, t, n_opt_steps)
            particles_history[t + self.h] = particles.clone()

            step_time = time.time() - step_start
            timing['per_step'].append(step_time)

            if callback is not None:
                callback(t + self.h, particles, {'loss': loss, 'step_time': step_time})

            print(f"  JKO step {step+1}/{n_jko_steps}, "
                  f"t={t+self.h:.3f}, loss={loss:.6f}, time={step_time:.1f}s")

        timing['total'] = time.time() - start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        timing['peak_memory_mb'] = peak / 1024 / 1024

        return particles_history, timing
