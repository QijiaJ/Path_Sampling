"""
Algorithm 1: Controlled Transport from Prior to Posterior.
Implements the annealing scheme from Section 3.1 / Appendix F.

The drift b_s is parameterized as b_s = b_0 + sum of incremental updates,
where each increment is learned by a neural network phi^theta.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import tracemalloc

from networks import DriftNetwork
from sde_utils import euler_maruyama_step


class ControlledTransport:
    """
    Algorithm 1: Controlled transport from prior to posterior.

    Given:
      - Prior SDE with drift u^ref(x)
      - Likelihood J(x;y)
      - Annealing schedule: pi_s(x) propto rho_0(x) exp(-I(x) - s*J(x;y))

    The algorithm learns db^s/ds at each annealing step s, accumulating
    the drift b_s = b_0 + integral of updates.
    """

    def __init__(self, d, ref_drift_fn, J_fn, T, dt,
                 hidden_dims=(20, 30), lr=1e-3, device='cpu',
                 x0_sampler=None):
        """
        d: spatial dimension
        ref_drift_fn: callable(x, t) -> reference drift
        J_fn: callable(x, t) -> likelihood J_t(x) (per-sample, summed over obs)
        T: time horizon
        dt: physical time step
        hidden_dims: NN hidden layer sizes
        lr: learning rate
        x0_sampler: optional callable(n, d) -> (n, d) tensor for custom initial conditions.
                    If None, defaults to N(0, I).
        """
        self.d = d
        self.ref_drift = ref_drift_fn
        self.J_fn = J_fn
        self.T = T
        self.dt = dt
        self.n_time_steps = int(T / dt)
        self.device = device
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.x0_sampler = x0_sampler

        # Accumulated drift corrections stored as list of (network, delta_s) pairs
        self.drift_increments = []

    def current_drift(self, x, t, s):
        """Evaluate b_s(x,t) = u^ref(x,t) + sum of learned increments.
        Previous networks run under no_grad to prevent quadratic memory growth."""
        with torch.no_grad():
            b = self.ref_drift(x, t)
            for net, ds_val, s_val in self.drift_increments:
                b = b + net(x, t, s_val) * ds_val
        b = b.detach().requires_grad_(x.requires_grad)
        return b

    def simulate_paths(self, n_paths, s, with_grad=False):
        """
        Simulate n_paths trajectories under the current drift b_s.
        Returns paths tensor of shape (n_paths, n_time_steps+1, d).
        """
        if self.x0_sampler is not None:
            x = self.x0_sampler(n_paths, self.d).to(self.device)
        else:
            x = torch.randn(n_paths, self.d, device=self.device)  # X_0 ~ rho_0
        paths = [x.clone()]

        for i in range(self.n_time_steps):
            t = i * self.dt
            drift = self.current_drift(x, t, s)
            noise = torch.randn_like(x) * np.sqrt(2.0 * self.dt)
            x = x + drift * self.dt + noise
            if not with_grad:
                x = x.detach()
            paths.append(x.clone())

        return torch.stack(paths, dim=1)  # (n_paths, n_steps+1, d)

    def compute_h_s(self, paths, s):
        """
        Compute h_s(x) = -J(x;y) + E_{pi_s}[J(x;y)] for the current paths.
        This is the RHS of the consistency equation (11).

        h_s(x) involves terms from the OM functional; simplified here as:
        h_s(x) = -1/2 int_0^T [(b_s - dx/dt)^T (db/ds) + 1/2 div(db/ds)] dt
                  - J(x;y) + E[J(x;y)]

        For the loss, we use the squared residual form from Algorithm 1 line 6.
        """
        n_paths, n_steps_plus_1, d = paths.shape
        n_steps = n_steps_plus_1 - 1

        # Compute J for each path (at observation times, including terminal)
        J_vals = torch.zeros(n_paths, device=self.device)
        for i in range(n_steps + 1):
            t = i * self.dt
            x = paths[:, i]
            J_vals += self.J_fn(x, t)  # point-sum, not time integral — no dt factor

        # h_s consists of the OM-related terms and J
        # For the update, we need the quantity from Eq (11):
        # h_s(x) = -J(x;y) + E_{pi_s}[J(x;y)]
        # The full h_s also includes path derivative terms
        J_mean = J_vals.mean()
        return J_vals, J_mean

    def train_step(self, s, delta_s, n_paths, n_opt_steps):
        """
        Train the drift update phi^theta for one annealing step s -> s + delta_s.
        Following Algorithm 1 lines 4-7.
        """
        # Create a fresh network for this annealing step
        phi = DriftNetwork(self.d, self.hidden_dims).to(self.device)
        optimizer = optim.Adam(phi.parameters(), lr=self.lr)

        for opt_step in range(n_opt_steps):
            # Simulate paths under current drift
            paths = self.simulate_paths(n_paths, s)  # (K, T/dt+1, d)

            # Compute quantities for the loss (Algorithm 1, line 6)
            K = paths.shape[0]
            loss = torch.tensor(0.0, device=self.device)

            # For each path, compute the h_s terms
            h_vals = torch.zeros(K, device=self.device)
            phi_vals = torch.zeros(K, device=self.device)

            for i in range(self.n_time_steps):
                t_val = i * self.dt
                x = paths[:, i].detach().requires_grad_(True)

                # Finite difference for dx/dt
                dx_dt = (paths[:, i + 1] - paths[:, i]) / self.dt

                # Current drift
                b_s = self.current_drift(x, t_val, s)

                # NN output for db/ds
                phi_out = phi(x, t_val, s)

                # Compute divergence of phi using autograd
                div_phi = torch.zeros(K, device=self.device)
                for dim in range(self.d):
                    grad_phi_dim = torch.autograd.grad(
                        phi_out[:, dim].sum(), x, create_graph=True
                    )[0][:, dim]
                    div_phi += grad_phi_dim

                # h_theta^s from Algorithm 1 line 6:
                # -delta_t/2 * (b_s - dx/dt)^T * phi - delta_t/4 * div(phi)
                residual = (b_s.detach() - dx_dt.detach())
                h_vals += (-self.dt / 2) * (residual * phi_out).sum(dim=1) \
                          - (self.dt / 4) * div_phi

            # J values for each path (include terminal point at t=T)
            J_vals = torch.zeros(K, device=self.device)
            for i in range(self.n_time_steps + 1):
                t_val = i * self.dt
                J_vals += self.J_fn(paths[:, i].detach(), t_val)  # point-sum, not time integral — no dt factor

            J_mean = J_vals.mean()

            # Loss: 1/K sum_k (h_theta^s_k - h_bar_theta^s + J_k - J_mean)^2
            h_bar = h_vals.mean()
            residuals = (h_vals - h_bar) + (J_vals - J_mean)
            loss = (residuals ** 2).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Store this increment
        self.drift_increments.append((phi.eval(), delta_s, s))
        return loss.item()

    def run(self, n_annealing_steps=10, n_paths=500, n_opt_steps=200,
            callback=None, cosine_schedule=False, n_final_batches=1):
        """
        Run the full Algorithm 1.

        Args:
            n_annealing_steps: number of s increments from 0 to 1
            n_paths: ensemble size K
            n_opt_steps: Adam steps per annealing iteration
            callback: optional fn(s, paths, metrics) called after each step
            cosine_schedule: if True, use cosine-spaced s values (denser near 0 and 1)
            n_final_batches: number of independent final path batches to average

        Returns:
            final_paths: (n_paths, n_steps+1, d)
            timing: dict with wall-clock time and memory stats
        """
        if cosine_schedule:
            s_vals = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n_annealing_steps + 1)))
        else:
            s_vals = np.linspace(0, 1, n_annealing_steps + 1)

        # Timing
        start_time = time.time()
        tracemalloc.start()
        timing = {'per_step': [], 'total': 0, 'peak_memory_mb': 0}

        import gc

        for step in range(n_annealing_steps):
            step_start = time.time()
            s = s_vals[step]
            delta_s = s_vals[step + 1] - s_vals[step]

            loss = self.train_step(s, delta_s, n_paths, n_opt_steps)

            step_time = time.time() - step_start
            timing['per_step'].append(step_time)

            if callback is not None:
                paths = self.simulate_paths(n_paths, s + delta_s)
                callback(s + delta_s, paths, {'loss': loss, 'step_time': step_time})

            print(f"  Annealing step {step+1}/{n_annealing_steps}, "
                  f"s={s+delta_s:.4f}, Δs={delta_s:.4f}, loss={loss:.6f}, time={step_time:.1f}s")

            # Free intermediate memory
            gc.collect()

        timing['total'] = time.time() - start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        timing['peak_memory_mb'] = peak / 1024 / 1024

        # Final paths at s=1 (average multiple batches for stability)
        if n_final_batches > 1:
            all_paths = []
            for _ in range(n_final_batches):
                all_paths.append(self.simulate_paths(n_paths, 1.0))
            final_paths = torch.cat(all_paths, dim=0)
        else:
            final_paths = self.simulate_paths(n_paths, 1.0)
        return final_paths, timing
