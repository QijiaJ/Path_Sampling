#!/usr/bin/env python3
"""
Algorithm 1 vs Algorithm 3 across likelihood regimes.

Setup: OU process β=0.25, d=1, T=1.0, dt=0.02, equilibrium initialization.
Reference: analytical Gaussian posterior via Kalman forward-backward smoother.
Algorithm 1: 30 annealing, 200 opt steps (full budget).
Algorithm 3: h=0.2, 300 particles, 300 opt steps.

Four likelihood regimes spanning dense/sparse × mild/extreme:
  - Dense/mild:    5 obs, targets [3,-3,3,-3,3], σ=0.3
  - Dense/abrupt:  5 obs, targets [5,-4,6,-3,7], σ=0.2
  - Sparse/mild:   1 obs at t=1, target=4, σ=0.3
  - Sparse/extreme: 1 obs at t=1, target=8, σ=0.3

"""

import json, os, time, math, gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ========================================================================
# Device
# ========================================================================
def get_device():
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

DEVICE = get_device()
print(f"[device] {DEVICE}")

# ========================================================================
# OU Process
# ========================================================================
BETA = 0.25        # OU parameter: dX = -β X dt + √2 dW
D = 1
T = 1.0
DT = 0.02
N_STEPS = int(T / DT)  # 50
SIGMA_DIFF = math.sqrt(2)  # diffusion coefficient
EQ_VAR = 1.0 / BETA        # equilibrium variance = 4.0
EQ_STD = math.sqrt(EQ_VAR) # = 2.0


def ou_drift(x):
    """u(x) = -β x."""
    return -BETA * x


def ou_potential(x):
    """V(x) = β/2 x²."""
    return 0.5 * BETA * x**2


def ou_grad_potential(x):
    """∇V(x) = β x."""
    return BETA * x


def simulate_ou_paths(n, x0=None):
    """Simulate n unconditional OU paths from x0 (or equilibrium)."""
    if x0 is None:
        x = torch.randn(n, D, device=DEVICE) * EQ_STD
    else:
        x = x0.clone()
    paths = [x.clone()]
    sq = SIGMA_DIFF * math.sqrt(DT)
    for _ in range(N_STEPS):
        x = x + ou_drift(x) * DT + sq * torch.randn_like(x)
        paths.append(x.clone())
    return torch.stack(paths, dim=1)  # (n, N_STEPS+1, D)


# ========================================================================
# Analytical Gaussian posterior via Kalman smoother
# ========================================================================
def ou_conditional_moments(obs_times, obs_targets, sigma_obs, beta=BETA,
                           sigma_diff=SIGMA_DIFF, eq_var=EQ_VAR,
                           dt_eval=DT, T_end=T):
    """
    Compute the exact conditional mean and variance of the OU process
    given Gaussian observations.

    The OU process dX = -β X dt + σ dW with equilibrium N(0, 1/β)
    is a linear-Gaussian system. With Gaussian observations
    y_k ~ N(x(t_k), σ_obs²), the conditional distribution is
    a Gaussian process. We compute it via Kalman forward-backward
    (Rauch-Tung-Striebel) smoother on the discrete grid.

    Returns:
        eval_times: array of shape (n_eval,) — the evaluation grid
        cond_mean:  array of shape (n_eval,) — E[X(t) | y_{1:K}]
        cond_var:   array of shape (n_eval,) — Var[X(t) | y_{1:K}]
    """
    n_eval = int(round(T_end / dt_eval)) + 1
    eval_times = np.linspace(0, T_end, n_eval)

    # OU transition: X(t+Δ) | X(t) ~ N(e^{-βΔ} X(t), (1/β)(1 - e^{-2βΔ}))
    def ou_transition(delta):
        a = math.exp(-beta * delta)
        q = (1.0 / beta) * (1.0 - math.exp(-2 * beta * delta))
        return a, q

    # --- Forward pass (Kalman filter) ---
    # State: (mean, variance) of X(t) | y_{1:k} for observations up to time t
    # Prior at t=0: equilibrium N(0, eq_var)
    fwd_mean = np.zeros(n_eval)
    fwd_var = np.zeros(n_eval)
    fwd_mean[0] = 0.0
    fwd_var[0] = eq_var

    # Sort observations by time
    obs_sorted = sorted(zip(obs_times, obs_targets))
    obs_idx = 0  # pointer into obs_sorted

    for i in range(1, n_eval):
        delta = eval_times[i] - eval_times[i - 1]
        a, q = ou_transition(delta)

        # Predict
        pred_mean = a * fwd_mean[i - 1]
        pred_var = a**2 * fwd_var[i - 1] + q

        # Update if there's an observation at this time
        t_i = eval_times[i]
        if obs_idx < len(obs_sorted) and abs(obs_sorted[obs_idx][0] - t_i) < dt_eval * 0.5:
            y_k = obs_sorted[obs_idx][1]
            # Kalman gain
            S = pred_var + sigma_obs**2
            K_gain = pred_var / S
            fwd_mean[i] = pred_mean + K_gain * (y_k - pred_mean)
            fwd_var[i] = pred_var - K_gain * pred_var
            obs_idx += 1
        else:
            fwd_mean[i] = pred_mean
            fwd_var[i] = pred_var

    # --- Backward pass (Rauch-Tung-Striebel smoother) ---
    smooth_mean = np.zeros(n_eval)
    smooth_var = np.zeros(n_eval)
    smooth_mean[-1] = fwd_mean[-1]
    smooth_var[-1] = fwd_var[-1]

    for i in range(n_eval - 2, -1, -1):
        delta = eval_times[i + 1] - eval_times[i]
        a, q = ou_transition(delta)

        pred_mean_ip1 = a * fwd_mean[i]
        pred_var_ip1 = a**2 * fwd_var[i] + q

        # Smoother gain
        G = a * fwd_var[i] / pred_var_ip1
        smooth_mean[i] = fwd_mean[i] + G * (smooth_mean[i + 1] - pred_mean_ip1)
        smooth_var[i] = fwd_var[i] + G**2 * (smooth_var[i + 1] - pred_var_ip1)

    return eval_times, smooth_mean, smooth_var


def sample_ou_conditional(n_samples, obs_times, obs_targets, sigma_obs,
                          dt_eval=DT, T_end=T):
    """
    Draw exact samples from the OU conditional Gaussian process.
    Uses the Kalman smoother moments + sequential conditional sampling.
    """
    eval_times, cond_mean, cond_var = ou_conditional_moments(
        obs_times, obs_targets, sigma_obs, dt_eval=dt_eval, T_end=T_end
    )
    n_eval = len(eval_times)


    # Sequentially sample: X(t_0) ~ N(μ_0, σ²_0), then
    # X(t_{i+1}) | X(t_i) ~ N(μ_{i+1|i}, σ²_{i+1|i})
    beta_val = BETA
    sigma_diff_val = SIGMA_DIFF
    eq_var_val = EQ_VAR

    def ou_transition(delta):
        a = math.exp(-beta_val * delta)
        q = (1.0 / beta_val) * (1.0 - math.exp(-2 * beta_val * delta))
        return a, q

    # Recompute forward pass to get gains
    fwd_mean_arr = np.zeros(n_eval)
    fwd_var_arr = np.zeros(n_eval)
    fwd_mean_arr[0] = 0.0
    fwd_var_arr[0] = eq_var_val
    obs_sorted = sorted(zip(obs_times, obs_targets))
    obs_idx = 0
    for i in range(1, n_eval):
        delta = eval_times[i] - eval_times[i - 1]
        a, q = ou_transition(delta)
        pred_m = a * fwd_mean_arr[i - 1]
        pred_v = a**2 * fwd_var_arr[i - 1] + q
        t_i = eval_times[i]
        if obs_idx < len(obs_sorted) and abs(obs_sorted[obs_idx][0] - t_i) < dt_eval * 0.5:
            y_k = obs_sorted[obs_idx][1]
            S = pred_v + sigma_obs**2
            K_g = pred_v / S
            fwd_mean_arr[i] = pred_m + K_g * (y_k - pred_m)
            fwd_var_arr[i] = pred_v - K_g * pred_v
            obs_idx += 1
        else:
            fwd_mean_arr[i] = pred_m
            fwd_var_arr[i] = pred_v

    # Smoother gains
    G_arr = np.zeros(n_eval - 1)
    pred_var_arr = np.zeros(n_eval)
    for i in range(n_eval - 1):
        delta = eval_times[i + 1] - eval_times[i]
        a, q = ou_transition(delta)
        pv = a**2 * fwd_var_arr[i] + q
        pred_var_arr[i + 1] = pv
        G_arr[i] = a * fwd_var_arr[i] / pv


    # Use cond_mean, cond_var from the smoother
    paths = np.zeros((n_samples, n_eval))
    paths[:, 0] = np.random.randn(n_samples) * math.sqrt(max(cond_var[0], 1e-12)) + cond_mean[0]

    for i in range(n_eval - 1):
        V_i = max(cond_var[i], 1e-12)
        V_ip1 = cond_var[i + 1]
        C_i = G_arr[i] * V_ip1  # cross-covariance

        cond_trans_mean = cond_mean[i + 1] + (C_i / V_i) * (paths[:, i] - cond_mean[i])
        cond_trans_var = max(V_ip1 - C_i**2 / V_i, 1e-12)
        paths[:, i + 1] = cond_trans_mean + np.random.randn(n_samples) * math.sqrt(cond_trans_var)

    # Convert to torch tensor (n_samples, n_eval, 1)
    paths_t = torch.tensor(paths, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
    return paths_t, eval_times, cond_mean, cond_var


# ========================================================================
# Multi-point observation likelihood
# ========================================================================
class Likelihood:
    """J(x;y) = Σ_k (1/(2σ²)) (x(t_k) - y_k)²."""

    def __init__(self, obs_times, obs_targets, sigma_obs):
        self.obs = []
        for t, y in zip(obs_times, obs_targets):
            idx = int(round(t / DT))
            y_t = torch.tensor([y], dtype=torch.float32, device=DEVICE)
            self.obs.append((t, idx, y_t))
        self.sigma_obs = sigma_obs
        self.obs_times = obs_times
        self.obs_targets = obs_targets

    def J_path(self, paths):
        """Evaluate total J for each path. paths: (n, N_STEPS+1, D)."""
        J = torch.zeros(paths.shape[0], device=DEVICE)
        for t, idx, y in self.obs:
            idx_c = min(idx, paths.shape[1] - 1)
            diff = paths[:, idx_c, :] - y.unsqueeze(0)
            J = J + (0.5 / self.sigma_obs**2) * (diff**2).sum(dim=1)
        return J

    def J_point(self, x, t_val):
        """Evaluate J contributions at physical time t_val. x: (n, D)."""
        J = torch.zeros(x.shape[0], device=DEVICE)
        for t, idx, y in self.obs:
            if abs(t_val - t) < DT * 1.5:
                diff = x - y.unsqueeze(0)
                J = J + (0.5 / self.sigma_obs**2) * (diff**2).sum(dim=1)
        return J

    def J_point_ramped(self, x, t_val):
        """
        Ramped J: for each observation at t_k, apply a 5-step linear ramp
        in [t_k - 5*DT, t_k] so the gradient signal is smoother than a
        single delta at t_k.  Matches the terminal ramp in run_dw_d12.py.
        """
        J = torch.zeros(x.shape[0], device=DEVICE)
        ramp_width = 5 * DT
        for t, idx, y in self.obs:
            if t - ramp_width - DT * 0.5 <= t_val <= t + DT * 0.5:
                if t_val >= t - DT * 0.5:
                    weight = 1.0  # at or past observation time
                else:
                    weight = max(0.0, (t_val - (t - ramp_width)) / ramp_width)
                diff = x - y.unsqueeze(0)
                J = J + weight * (0.5 / self.sigma_obs**2) * (diff**2).sum(dim=1)
        return J

    def J_in_interval(self, x, t_lo, t_hi):
        """Sum J for observations with t_lo < t_k ≤ t_hi. For JKO steps."""
        J = torch.zeros(x.shape[0], device=DEVICE)
        for t, idx, y in self.obs:
            if t_lo - 1e-8 < t <= t_hi + 1e-8:
                diff = x - y.unsqueeze(0)
                J = J + (0.5 / self.sigma_obs**2) * (diff**2).sum(dim=1)
        return J


# ========================================================================
# Metrics (against analytical reference)
# ========================================================================
def sliced_wasserstein_1d(x, y):
    """SW₂ for 1D samples. x, y: (n,) tensors."""
    x = x.flatten().sort().values
    y = y.flatten().sort().values
    n = min(len(x), len(y))
    return ((x[:n] - y[:n])**2).mean().item()


def sliced_wasserstein_path(paths_a, paths_b, n_proj=50):
    """
    Path-space sliced Wasserstein-2 distance.
    Flatten each path to R^{n_time * d}, project onto random directions,
    compute 1D W₂ for each projection, average.
    paths_a, paths_b: (n, n_time, d) tensors.
    """
    na, nt_a, d = paths_a.shape
    nb, nt_b, _ = paths_b.shape
    nt = min(nt_a, nt_b)
    a = paths_a[:, :nt, :].reshape(na, -1)  # (na, nt*d)
    b = paths_b[:, :nt, :].reshape(nb, -1)  # (nb, nt*d)
    dim = a.shape[1]

    # Random projections
    torch.manual_seed(0)  # reproducible projections
    dirs = torch.randn(n_proj, dim, device=a.device)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)

    sw2_vals = []
    for j in range(n_proj):
        proj_a = (a @ dirs[j]).sort().values
        proj_b = (b @ dirs[j]).sort().values
        n = min(len(proj_a), len(proj_b))
        sw2_vals.append(((proj_a[:n] - proj_b[:n])**2).mean().item())

    return sum(sw2_vals) / len(sw2_vals)


def compute_metrics(sample_paths, ref_paths, lik, cond_mean, cond_var, eval_times):
    """
    Compute per-observation-time SW₂ and overall metrics.
    Uses both sample-vs-analytical and sample-vs-reference-samples.
    """
    metrics = {}
    sw2_list = []

    for t_obs, idx, y in lik.obs:
        idx_c = min(idx, sample_paths.shape[1] - 1)
        idx_r = min(idx, ref_paths.shape[1] - 1)
        s = sample_paths[:, idx_c, 0]
        r = ref_paths[:, idx_r, 0]

        # SW₂ vs analytical samples
        sw2_vs_ref = sliced_wasserstein_1d(s, r)
        sw2_list.append(sw2_vs_ref)

        # Find the closest eval_time index for analytical moments
        t_idx = int(round(t_obs / DT))
        t_idx = min(t_idx, len(cond_mean) - 1)
        ana_mean = cond_mean[t_idx]
        ana_std = math.sqrt(max(cond_var[t_idx], 1e-12))

        metrics[f"t={t_obs:.2f}"] = {
            "sw2": sw2_vs_ref,
            "sample_mean": s.mean().item(),
            "sample_std": s.std().item(),
            "analytical_mean": ana_mean,
            "analytical_std": ana_std,
            "mean_error": abs(s.mean().item() - ana_mean),
        }

    metrics["avg_sw2"] = sum(sw2_list) / len(sw2_list) if sw2_list else 0.0
    metrics["per_obs_sw2"] = sw2_list

    # Terminal
    st = sample_paths[:, -1, 0]
    rt = ref_paths[:, -1, 0]
    metrics["terminal_sw2"] = sliced_wasserstein_1d(st, rt)
    metrics["terminal_mean"] = st.mean().item()
    metrics["terminal_std"] = st.std().item()

    # Trajectory-level: mean trajectory vs conditional mean
    n_t = min(sample_paths.shape[1], len(cond_mean))
    sample_traj_mean = sample_paths[:, :n_t, 0].mean(dim=0).cpu().numpy()
    traj_rmse = np.sqrt(((sample_traj_mean - cond_mean[:n_t])**2).mean())
    metrics["traj_rmse"] = traj_rmse

    # Path-space SW₂: flatten full trajectories and compare
    metrics["path_sw2"] = sliced_wasserstein_path(sample_paths, ref_paths)

    return metrics


# ========================================================================
# Networks
# ========================================================================
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h = self.norm(x)
        h = F.silu(self.fc1(h))
        return x + self.fc2(h)


class DriftNet(nn.Module):
    """∂b/∂s network for Algorithm 1: (x, t, s) → ℝ¹."""
    def __init__(self, width=128, depth=4):
        super().__init__()
        self.proj = nn.Linear(3, width)  # x(1) + t(1) + s(1)
        self.blocks = nn.ModuleList([ResBlock(width) for _ in range(depth)])
        self.out = nn.Linear(width, D)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t, s):
        B = x.shape[0]
        dev, dt = x.device, x.dtype
        t_v = torch.full((B, 1), t, device=dev, dtype=dt) if isinstance(t, (int, float)) else t.view(-1, 1)
        s_v = torch.full((B, 1), s, device=dev, dtype=dt) if isinstance(s, (int, float)) else s.view(-1, 1)
        h = F.silu(self.proj(torch.cat([x, t_v, s_v], 1)))
        for blk in self.blocks:
            h = blk(h)
        return self.out(h).clamp(-10., 10.)


class PushNet(nn.Module):
    """Pushforward for Algorithm 3: x → x + f(x, t)."""
    def __init__(self, width=256, depth=4):
        super().__init__()
        self.proj = nn.Linear(D + 1, width)
        self.blocks = nn.ModuleList([ResBlock(width) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, D)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t):
        B = x.shape[0]
        t_v = torch.full((B, 1), t, device=x.device, dtype=x.dtype)
        h = F.silu(self.proj(torch.cat([x, t_v], 1)))
        for blk in self.blocks:
            h = blk(h)
        return x + self.out(self.out_norm(h))


# ========================================================================
# Algorithm 3 (JKO / Wasserstein Gradient Flow)
# ========================================================================
class Algorithm3:
    def __init__(self, lik, h=0.2, width=256, depth=4, lr=1e-3):
        self.lik = lik
        self.h = h
        self.width, self.depth, self.lr = width, depth, lr

    @staticmethod
    def kde_score(samples, bw=None):
        """Leave-one-out KDE score estimator."""
        n = samples.shape[0]
        d = samples.shape[1]
        if bw is None:
            bw = max(n ** (-1.0 / (d + 4)) * samples.std().item(), 0.01)
        diffs = samples.unsqueeze(0) - samples.unsqueeze(1)
        dsq = (diffs**2).sum(2)
        mask = 1.0 - torch.eye(n, device=samples.device)
        w = torch.exp(-dsq / (2 * bw**2)) * mask
        w = w / w.sum(1, keepdim=True).clamp(min=1e-10)
        return -(1.0 / bw**2) * (w.unsqueeze(2) * diffs).sum(1)

    def jko_step(self, particles, t, n_opt):
        """One JKO step from t to t+h."""
        phi = PushNet(self.width, self.depth).to(DEVICE)
        opt_obj = optim.AdamW(phi.parameters(), lr=self.lr, weight_decay=1e-4)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt_obj, n_opt, eta_min=self.lr * 0.05)
        h = self.h
        t_next = t + h

        for _ in range(n_opt):
            xn = phi(particles.detach(), t)
            tc = (1.0 / (2 * h**2 * particles.shape[0])) * ((xn - particles.detach())**2).sum()
            sc = self.kde_score(xn.detach())
            fi = 0.5 * (sc**2).sum(1).mean()
            gV = ou_grad_potential(xn)
            pc = 0.5 * (gV**2).sum(1).mean()
            Jv = self.lik.J_in_interval(xn, t, t_next)
            lc = (2.0 / h) * Jv.mean()
            loss = tc + fi + lc + pc
            if not torch.isfinite(loss):
                continue
            opt_obj.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(phi.parameters(), 5.0)
            opt_obj.step()
            sch.step()

        with torch.no_grad():
            return phi(particles, t)

    def run(self, n_particles=300, n_opt=300):
        """Run full Algorithm 3."""
        n_jko = int(round(T / self.h))
        x = torch.randn(n_particles, D, device=DEVICE) * EQ_STD
        hist = {0.0: x.clone().cpu()}
        t0 = time.time()
        for step in range(n_jko):
            t = step * self.h
            x = self.jko_step(x, t, n_opt)
            t_next = round(t + self.h, 6)
            hist[t_next] = x.clone().cpu()
            print(f"    JKO step {step+1}/{n_jko}, t={t_next:.3f}, "
                  f"mean={x.mean().item():.3f}, std={x.std().item():.3f}")
        elapsed = time.time() - t0

        # Build path-like tensor at DT grid via interpolation
        paths = torch.zeros(n_particles, N_STEPS + 1, D)
        jko_times = sorted(hist.keys())
        for i in range(N_STEPS + 1):
            t_i = i * DT
            t_lo = max([t for t in jko_times if t <= t_i + 1e-8])
            t_hi = min([t for t in jko_times if t >= t_i - 1e-8])
            if abs(t_lo - t_hi) < 1e-8:
                paths[:, i] = hist[t_lo]
            else:
                frac = (t_i - t_lo) / (t_hi - t_lo)
                paths[:, i] = (1 - frac) * hist[t_lo] + frac * hist[t_hi]
        return paths.to(DEVICE), elapsed


# ========================================================================
# Algorithm 1 (Controlled Transport) — FULL BUDGET
# ========================================================================
class Algorithm1:
    def __init__(self, lik, width=128, depth=4, lr=2e-3):
        self.lik = lik
        self.width, self.depth, self.lr = width, depth, lr
        self.increments = []

    def _drift(self, x, t, s):
        b = ou_drift(x)
        for net, ds_v, sv in self.increments:
            b = b + net(x, t, sv) * ds_v
        return b.clamp(-30., 30.)

    def _simulate(self, n, s, x0=None):
        if x0 is None:
            x = torch.randn(n, D, device=DEVICE) * EQ_STD
        else:
            x = x0.clone()
        paths = [x.clone()]
        sq = SIGMA_DIFF * math.sqrt(DT)
        for i in range(N_STEPS):
            t = i * DT
            dr = self._drift(x, t, s)
            x = x + dr * DT + sq * torch.randn_like(x)
            x = x.detach()
            paths.append(x.clone())
        return torch.stack(paths, 1)

    def train_step(self, s, ds_val, K, n_opt):
        phi = DriftNet(self.width, self.depth).to(DEVICE)
        opt_obj = optim.AdamW(phi.parameters(), lr=self.lr, weight_decay=1e-4)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt_obj, n_opt, eta_min=self.lr * 0.05)
        best = float('inf')

        for _ in range(n_opt):
            paths = self._simulate(K, s)
            B = paths.shape[0]
            h_vals = torch.zeros(B, device=DEVICE)

            for i in range(N_STEPS):
                tv = i * DT
                x = paths[:, i].detach().requires_grad_(True)
                dx = (paths[:, i + 1] - paths[:, i]) / DT
                bs = self._drift(x, tv, s)
                po = phi(x, tv, s)
                div_phi = torch.autograd.grad(po.sum(), x, create_graph=True)[0].squeeze(-1)
                res = (bs.detach() - dx.detach()).squeeze(-1)
                h_vals = h_vals + (-DT / 2) * (res * po.squeeze(-1)) - (DT / 4) * div_phi

            # J — ramped point sum (5-step ramp near each obs), no * dt
            J_vals = torch.zeros(B, device=DEVICE)
            for i in range(N_STEPS + 1):
                tv = i * DT
                J_vals = J_vals + self.lik.J_point_ramped(paths[:, i].detach(), tv)

            hbar = h_vals.mean()
            Jm = J_vals.mean()
            loss = ((h_vals - hbar) + (J_vals - Jm)).pow(2).sum()
            if not torch.isfinite(loss):
                continue
            opt_obj.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(phi.parameters(), 5.0)
            opt_obj.step()
            sch.step()
            if loss.item() < best:
                best = loss.item()

        self.increments.append((phi.eval(), ds_val, s))
        return best

    def run(self, n_anneal=30, K=200, n_opt=200):
        """Run Algorithm 1 with full budget (30 anneal, 200 opt)."""
        s_vals = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n_anneal + 1)))
        t0 = time.time()
        for step in range(n_anneal):
            s = s_vals[step]
            ds = s_vals[step + 1] - s_vals[step]
            loss = self.train_step(s, ds, K, n_opt)
            print(f"    Alg1 anneal {step+1}/{n_anneal}, "
                  f"s={s_vals[step+1]:.3f}, loss={loss:.1f}")

        # Final paths (3 batches)
        all_p = []
        for _ in range(3):
            all_p.append(self._simulate(K, 1.0))
        paths = torch.cat(all_p, 0)
        elapsed = time.time() - t0

        # Cleanup GPU
        for net, _, _ in self.increments:
            net.cpu()
        self.increments.clear()
        gc.collect()
        if DEVICE.type == 'mps':
            torch.mps.empty_cache()
        elif DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

        return paths, elapsed


# ========================================================================
# Experiment B: Alg 1 vs Alg 3 across likelihood regimes
# ========================================================================
def run_experiment_B():
    print("\n" + "=" * 70)
    print("EXPERIMENT B: Alg 1 vs Alg 3 across likelihood regimes")
    print("=" * 70)

    # OU equilibrium: N(0, 4), std=2.  Targets must be well outside 1σ
    # to create meaningful constraints.  σ_obs controls tightness.
    # Dense setups: 5 observations forcing rapid marginal reversals
    # Sparse setups: single endpoint, vary only target (keep σ_obs fixed)
    setups = {
        "dense_mild": {
            "obs_times": [0.2, 0.4, 0.6, 0.8, 1.0],
            "obs_targets": [3.0, -3.0, 3.0, -3.0, 3.0],
            "sigma_obs": 0.3,
            "desc": "Dense/mild: symmetric zig-zag ±3, σ=0.3",
        },
        "dense_abrupt": {
            "obs_times": [0.2, 0.4, 0.6, 0.8, 1.0],
            "obs_targets": [5.0, -4.0, 6.0, -3.0, 7.0],
            "sigma_obs": 0.2,
            "desc": "Dense/abrupt: extreme zig-zag, σ=0.2",
        },
        "sparse_mild": {
            "obs_times": [1.0],
            "obs_targets": [4.0],
            "sigma_obs": 0.3,
            "desc": "Sparse/mild: endpoint x=4 (2σ), σ=0.3",
        },
        "sparse_extreme": {
            "obs_times": [1.0],
            "obs_targets": [8.0],
            "sigma_obs": 0.3,
            "desc": "Sparse/extreme: endpoint x=8 (4σ), σ=0.3",
        },
    }

    results_B = {}

    for name, cfg in setups.items():
        print(f"\n{'='*60}")
        print(f"Setup: {cfg['desc']}")
        print(f"{'='*60}")

        lik = Likelihood(cfg["obs_times"], cfg["obs_targets"], cfg["sigma_obs"])

        # Analytical reference
        print("  Computing analytical reference...")
        ref_paths, eval_times, cond_mean, cond_var = sample_ou_conditional(
            500, cfg["obs_times"], cfg["obs_targets"], cfg["sigma_obs"]
        )
        for t_obs, target in zip(cfg["obs_times"], cfg["obs_targets"]):
            t_idx = int(round(t_obs / DT))
            print(f"  Analytical @ t={t_obs:.1f}: mean={cond_mean[t_idx]:.3f}, "
                  f"std={math.sqrt(max(cond_var[t_idx], 0)):.3f} (target={target:.1f})")

        setup_result = {
            "desc": cfg["desc"],
            "obs_times": cfg["obs_times"],
            "obs_targets": cfg["obs_targets"],
            "sigma_obs": cfg["sigma_obs"],
            "analytical_cond_mean": {
                f"t={t:.2f}": float(cond_mean[int(round(t/DT))])
                for t in cfg["obs_times"]
            },
            "analytical_cond_std": {
                f"t={t:.2f}": float(math.sqrt(max(cond_var[int(round(t/DT))], 0)))
                for t in cfg["obs_times"]
            },
        }

        # Algorithm 3 — h aligned with obs grid
        print(f"\n  --- Algorithm 3 ---")
        if len(cfg["obs_times"]) > 1:
            h3 = cfg["obs_times"][0]  # = 0.2
        else:
            h3 = 0.2
        alg3 = Algorithm3(lik, h=h3, width=256, depth=4, lr=1e-3)
        a3_paths, a3_time = alg3.run(n_particles=300, n_opt=300)
        a3_metrics = compute_metrics(a3_paths, ref_paths, lik, cond_mean, cond_var, eval_times)
        a3_metrics["time_s"] = a3_time
        a3_metrics["h"] = h3
        setup_result["alg3"] = a3_metrics
        print(f"  Alg3: avg_sw2={a3_metrics['avg_sw2']:.5f}, "
              f"path_sw2={a3_metrics['path_sw2']:.5f}, "
              f"traj_rmse={a3_metrics['traj_rmse']:.5f}, "
              f"terminal_mean={a3_metrics['terminal_mean']:.3f}, "
              f"time={a3_time:.1f}s")
        del alg3, a3_paths
        gc.collect()

        # Algorithm 1 — FULL BUDGET: 30 annealing, 200 opt
        print(f"\n  --- Algorithm 1 (30 anneal, 200 opt) ---")
        alg1 = Algorithm1(lik, width=128, depth=4, lr=2e-3)
        a1_paths, a1_time = alg1.run(n_anneal=30, K=200, n_opt=200)
        a1_metrics = compute_metrics(a1_paths, ref_paths, lik, cond_mean, cond_var, eval_times)
        a1_metrics["time_s"] = a1_time
        setup_result["alg1"] = a1_metrics
        print(f"  Alg1: avg_sw2={a1_metrics['avg_sw2']:.5f}, "
              f"path_sw2={a1_metrics['path_sw2']:.5f}, "
              f"traj_rmse={a1_metrics['traj_rmse']:.5f}, "
              f"terminal_mean={a1_metrics['terminal_mean']:.3f}, "
              f"time={a1_time:.1f}s")
        del alg1, a1_paths
        gc.collect()

        results_B[name] = setup_result
        del ref_paths
        gc.collect()

    return results_B


# ========================================================================
# Plotting — Experiment B
# ========================================================================
def plot_experiment_B(results_B, save_dir):
    """
    Alg 1 vs Alg 3 comparison with:
      (1) per-obs SW₂ grouped bars
      (2) conditional mean overlay
      (3) cost comparison
      (4) summary table
    """
    setup_names = list(results_B.keys())
    n_setups = len(setup_names)

    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, n_setups, hspace=0.4, wspace=0.3)

    # Row 1: Conditional mean trajectory overlay (one panel per setup)
    eval_t = np.arange(N_STEPS + 1) * DT
    for col, sname in enumerate(setup_names):
        ax = fig.add_subplot(gs[0, col])
        r = results_B[sname]

        # Analytical conditional mean ± std
        ana_means = []
        ana_stds = []
        for t in eval_t:
            t_key = f"t={t:.2f}"
            # Recompute analytical moments for this setup
            pass
        # We need the full analytical trajectory. Recompute.
        _, cm, cv = ou_conditional_moments(
            r["obs_times"], r["obs_targets"], r["sigma_obs"]
        )
        n_t = min(len(eval_t), len(cm))
        ax.fill_between(eval_t[:n_t],
                        cm[:n_t] - 2*np.sqrt(np.maximum(cv[:n_t], 0)),
                        cm[:n_t] + 2*np.sqrt(np.maximum(cv[:n_t], 0)),
                        alpha=0.15, color='gray', label='Analytical ±2σ')
        ax.plot(eval_t[:n_t], cm[:n_t], 'k-', lw=2, label='Analytical mean')

        # Algorithm sample means
        # We don't have the paths anymore, but we have per-obs means
        a1_means = [r["alg1"][f"t={t:.2f}"]["sample_mean"]
                     for t in r["obs_times"]]
        a3_means = [r["alg3"][f"t={t:.2f}"]["sample_mean"]
                     for t in r["obs_times"]]
        ax.plot(r["obs_times"], a1_means, 'b^-', ms=8, lw=1.5, label='Alg 1')
        ax.plot(r["obs_times"], a3_means, 'o-', color='#FF9800', ms=8, lw=1.5, label='Alg 3')

        # Observation targets
        ax.scatter(r["obs_times"], r["obs_targets"], marker='*',
                   s=150, color='red', zorder=10, label='Targets')

        ax.set_title(r["desc"].split(":")[0], fontsize=11, fontweight='bold')
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        if col == 0:
            ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.2)

    # Row 2: Per-observation SW₂ breakdown (one panel per setup)
    for col, sname in enumerate(setup_names):
        ax = fig.add_subplot(gs[1, col])
        r = results_B[sname]
        obs_t = r["obs_times"]
        n_obs = len(obs_t)

        a1_sw2 = [r["alg1"][f"t={t:.2f}"]["sw2"] for t in obs_t]
        a3_sw2 = [r["alg3"][f"t={t:.2f}"]["sw2"] for t in obs_t]

        x_pos = np.arange(n_obs)
        w = 0.35
        ax.bar(x_pos - w/2, a1_sw2, w, label='Alg 1', color='#2196F3', alpha=0.85)
        ax.bar(x_pos + w/2, a3_sw2, w, label='Alg 3', color='#FF9800', alpha=0.85)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f't={t:.1f}\ny={r["obs_targets"][i]:.0f}'
                            for i, t in enumerate(obs_t)], fontsize=8)
        ax.set_ylabel('SW₂', fontsize=10)
        ax.set_title(f'{r["desc"].split(":")[0]}: Per-obs SW₂', fontsize=10)
        if col == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis='y')

    # Row 3: Summary panels
    # (0) Avg SW₂ bar chart
    ax = fig.add_subplot(gs[2, 0])
    x_pos = np.arange(n_setups)
    w = 0.35
    a1_sw2_avg = [results_B[s]["alg1"]["avg_sw2"] for s in setup_names]
    a3_sw2_avg = [results_B[s]["alg3"]["avg_sw2"] for s in setup_names]
    bars1 = ax.bar(x_pos - w/2, a1_sw2_avg, w, label='Alg 1', color='#2196F3', alpha=0.85)
    bars2 = ax.bar(x_pos + w/2, a3_sw2_avg, w, label='Alg 3', color='#FF9800', alpha=0.85)
    ax.set_xticks(x_pos)
    labels = [results_B[s]["desc"].split(":")[0] for s in setup_names]
    ax.set_xticklabels(labels, fontsize=8, rotation=15)
    ax.set_ylabel('Avg Marginal SW₂')
    ax.set_title('Accuracy: Avg SW₂', fontsize=11)
    ax.legend(fontsize=8)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)

    # (1) Trajectory RMSE
    ax = fig.add_subplot(gs[2, 1])
    a1_rmse = [results_B[s]["alg1"]["traj_rmse"] for s in setup_names]
    a3_rmse = [results_B[s]["alg3"]["traj_rmse"] for s in setup_names]
    ax.bar(x_pos - w/2, a1_rmse, w, label='Alg 1', color='#2196F3', alpha=0.85)
    ax.bar(x_pos + w/2, a3_rmse, w, label='Alg 3', color='#FF9800', alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=8, rotation=15)
    ax.set_ylabel('Trajectory RMSE')
    ax.set_title('Mean trajectory vs analytical', fontsize=11)
    ax.legend(fontsize=8)

    # (2) Wall-clock time
    ax = fig.add_subplot(gs[2, 2])
    a1_t = [results_B[s]["alg1"]["time_s"] for s in setup_names]
    a3_t = [results_B[s]["alg3"]["time_s"] for s in setup_names]
    ax.bar(x_pos - w/2, a1_t, w, label='Alg 1', color='#2196F3', alpha=0.85)
    ax.bar(x_pos + w/2, a3_t, w, label='Alg 3', color='#FF9800', alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=8, rotation=15)
    ax.set_ylabel('Time (s)')
    ax.set_title('Cost: Wall-clock time', fontsize=11)
    ax.legend(fontsize=8)
    ax.set_yscale('log')

    # (3) Summary table
    ax = fig.add_subplot(gs[2, 3])
    ax.axis('off')
    cell_text = []
    for s in setup_names:
        r = results_B[s]
        a1_w = "✓" if r['alg1']['avg_sw2'] < r['alg3']['avg_sw2'] else ""
        a3_w = "✓" if r['alg3']['avg_sw2'] < r['alg1']['avg_sw2'] else ""
        row = [
            r["desc"].split(":")[0],
            f"{r['alg1']['avg_sw2']:.4f} {a1_w}",
            f"{r['alg3']['avg_sw2']:.4f} {a3_w}",
            f"{r['alg1']['path_sw2']:.4f}",
            f"{r['alg3']['path_sw2']:.4f}",
            f"{r['alg1']['traj_rmse']:.4f}",
            f"{r['alg3']['traj_rmse']:.4f}",
            f"{r['alg1']['time_s']:.0f}",
            f"{r['alg3']['time_s']:.0f}",
        ]
        cell_text.append(row)
    col_labels = ['Setup', 'A1 SW₂', 'A3 SW₂', 'A1 pSW₂', 'A3 pSW₂',
                  'A1 RMSE', 'A3 RMSE', 'A1 (s)', 'A3 (s)']
    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#E8E8E8')
    # Highlight winner
    for i, s in enumerate(setup_names):
        a1 = results_B[s]["alg1"]["avg_sw2"]
        a3 = results_B[s]["alg3"]["avg_sw2"]
        winner_col = 1 if a1 < a3 else 2
        table[i + 1, winner_col].set_facecolor('#C8E6C9')
    ax.set_title('Summary', fontsize=11, pad=15)

    fig.suptitle(
        'Algorithm 1 vs Algorithm 3: When is each preferable?\n'
        'Reference: analytical Gaussian posterior (Kalman smoother) | '
        'Alg 1: 30 anneal, 200 opt | Alg 3: h=0.2, 300 particles, 300 opt',
        fontsize=12, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    path = os.path.join(save_dir, "alg_comparison.pdf")
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ========================================================================
# Main
# ========================================================================
if __name__ == "__main__":
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Experiment B: Algorithm 1 vs Algorithm 3 across likelihood regimes
    results_B = run_experiment_B()
    plot_experiment_B(results_B, RESULTS_DIR)

    # Save JSON
    json_path = os.path.join(RESULTS_DIR, "alg_comparison.json")
    with open(json_path, "w") as f:
        json.dump(results_B, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
    print(f"\nSaved results to {json_path}")

    print("\n" + "=" * 70)
    print("SUMMARY: Alg 1 vs Alg 3 (analytical Kalman reference)")
    print("=" * 70)
    for name, r in results_B.items():
        a1 = r["alg1"]["avg_sw2"]
        a3 = r["alg3"]["avg_sw2"]
        winner = "Alg1" if a1 < a3 else "Alg3"
        print(f"  {r['desc'].split(':')[0]:20s}: "
              f"Alg1 marg_sw2={a1:.4f} path_sw2={r['alg1']['path_sw2']:.4f} "
              f"RMSE={r['alg1']['traj_rmse']:.4f} ({r['alg1']['time_s']:.0f}s), "
              f"Alg3 marg_sw2={a3:.4f} path_sw2={r['alg3']['path_sw2']:.4f} "
              f"RMSE={r['alg3']['traj_rmse']:.4f} ({r['alg3']['time_s']:.0f}s) "
              f"→ {winner} wins (marginal)")
