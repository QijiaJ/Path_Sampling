#!/usr/bin/env python3
"""
Baseline Comparison on 2D Double-Well Transition Path Sampling 

Methods compared:
  1. Algorithm 1 (Controlled Transport) without SPDE refinement
  2. Algorithm 1 (Controlled Transport) with SPDE refinement steps
  3. Path-space Sinkhorn / IPF (Diffusion Schrödinger Bridge)

Setup: 2D double-well V(x) = scale*(x1^2 - 1)^2 + 0.5*x2^2
  - Genuine 2D structure: DW in x1, harmonic confinement in x2
  - Rare-event regime: scale=3.5, T=0.5
  - Well A = (-1, 0), Well B = (1, 0)

Metrics: path-space sliced Wasserstein, OM action distribution,
         dense marginal evaluation, barrier crossing fraction,
         transition state density at x1≈0.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
import os
import gc
import tracemalloc

# Auto-detect device
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")


device = torch.device('cpu') 

torch.manual_seed(42)
np.random.seed(42)

# =============================================================================
# Problem setup
# =============================================================================
d = 2
T = 0.5
dt = 0.01
n_time_steps = int(T / dt)  # 50
SCALE = 3.5     # barrier height ΔV = 3.5 
HARMONIC = 0.5  # harmonic confinement in x2

A = torch.tensor([-1.0, 0.0], device=device)
B = torch.tensor([ 1.0, 0.0], device=device)


def potential(x):
    """V(x) = scale*(x1^2-1)^2 + 0.5*x2^2."""
    return SCALE * (x[..., 0]**2 - 1)**2 + HARMONIC * x[..., 1]**2


def grad_potential(x):
    """∇V(x). Acts on both coordinates."""
    g = torch.zeros_like(x)
    g[..., 0] = 4 * SCALE * x[..., 0] * (x[..., 0]**2 - 1)
    g[..., 1] = 2 * HARMONIC * x[..., 1]
    return g


def dw_drift(x, t=None):
    """Double-well drift = -∇V, clamped for stability."""
    return (-grad_potential(x)).clamp(-15.0, 15.0)


def terminal_log_likelihood(x, sigma):
    """Log-likelihood: -1/(2σ²) ||x - B||²."""
    diff = x - B.unsqueeze(0) if x.dim() == 2 else x - B
    return -(1.0 / (2 * sigma**2)) * (diff**2).sum(dim=-1)


def terminal_J(x, sigma):
    """J(x) = 1/(2σ²) ||x - B||² (positive, to be minimized)."""
    diff = x - B.unsqueeze(0) if x.dim() == 2 else x - B
    return (1.0 / (2 * sigma**2)) * (diff**2).sum(dim=-1)


# =============================================================================
# Shared utilities
# =============================================================================
def simulate_dw_paths(x0, n_steps, dt_val, drift_fn=dw_drift, clamp_val=5.0):
    """Simulate DW SDE paths. Returns (batch, n_steps+1, d)."""
    x = x0.clone()
    paths = [x.clone()]
    for i in range(n_steps):
        t = i * dt_val
        noise = torch.randn_like(x) * np.sqrt(2.0 * dt_val)
        x = x + drift_fn(x, t) * dt_val + noise
        x = x.clamp(-clamp_val, clamp_val)
        paths.append(x.clone())
    return torch.stack(paths, dim=1)


def onsager_machlup_action(paths, dt_val):
    """
    Compute OM action: I(x) = 1/2 ∫₀ᵀ (1/2 ||dx/dt - u^ref||² + 1/2 div u^ref) dt.
    Simplified: just the kinetic part for comparison.
    paths: (batch, n_steps+1, d)
    """
    batch = paths.shape[0]
    action = torch.zeros(batch, device=paths.device)
    for i in range(paths.shape[1] - 1):
        x = paths[:, i]
        dx = (paths[:, i+1] - paths[:, i]) / dt_val
        drift = dw_drift(x)
        action += 0.5 * ((dx - drift)**2).sum(dim=-1) * dt_val
    return action


# =============================================================================
# Comprehensive metrics
# =============================================================================
def compute_all_metrics(sample_paths, ref_paths, dt_val, label=""):
    """
    Compute a comprehensive set of metrics comparing sample_paths to ref_paths.
    Both: (n, n_steps+1, d)
    """
    n_s, n_t, d_dim = sample_paths.shape
    n_r = ref_paths.shape[0]
    results = {}

    # 1. Dense marginal evaluation (every 5th time step)
    eval_indices = list(range(5, n_t - 1, 5))  # skip endpoints
    marginal_metrics = {}
    for t_idx in eval_indices:
        x_s = sample_paths[:, t_idx]
        x_r = ref_paths[:, t_idx]

        # MMD²
        sigma_k = 1.0
        xx = torch.cdist(x_s, x_s)**2
        rr = torch.cdist(x_r, x_r)**2
        xr = torch.cdist(x_s, x_r)**2
        kxx = torch.exp(-xx / (2*sigma_k**2)).mean()
        krr = torch.exp(-rr / (2*sigma_k**2)).mean()
        kxr = torch.exp(-xr / (2*sigma_k**2)).mean()
        mmd2 = (kxx + krr - 2*kxr).item()

        # Sliced Wasserstein
        n_proj = 100
        projs = torch.randn(n_proj, d_dim, device=x_s.device)
        projs = projs / projs.norm(dim=1, keepdim=True)
        n_min = min(n_s, n_r)
        proj_s = (x_s[:n_min] @ projs.T).sort(dim=0).values
        proj_r = (x_r[:n_min] @ projs.T).sort(dim=0).values
        sw2 = ((proj_s - proj_r)**2).mean().item()

        t_val = t_idx * dt_val
        marginal_metrics[f"t={t_val:.2f}"] = {"mmd2": round(mmd2, 5), "sw2": round(sw2, 5)}

    results["marginal_metrics"] = marginal_metrics

    # 2. Path-space sliced Wasserstein (flatten full paths)
    # Subsample time points for tractability
    t_sub = list(range(0, n_t, max(1, n_t//20)))
    flat_s = sample_paths[:, t_sub, :].reshape(n_s, -1)  # (n, 20*d)
    flat_r = ref_paths[:, t_sub, :].reshape(n_r, -1)
    n_min = min(n_s, n_r)
    d_flat = flat_s.shape[1]
    n_proj = 200
    projs = torch.randn(n_proj, d_flat, device=flat_s.device)
    projs = projs / projs.norm(dim=1, keepdim=True)
    proj_s = (flat_s[:n_min] @ projs.T).sort(dim=0).values
    proj_r = (flat_r[:n_min] @ projs.T).sort(dim=0).values
    path_sw2 = ((proj_s - proj_r)**2).mean().item()
    results["path_sw2"] = round(path_sw2, 5)

    # 3. Onsager-Machlup action distribution
    om_sample = onsager_machlup_action(sample_paths, dt_val)
    om_ref = onsager_machlup_action(ref_paths, dt_val)
    results["om_action"] = {
        "sample_mean": round(om_sample.mean().item(), 3),
        "sample_std": round(om_sample.std().item(), 3),
        "ref_mean": round(om_ref.mean().item(), 3),
        "ref_std": round(om_ref.std().item(), 3),
    }

    # 4. Barrier crossing fraction (what % of paths have x1 > 0 at some point past midway)
    x1_second_half = sample_paths[:, n_t//2:, 0]  # x1 in second half
    crossed = (x1_second_half > 0).any(dim=1).float().mean().item()
    results["barrier_crossing_frac"] = round(crossed, 3)

    x1_ref_second = ref_paths[:, n_t//2:, 0]
    ref_crossed = (x1_ref_second > 0).any(dim=1).float().mean().item()
    results["ref_crossing_frac"] = round(ref_crossed, 3)

    # 5. Transition state density at x1 ≈ 0 (barrier top)
    # Look at the time point closest to t=T/2 and measure spread around x1=0
    mid_idx = n_t // 2
    x1_mid_sample = sample_paths[:, mid_idx, 0]
    x1_mid_ref = ref_paths[:, mid_idx, 0]
    # Fraction of paths near barrier top |x1| < 0.3 at midpoint
    near_barrier_s = (x1_mid_sample.abs() < 0.3).float().mean().item()
    near_barrier_r = (x1_mid_ref.abs() < 0.3).float().mean().item()
    results["transition_state"] = {
        "sample_frac_near_barrier": round(near_barrier_s, 3),
        "ref_frac_near_barrier": round(near_barrier_r, 3),
        "sample_x1_mid_mean": round(x1_mid_sample.mean().item(), 3),
        "ref_x1_mid_mean": round(x1_mid_ref.mean().item(), 3),
    }

    # 6. Terminal statistics
    term_s = sample_paths[:, -1]
    term_r = ref_paths[:, -1]
    results["terminal"] = {
        "sample_x1_mean": round(term_s[:, 0].mean().item(), 3),
        "sample_x2_mean": round(term_s[:, 1].mean().item(), 3),
        "ref_x1_mean": round(term_r[:, 0].mean().item(), 3),
        "ref_x2_mean": round(term_r[:, 1].mean().item(), 3),
        "mean_dist_to_B": round(((term_s - B.cpu().unsqueeze(0))**2).sum(1).sqrt().mean().item(), 3),
    }

    if label:
        print(f"  [{label}] path_SW2={path_sw2:.4f}, crossing={crossed:.1%}, "
              f"terminal_x1={term_s[:,0].mean():.3f}, OM_action={om_sample.mean():.1f}")

    return results


# =============================================================================
# Generate reference paths via large-scale IS
# =============================================================================
print("=" * 70)
print("Generating reference paths via importance-weighted resampling...")
print("=" * 70)

n_ref_proposals = 50000  # need many more proposals for rare-event regime
x0_ref = A.unsqueeze(0).expand(n_ref_proposals, -1).clone()
ref_paths_all = simulate_dw_paths(x0_ref, n_time_steps, dt)

# Compute importance weights — terminal proximity to B
sigma_ref = 0.15  # tighter constraint for rare-event regime
log_w = terminal_log_likelihood(ref_paths_all[:, -1], sigma=sigma_ref)
log_w = log_w - log_w.max()
weights = torch.exp(log_w)
weights = weights / weights.sum()
ess_ref = (1.0 / (weights**2).sum()).item()
print(f"  Reference IS: ESS = {ess_ref:.0f}/{n_ref_proposals}")

# Resample
n_ref = 500
indices = torch.multinomial(weights, n_ref, replacement=True)
ref_paths = ref_paths_all[indices].cpu()

ref_term = ref_paths[:, -1]
print(f"  Ref terminal mean: ({ref_term[:,0].mean():.3f}, {ref_term[:,1].mean():.3f})")
print(f"  Ref crossing frac: {(ref_paths[:, n_time_steps//2:, 0] > 0).any(dim=1).float().mean():.1%}")

# Clean up large tensor
del ref_paths_all
gc.collect()


# =============================================================================
# 1. Algorithm 1 (with and without SPDE refinement)
# =============================================================================


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        h = self.norm(x)
        h = torch.nn.functional.silu(self.fc1(h))
        return x + self.fc2(h)

class DriftNet(nn.Module):
    """Drift network: (x, t, s) -> R^d with ResBlocks + output clamping."""
    def __init__(self, d_in, hidden=128, depth=4):
        super().__init__()
        self.proj = nn.Linear(d_in + 2, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, d_in)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t_val, s_val):
        batch = x.shape[0]
        t_enc = torch.full((batch, 1), float(t_val), device=x.device)
        s_enc = torch.full((batch, 1), float(s_val), device=x.device)
        h = torch.nn.functional.silu(self.proj(torch.cat([x, t_enc, s_enc], dim=1)))
        for blk in self.blocks:
            h = blk(h)
        return self.out(self.out_norm(h)).clamp(-10.0, 10.0)


def spde_refinement_step(paths, s, sigma_s, n_spde=5):
    """
    Run a few Langevin SPDE steps on path space to correct
    the lag ρ_s ≠ π_s after an annealing increment.

    The variational derivative δ_x log π_s(x) has three terms:
      (1) OM kinetic: -(1/2)(d²x/dt² - ∇drift · drift) [Laplacian in path space]
      (2) Potential drift correction: from the reference SDE
      (3) Likelihood: s · ∇_x J_t(x) at observation times

    We use a simplified version: just move paths toward lower
    OM action + likelihood via gradient on the discretized path.
    """
    paths_out = paths.clone()
    batch, n_t, d_dim = paths_out.shape
    ds_spde = dt**2  # SPDE time step

    for _ in range(n_spde):
        # Force from OM action: encourage smoother paths consistent with drift
        force = torch.zeros_like(paths_out)
        for i in range(1, n_t - 1):
            x_prev = paths_out[:, i-1].detach()
            x_curr = paths_out[:, i].detach()
            x_next = paths_out[:, i+1].detach()

            # Laplacian term (path-space diffusion)
            laplacian = (x_next - 2*x_curr + x_prev) / dt**2

            # Drift consistency: push toward reference drift
            drift_at_curr = dw_drift(x_curr)
            velocity = (x_next - x_curr) / dt
            drift_force = -(velocity - drift_at_curr) / dt

            # Likelihood force at all times (weighted by s)
            J_force = torch.zeros_like(x_curr)
            J_force[:, 0] = -s / sigma_s**2 * (x_curr[:, 0] - B[0])
            J_force[:, 1] = -s / sigma_s**2 * (x_curr[:, 1] - B[1])
            # Only apply terminal-ish force in last 20% of path
            if i >= int(0.8 * n_t):
                force[:, i] = 0.3 * laplacian + 0.3 * drift_force + J_force
            else:
                force[:, i] = 0.3 * laplacian + 0.3 * drift_force

        # Langevin update with noise
        noise = torch.randn_like(paths_out) * np.sqrt(2 * ds_spde)
        paths_out = paths_out + force * ds_spde + noise
        paths_out = paths_out.clamp(-5.0, 5.0)

        # Pin initial condition
        paths_out[:, 0] = A.unsqueeze(0).expand(batch, -1)

    return paths_out


def run_algorithm1(use_spde=True):
    tag = "+SPDE" if use_spde else "no SPDE"
    n_annealing = 30       # more steps for harder barrier
    n_paths = 500
    n_opt = 200
    n_spde_refine = 8 if use_spde else 0

    # Cosine annealing schedule (spends more iterations at small s)
    s_vals = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n_annealing + 1)))

    # Fixed sigma for J penalty — must match reference constraint strength.
    # The annealing parameter s already controls interpolation; adaptive sigma
    # double-dips and makes early J negligible (1/(2*1.5²)=0.22 vs 1/(2*0.15²)=22.2).
    sigma_obs = 0.15  # matches run_dw_d12.py

    drift_increments = []

    def current_drift_fn(x, t_val, s):
        b = dw_drift(x, t_val)
        for net, ds, sv in drift_increments:
            b = b + net(x, t_val, sv) * ds
        return b.clamp(-15.0, 15.0)  # Safety clamp on total drift

    def simulate_with_drift(n, s):
        x = A.unsqueeze(0).expand(n, -1).clone()
        paths = [x.clone()]
        for i in range(n_time_steps):
            t_val = i * dt
            drift = current_drift_fn(x, t_val, s)
            noise = torch.randn_like(x) * np.sqrt(2.0 * dt)
            x = x + drift * dt + noise
            x = x.clamp(-5.0, 5.0)
            paths.append(x.clone())
        return torch.stack(paths, dim=1)

    tracemalloc.start()
    start = time.time()

    for step in range(n_annealing):
        s = s_vals[step]
        delta_s = s_vals[step + 1] - s_vals[step]
        sigma_s = sigma_obs  # fixed constraint strength

        phi = DriftNet(d).to(device)
        optimizer = optim.AdamW(phi.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_opt, eta_min=1e-4)
        nan_count = 0

        for opt_step in range(n_opt):
            # Simulate paths
            with torch.no_grad():
                paths = simulate_with_drift(n_paths, s)

            K = paths.shape[0]

            # Compute h_theta^s terms
            h_vals = torch.zeros(K, device=device)
            for i in range(n_time_steps):
                t_val = i * dt
                x = paths[:, i].detach().requires_grad_(True)
                dx_dt = (paths[:, i+1].detach() - paths[:, i].detach()) / dt
                b_s = current_drift_fn(x, t_val, s)
                phi_out = phi(x, t_val, s)

                # Divergence via autograd
                div_phi = torch.zeros(K, device=device)
                for dim_j in range(d):
                    grad_phi_dim = torch.autograd.grad(
                        phi_out[:, dim_j].sum(), x, create_graph=True
                    )[0][:, dim_j]
                    div_phi += grad_phi_dim

                residual = (b_s.detach() - dx_dt)
                h_vals += (-dt / 2) * (residual * phi_out).sum(dim=1) - (dt / 4) * div_phi

            # J values with current sigma_s — point penalty at terminal
            # Terminal ramp over last 5 time steps (matching run_dw_d12.py)
            # Each step contributes the full J scaled by an increasing weight
            J_vals = torch.zeros(K, device=device)
            ramp_len = 5
            ramp_start_idx = max(0, n_time_steps - ramp_len)
            for i in range(ramp_start_idx, n_time_steps):
                weight = (i - ramp_start_idx + 1) / ramp_len  # 0.2, 0.4, 0.6, 0.8, 1.0
                J_vals += weight * terminal_J(paths[:, i].detach(), sigma=sigma_s)
            J_mean = J_vals.mean()

            h_bar = h_vals.mean()
            residuals = (h_vals - h_bar) + (J_vals - J_mean)
            loss = (residuals**2).sum()

            # NaN/inf watchdog
            if not torch.isfinite(loss):
                nan_count += 1
                if nan_count > 5:
                    print(f"    WARNING: {nan_count} NaN/inf losses, stopping early")
                    break
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(phi.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

        drift_increments.append((phi.eval(), delta_s, s))

        # === SPDE refinement steps ===
        if n_spde_refine > 0 and step > 0:
            with torch.no_grad():
                test_paths = simulate_with_drift(min(200, n_paths), s_vals[step+1])
                test_paths = spde_refinement_step(
                    test_paths, s_vals[step+1], sigma_obs, n_spde=n_spde_refine
                )
                crossing = (test_paths[:, n_time_steps//2:, 0] > 0).any(dim=1).float().mean()
        else:
            crossing = torch.tensor(0.0)

        print(f"  Alg1({tag}) step {step+1}/{n_annealing}, s={s_vals[step+1]:.3f}, "
              f"Δs={delta_s:.4f}, σ={sigma_s:.2f}, loss={loss.item():.2f}, crossing={crossing:.1%}")

    alg1_time = time.time() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    alg1_mem = peak / 1024 / 1024

    # Generate final paths at s=1
    with torch.no_grad():
        alg1_paths = simulate_with_drift(n_ref, 1.0)
        if use_spde:
            # Final SPDE polish
            alg1_paths = spde_refinement_step(
                alg1_paths, 1.0, sigma_obs, n_spde=10
            )

    terminal_mean = alg1_paths[:, -1].mean(dim=0)
    crossing = (alg1_paths[:, n_time_steps//2:, 0] > 0).any(dim=1).float().mean()
    print(f"  Alg1({tag}) terminal mean: ({terminal_mean[0]:.3f}, {terminal_mean[1]:.3f})")
    print(f"  Alg1({tag}) crossing frac: {crossing:.1%}")
    print(f"  Alg1({tag}) time: {alg1_time:.1f}s, memory: {alg1_mem:.1f}MB")

    # Free drift networks from GPU before returning
    for net, _, _ in drift_increments:
        net.cpu()
    del drift_increments
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()

    return alg1_paths.cpu(), alg1_time, alg1_mem


# Run Algorithm 1 without SPDE
print("\n" + "=" * 70)
print("Running Algorithm 1 (Controlled Transport, no SPDE)...")
print("=" * 70)
alg1_paths, alg1_time, alg1_mem = run_algorithm1(use_spde=False)

# Force cleanup before second run
gc.collect()
if device.type == 'mps':
    torch.mps.empty_cache()

# Run Algorithm 1 with SPDE refinement
print("\n" + "=" * 70)
print("Running Algorithm 1 (Controlled Transport + SPDE refinement)...")
print("=" * 70)
alg1_spde_paths, alg1_spde_time, alg1_spde_mem = run_algorithm1(use_spde=True)


# =============================================================================
# 2. Importance Sampling
# =============================================================================
print("\n" + "=" * 70)
print("Running Importance Sampling baseline...")
print("=" * 70)


def run_importance_sampling():
    n_proposals = 20000  # generous pool for rare-event regime
    n_resample = 500

    tracemalloc.start()
    start = time.time()

    x0_is = A.unsqueeze(0).expand(n_proposals, -1).clone()
    is_paths = simulate_dw_paths(x0_is, n_time_steps, dt)

    terminal = is_paths[:, -1]
    log_w = terminal_log_likelihood(terminal, sigma=0.15)
    log_w = log_w - log_w.max()
    weights = torch.exp(log_w)
    weights = weights / weights.sum()

    ess = (1.0 / (weights**2).sum()).item()
    print(f"  IS: ESS = {ess:.0f}/{n_proposals}")

    indices = torch.multinomial(weights, n_resample, replacement=True)
    is_resampled = is_paths[indices].cpu()

    is_time = time.time() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    is_mem = peak / 1024 / 1024

    terminal_mean = is_resampled[:, -1].mean(dim=0)
    crossing = (is_resampled[:, n_time_steps//2:, 0] > 0).any(dim=1).float().mean()
    print(f"  IS terminal: ({terminal_mean[0]:.3f}, {terminal_mean[1]:.3f})")
    print(f"  IS crossing: {crossing:.1%}, ESS={ess:.0f}")
    print(f"  IS time: {is_time:.1f}s, memory: {is_mem:.1f}MB")

    del is_paths; gc.collect()
    return is_resampled, is_time, is_mem, ess


is_paths, is_time, is_mem, is_ess = run_importance_sampling()


# =============================================================================
# 3. Sinkhorn/IPF (Diffusion Schrödinger Bridge)
# =============================================================================
print("\n" + "=" * 70)
print("Running Sinkhorn/IPF (Diffusion Schrödinger Bridge)...")
print("=" * 70)


class DSBDriftNetwork(nn.Module):
    """Forward drift = ref_drift + learned correction. Zero-init correction."""
    def __init__(self, d_in, n_steps, hidden=128, depth=3):
        super().__init__()
        self.time_embed = nn.Embedding(n_steps + 2, 32)
        layers = [nn.Linear(d_in + 32, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, d_in))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, k):
        batch = x.shape[0]
        k_t = torch.full((batch,), k, dtype=torch.long, device=x.device)
        return self.net(torch.cat([x, self.time_embed(k_t)], dim=1))

    def full_drift(self, x, k):
        return dw_drift(x) + self.forward(x, k)


class DSBBackwardNetwork(nn.Module):
    """Backward drift = -ref_drift + learned correction."""
    def __init__(self, d_in, n_steps, hidden=128, depth=3):
        super().__init__()
        self.time_embed = nn.Embedding(n_steps + 2, 32)
        layers = [nn.Linear(d_in + 32, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, d_in))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, k):
        batch = x.shape[0]
        k_t = torch.full((batch,), k, dtype=torch.long, device=x.device)
        return self.net(torch.cat([x, self.time_embed(k_t)], dim=1))

    def full_drift(self, x, k):
        return -dw_drift(x) + self.forward(x, k)


def run_sinkhorn_ipf():
    """DSB Algorithm 1 (De Bortoli et al. 2021) with eqs (12)-(13)."""
    n_ipf_iters = 16     # more iterations for harder barrier
    n_train_steps = 400
    n_sample = 500
    lr = 3e-4
    sigma_d = np.sqrt(2.0 * dt)

    F_net = DSBDriftNetwork(d, n_time_steps, hidden=128, depth=3).to(device)
    B_net = DSBBackwardNetwork(d, n_time_steps, hidden=128, depth=3).to(device)
    F_opt = optim.Adam(F_net.parameters(), lr=lr)
    B_opt = optim.Adam(B_net.parameters(), lr=lr)

    tracemalloc.start()
    start = time.time()

    def sample_forward(n, f_net):
        x = A.unsqueeze(0).expand(n, -1).clone()
        paths = [x.clone()]
        for k in range(n_time_steps):
            x = x + f_net.full_drift(x, k) * dt + sigma_d * torch.randn_like(x)
            x = x.clamp(-5.0, 5.0)
            paths.append(x.clone())
        return torch.stack(paths, dim=1)

    def sample_backward(n, b_net):
        x = B.unsqueeze(0).expand(n, -1).clone()
        paths = [None] * (n_time_steps + 1)
        paths[n_time_steps] = x.clone()
        for k in range(n_time_steps - 1, -1, -1):
            x = x + b_net.full_drift(x, k+1) * dt + sigma_d * torch.randn_like(x)
            x = x.clamp(-5.0, 5.0)
            paths[k] = x.clone()
        return torch.stack(paths, dim=1)

    for ipf_iter in range(n_ipf_iters):
        # Step 1: sample forward
        with torch.no_grad():
            fwd_paths = sample_forward(n_sample, F_net)

        # Step 2: train backward (eq 12)
        for _ in range(n_train_steps):
            k = np.random.randint(0, n_time_steps)
            x_k = fwd_paths[:, k].detach()
            x_k1 = fwd_paths[:, k+1].detach()
            with torch.no_grad():
                f_xk = F_net.full_drift(x_k, k)
                f_xk1 = F_net.full_drift(x_k1, k)
            target = (x_k - x_k1) / dt + f_xk - f_xk1
            pred = B_net.full_drift(x_k1, k+1)
            loss_B = ((pred - target)**2).sum(1).mean()
            B_opt.zero_grad(); loss_B.backward()
            torch.nn.utils.clip_grad_norm_(B_net.parameters(), 5.0)
            B_opt.step()

        # Step 3: sample backward
        with torch.no_grad():
            bwd_paths = sample_backward(n_sample, B_net)

        # Step 4: train forward (eq 13)
        for _ in range(n_train_steps):
            k = np.random.randint(0, n_time_steps)
            x_k = bwd_paths[:, k].detach()
            x_k1 = bwd_paths[:, k+1].detach()
            with torch.no_grad():
                b_xk1 = B_net.full_drift(x_k1, k+1)
                b_xk = B_net.full_drift(x_k, k+1)
            target = (x_k1 - x_k) / dt + b_xk1 - b_xk
            pred = F_net.full_drift(x_k, k)
            loss_F = ((pred - target)**2).sum(1).mean()
            F_opt.zero_grad(); loss_F.backward()
            torch.nn.utils.clip_grad_norm_(F_net.parameters(), 5.0)
            F_opt.step()

        with torch.no_grad():
            tp = sample_forward(200, F_net)
            tm = tp[:, -1].mean(dim=0)
            td = ((tp[:, -1] - B.unsqueeze(0))**2).sum(1).sqrt().mean()
            cr = (tp[:, n_time_steps//2:, 0] > 0).any(dim=1).float().mean()
        print(f"  IPF {ipf_iter+1}/{n_ipf_iters}: loss_B={loss_B.item():.3f}, "
              f"loss_F={loss_F.item():.3f}, terminal=({tm[0]:.2f},{tm[1]:.2f}), "
              f"dist={td:.2f}, crossing={cr:.0%}")

    ipf_time = time.time() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ipf_mem = peak / 1024 / 1024

    with torch.no_grad():
        ipf_paths = sample_forward(n_ref, F_net).cpu()

    terminal_mean = ipf_paths[:, -1].mean(dim=0)
    print(f"  IPF terminal: ({terminal_mean[0]:.3f}, {terminal_mean[1]:.3f})")
    print(f"  IPF time: {ipf_time:.1f}s, memory: {ipf_mem:.1f}MB")
    return ipf_paths, ipf_time, ipf_mem


ipf_paths, ipf_time, ipf_mem = run_sinkhorn_ipf()


# =============================================================================
# Compute all metrics
# =============================================================================
print("\n" + "=" * 70)
print("Computing comprehensive metrics...")
print("=" * 70)

alg1_metrics = compute_all_metrics(alg1_paths, ref_paths, dt, "Alg1")
alg1_spde_metrics = compute_all_metrics(alg1_spde_paths, ref_paths, dt, "Alg1+SPDE")
is_metrics = compute_all_metrics(is_paths, ref_paths, dt, "IS")
ipf_metrics = compute_all_metrics(ipf_paths, ref_paths, dt, "IPF")

# =============================================================================
# Save results
# =============================================================================
results = {
    "setup": {
        "d": d, "T": T, "dt": dt, "scale": SCALE, "harmonic": HARMONIC,
        "potential": f"V(x) = {SCALE}*(x1^2-1)^2 + {HARMONIC}*x2^2",
        "well_A": [-1.0, 0.0], "well_B": [1.0, 0.0],
        "device": str(device)
    },
    "reference": {
        "n_proposals": 50000, "n_resampled": n_ref,
        "ess": round(ess_ref, 0), "sigma_ref": 0.15,
    },
    "algorithm1": {
        "n_annealing": 30, "n_paths": 500, "n_opt": 200,
        "n_spde_refine": 0, "sigma_obs": 0.15,
        "time_s": round(alg1_time, 1), "memory_mb": round(alg1_mem, 1),
        **alg1_metrics,
    },
    "algorithm1_spde": {
        "n_annealing": 30, "n_paths": 500, "n_opt": 200,
        "n_spde_refine": 8, "sigma_obs": 0.15,
        "time_s": round(alg1_spde_time, 1), "memory_mb": round(alg1_spde_mem, 1),
        **alg1_spde_metrics,
    },
    "importance_sampling": {
        "n_proposals": 20000, "n_resampled": 500,
        "ess": round(is_ess, 0),
        "time_s": round(is_time, 2), "memory_mb": round(is_mem, 1),
        **is_metrics,
    },
    "sinkhorn_ipf": {
        "n_ipf_iters": 16, "n_sample": 500, "n_train_steps": 400,
        "time_s": round(ipf_time, 1), "memory_mb": round(ipf_mem, 1),
        **ipf_metrics,
    },
}

os.makedirs("results", exist_ok=True)
with open("results/baseline_2d_dw.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to results/baseline_2d_dw.json")


# =============================================================================
# Print summary
# =============================================================================
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"{'Method':<20} {'Path SW₂':>10} {'Crossing%':>10} {'Term x₁':>10} "
      f"{'OM action':>10} {'Time(s)':>8}")
print("-" * 78)
for name, m, t_sec in [
    ("Alg 1 (no SPDE)", alg1_metrics, alg1_time),
    ("Alg 1 (+SPDE)", alg1_spde_metrics, alg1_spde_time),
    ("IS", is_metrics, is_time),
    ("Sinkhorn/IPF", ipf_metrics, ipf_time),
]:
    print(f"{name:<20} {m['path_sw2']:>10.4f} "
          f"{m['barrier_crossing_frac']:>9.1%} "
          f"{m['terminal']['sample_x1_mean']:>10.3f} "
          f"{m['om_action']['sample_mean']:>10.1f} "
          f"{t_sec:>8.1f}")
print(f"{'Reference':<20} {'—':>10} "
      f"{alg1_metrics['ref_crossing_frac']:>9.1%} "
      f"{alg1_metrics['terminal']['ref_x1_mean']:>10.3f} "
      f"{alg1_metrics['om_action']['ref_mean']:>10.1f} "
      f"{'—':>8}")
print(f"{'='*78}")


# =============================================================================
# Generate figure
# =============================================================================
print("\nGenerating figure...")
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_cols = 5  # ref, alg1, alg1+spde, IS, IPF
    fig = plt.figure(figsize=(25, 14))
    fig.suptitle("2D Double-Well TPS: Baseline Comparison (v2)\n"
                 f"V(x) = {SCALE}(x₁²−1)² + {HARMONIC}x₂², "
                 f"T={T}, A=(-1,0)→B=(1,0)",
                 fontsize=14, fontweight='bold')

    times = np.arange(n_time_steps + 1) * dt

    methods = [
        ("Reference", ref_paths, 'gray', '—'),
        ("Algorithm 1\n(no SPDE)", alg1_paths, '#1f77b4', f'{alg1_time:.0f}s'),
        ("Algorithm 1\n(+SPDE)", alg1_spde_paths, '#9467bd', f'{alg1_spde_time:.0f}s'),
        ("Importance\nSampling", is_paths, '#ff7f0e', f'{is_time:.1f}s'),
        ("Sinkhorn/IPF", ipf_paths, '#2ca02c', f'{ipf_time:.0f}s'),
    ]

    # Row 1: x₁ trajectories
    for col, (label, paths, color, tstr) in enumerate(methods):
        ax = fig.add_subplot(3, n_cols, col + 1)
        n_show = min(50, paths.shape[0])
        for j in range(n_show):
            ax.plot(times, paths[j, :, 0].numpy(), alpha=0.12, color=color, lw=0.5)
        ax.plot(times, paths[:n_show, :, 0].mean(0).numpy(), color=color, lw=2.5)
        ax.axhline(y=-1, color='k', ls='--', alpha=0.2, lw=0.5)
        ax.axhline(y=1, color='k', ls='--', alpha=0.2, lw=0.5)
        ax.axhline(y=0, color='red', ls=':', alpha=0.3, lw=0.5, label='barrier')
        ax.set_title(f"{label}\n({tstr})", fontsize=10)
        ax.set_xlabel("t"); ax.set_ylabel("x₁")
        ax.set_ylim(-2.5, 2.5)

    # Row 2: 2D path plots
    for col, (label, paths, color, _) in enumerate(methods):
        ax = fig.add_subplot(3, n_cols, n_cols + col + 1)
        n_show = min(30, paths.shape[0])
        for j in range(n_show):
            ax.plot(paths[j, :, 0].numpy(), paths[j, :, 1].numpy(),
                   alpha=0.15, color=color, lw=0.5)
        mp = paths[:n_show].mean(0)
        ax.plot(mp[:, 0].numpy(), mp[:, 1].numpy(), color=color, lw=2)
        ax.plot(-1, 0, 'r*', ms=12, zorder=10)
        ax.plot(1, 0, 'r*', ms=12, zorder=10)
        ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
        ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2, 2)
        ax.set_title(f"2D paths: {label.split(chr(10))[0]}", fontsize=9)

    # Row 3, col 1: OM action histograms
    ax = fig.add_subplot(3, n_cols, 2 * n_cols + 1)
    om_ref = onsager_machlup_action(ref_paths, dt).numpy()
    om_a1 = onsager_machlup_action(alg1_paths, dt).numpy()
    om_a1s = onsager_machlup_action(alg1_spde_paths, dt).numpy()
    om_is = onsager_machlup_action(is_paths, dt).numpy()
    om_ipf = onsager_machlup_action(ipf_paths, dt).numpy()
    bins = np.linspace(0, max(np.percentile(om_ref, 95), 50), 40)
    ax.hist(om_ref, bins, alpha=0.35, color='gray', label='Ref', density=True)
    ax.hist(om_a1, bins, alpha=0.35, color='#1f77b4', label='Alg1', density=True)
    ax.hist(om_a1s, bins, alpha=0.35, color='#9467bd', label='Alg1+SPDE', density=True)
    ax.hist(om_is, bins, alpha=0.35, color='#ff7f0e', label='IS', density=True)
    ax.hist(om_ipf, bins, alpha=0.35, color='#2ca02c', label='IPF', density=True)
    ax.set_xlabel("OM action I(x)"); ax.set_ylabel("Density")
    ax.set_title("Onsager-Machlup action", fontsize=10)
    ax.legend(fontsize=7)

    # Row 3, col 2: barrier crossing over time
    ax = fig.add_subplot(3, n_cols, 2 * n_cols + 2)
    for label_short, paths, color in [
        ("Ref", ref_paths, 'gray'),
        ("Alg1", alg1_paths, '#1f77b4'),
        ("Alg1+SPDE", alg1_spde_paths, '#9467bd'),
        ("IS", is_paths, '#ff7f0e'),
        ("IPF", ipf_paths, '#2ca02c'),
    ]:
        x1 = paths[:, :, 0].numpy()
        frac_positive = (x1 > 0).mean(axis=0)
        ax.plot(times, frac_positive, color=color, lw=2, label=label_short)
    ax.axhline(y=0.5, color='k', ls=':', alpha=0.3)
    ax.set_xlabel("t"); ax.set_ylabel("Fraction x₁ > 0")
    ax.set_title("Barrier crossing over time", fontsize=10)
    ax.legend(fontsize=7)

    # Row 3, col 3: path SW₂ bar chart
    ax = fig.add_subplot(3, n_cols, 2 * n_cols + 3)
    method_names = ['Alg 1', 'Alg1\n+SPDE', 'IS', 'IPF']
    path_sw2s = [alg1_metrics['path_sw2'], alg1_spde_metrics['path_sw2'],
                 is_metrics['path_sw2'], ipf_metrics['path_sw2']]
    colors_bar = ['#1f77b4', '#9467bd', '#ff7f0e', '#2ca02c']
    bars = ax.bar(method_names, path_sw2s, color=colors_bar, alpha=0.8)
    ax.set_ylabel("Path-space SW₂")
    ax.set_title("Path-space Sliced Wasserstein", fontsize=10)
    for bar, val in zip(bars, path_sw2s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
               f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    # Row 3, cols 4-5: summary table (span 2 columns)
    ax = fig.add_subplot(3, n_cols, (2 * n_cols + 4, 2 * n_cols + 5))
    ax.axis('off')
    table_data = [
        ['Metric', 'Alg 1', 'Alg1+SPDE', 'IS', 'IPF'],
        ['Path SW₂', f'{alg1_metrics["path_sw2"]:.4f}',
                      f'{alg1_spde_metrics["path_sw2"]:.4f}',
                      f'{is_metrics["path_sw2"]:.4f}',
                      f'{ipf_metrics["path_sw2"]:.4f}'],
        ['Crossing %', f'{alg1_metrics["barrier_crossing_frac"]:.0%}',
                       f'{alg1_spde_metrics["barrier_crossing_frac"]:.0%}',
                       f'{is_metrics["barrier_crossing_frac"]:.0%}',
                       f'{ipf_metrics["barrier_crossing_frac"]:.0%}'],
        ['Term x₁', f'{alg1_metrics["terminal"]["sample_x1_mean"]:.3f}',
                     f'{alg1_spde_metrics["terminal"]["sample_x1_mean"]:.3f}',
                     f'{is_metrics["terminal"]["sample_x1_mean"]:.3f}',
                     f'{ipf_metrics["terminal"]["sample_x1_mean"]:.3f}'],
        ['OM action', f'{alg1_metrics["om_action"]["sample_mean"]:.1f}',
                      f'{alg1_spde_metrics["om_action"]["sample_mean"]:.1f}',
                      f'{is_metrics["om_action"]["sample_mean"]:.1f}',
                      f'{ipf_metrics["om_action"]["sample_mean"]:.1f}'],
        ['Time (s)', f'{alg1_time:.0f}', f'{alg1_spde_time:.0f}',
                     f'{is_time:.1f}', f'{ipf_time:.0f}'],
    ]
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False); table.set_fontsize(9)
    for j in range(5):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')
    ax.set_title("Summary", fontsize=10, pad=15)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig_path = "results/baseline_2d_dw.pdf"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {fig_path}")
except ImportError:
    print("matplotlib not available, skipping figure")

print("\nDone!")
