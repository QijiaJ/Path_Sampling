"""
Main experiment runner.

Experiments:
  - brownian_bridge_metrics — Quantitative metrics (MMD², SW₂, KL) on Brownian bridge
  - ablation               — Ablation study for Algorithm 1 hyperparameters
  - icnn_vs_mlp            — ICNN vs MLP comparison for Algorithm 3
  - scaling_ou            — Higher-dimensional scaling for Algorithm 3 (OU process, d=1..20)
  - scaling_dw            — Higher-dimensional scaling for Algorithm 1 (double-well TPS, d=1..10)
  - h_sweep                — JKO step-size sweep for Algorithm 3 (dense extreme zig-zag)
  - costs                  — Computational cost summary (wall-clock + memory)
  - fi_scaling            — Fisher information estimator scaling with dimension

Usage:
  python run_experiments.py --experiment all                     # Run everything
  python run_experiments.py --experiment brownian_bridge_metrics  
  python run_experiments.py --experiment ablation                 
  python run_experiments.py --experiment icnn_vs_mlp              
  python run_experiments.py --experiment scaling_ou              
  python run_experiments.py --experiment scaling_dw               
  python run_experiments.py --experiment h_sweep                  
  python run_experiments.py --experiment costs                    
  python run_experiments.py --experiment fi_scaling               

"""

import argparse
import json
import os
import time
import tracemalloc
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

from sde_utils import (
    BrownianMotion, OrnsteinUhlenbeck, DoubleWell,
    simulate_brownian_bridge, simulate_sde, soft_constraint_J
)
from metrics import (
    mmd_multi_sigma, sliced_wasserstein_2, marginal_kl_kde,
    compute_all_metrics, fisher_information_estimator,
    fisher_information_true_gaussian
)
# Lazy imports: algorithm1/algorithm3 depend on networks.py which may not
# exist when only running standalone experiments (h_sweep, scaling_double_well).
# Import inside functions that need them instead.
# from algorithm1 import ControlledTransport
# from algorithm3 import WassersteinJKO

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# Experiment 1: Quantitative Metrics on Brownian Bridge
# =============================================================================

def experiment_brownian_bridge_metrics():
    """
    Run Algorithm 1 on the Brownian bridge problem and compute
    quantitative metrics against exact ground truth samples.
    Ground truth: Brownian bridge SDE (32) with x(0)=-2, x(1)=+2.
    """
    from algorithm1 import ControlledTransport
    print("\n" + "="*70)
    print("EXPERIMENT 1: Quantitative Metrics on Brownian Bridge (improved)")
    print("="*70)

    d = 1
    T = 1.0
    dt = 0.01
    n_time_steps = int(T / dt)
    A, B = -2.0, 2.0
    sigma_obs = 0.1
    n_paths = 500
    n_annealing = 20
    n_opt_steps = 400
    n_gt = 1500  # more ground truth samples for stable metrics (3 batches × 500)

    # Ground truth: exact Brownian bridge samples
    print("Generating ground truth Brownian bridge samples...")
    gt_paths = simulate_brownian_bridge(n_gt, d, T, dt, A, B)

    # Algorithm 1: controlled transport
    print("Running Algorithm 1 (Controlled Transport)...")

    def ref_drift(x, t):
        return torch.zeros_like(x)

    # Terminal J_fn with 5-step ramp: weights [0.2, 0.4, 0.6, 0.8, 1.0]
    # applied at the last 5 time steps to avoid an abrupt constraint jump
    ramp_steps = 5
    ramp_weights = [(i + 1) / ramp_steps for i in range(ramp_steps)]
    # Time values for the last 5 steps: t = 0.96, 0.97, 0.98, 0.99, 1.00
    ramp_times = [(n_time_steps - ramp_steps + i) * dt for i in range(ramp_steps)]
    # Also include the actual endpoint t = T = 1.0
    ramp_times_set = set()
    for rt in ramp_times:
        ramp_times_set.add(round(rt, 6))
    ramp_times_set.add(round(T, 6))

    def J_fn(x, t):
        """Terminal constraint with 5-step ramp. Fires at last 5 time steps."""
        t_round = round(t, 6)
        for i, rt in enumerate(ramp_times):
            if abs(t_round - round(rt, 6)) < dt * 0.5:
                w = ramp_weights[i]
                return w * (1.0 / (2 * sigma_obs ** 2)) * ((x - B) ** 2).sum(dim=-1)
        return torch.zeros(x.shape[0], device=x.device)

    # Deterministic initial condition: all paths start at x(0) = A = -2
    def x0_sampler(n, d_dim):
        return torch.full((n, d_dim), A)

    ct = ControlledTransport(d, ref_drift, J_fn, T, dt,
                             hidden_dims=(80, 120), lr=1e-3,
                             x0_sampler=x0_sampler)
    learned_paths, timing = ct.run(n_annealing_steps=n_annealing,
                                    n_paths=n_paths,
                                    n_opt_steps=n_opt_steps,
                                    cosine_schedule=True,
                                    n_final_batches=3)

    # Compute metrics at multiple time points
    time_indices = [0, 25, 50, 75, 100]  # t = 0, 0.25, 0.5, 0.75, 1.0
    results = {'time_points': [], 'mmd2': [], 'sw2': [], 'kl': []}

    for idx in time_indices:
        t_val = idx * dt
        learned_marginal = learned_paths[:, idx, :]
        gt_marginal = gt_paths[:, idx, :]
        metrics = compute_all_metrics(learned_marginal, gt_marginal)
        results['time_points'].append(t_val)
        results['mmd2'].append(metrics['mmd2'])
        results['sw2'].append(metrics['sw2'])
        results['kl'].append(metrics['marginal_kl'])
        print(f"  t={t_val:.2f}: MMD²={metrics['mmd2']:.6f}, "
              f"SW2={metrics['sw2']:.6f}, KL={metrics['marginal_kl']:.4f}")

    results['timing'] = timing

    # Save results
    with open(os.path.join(RESULTS_DIR, "brownian_bridge_metrics.json"), 'w') as f:
        json.dump({k: v if not isinstance(v, dict) else v for k, v in results.items()},
                  f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric, name in zip(axes, ['mmd2', 'sw2', 'kl'],
                                 ['MMD²', 'Sliced W₂', 'Marginal KL']):
        vals = results[metric]
        ax.plot(results['time_points'], vals, 'bo-', linewidth=2, markersize=6)
        ax.set_xlabel('Time t')
        ax.set_ylabel(name)
        ax.set_title(f'{name} vs Ground Truth')
        ax.grid(True, alpha=0.3)
        # Add horizontal reference line at 0
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "brownian_bridge_metrics.pdf"), dpi=150)
    plt.close()

    print(f"\nTotal time: {timing['total']:.1f}s, "
          f"Peak memory: {timing['peak_memory_mb']:.1f}MB")
    return results


# =============================================================================
# Experiment 3: Ablation Study for Algorithm 1
# =============================================================================

def experiment_ablation_algorithm1():
    """
    Ablation study varying:
    - Network size: hidden_dims in {(10,10), (20,30), (40,60), (80,120)}
    - Number of annealing steps: {5, 10, 20, 40}
    - Number of paths: {100, 250, 500, 1000}
    - Optimization steps per iteration: {50, 100, 200, 400}
    """
    from algorithm1 import ControlledTransport
    print("\n" + "="*70)
    print("EXPERIMENT 3: Ablation Study for Algorithm 1")
    print("="*70)

    d = 1
    T = 1.0
    dt = 0.01
    A, B = -2.0, 2.0
    sigma_obs = 0.1
    n_gt = 500

    # Ground truth
    gt_paths = simulate_brownian_bridge(n_gt, d, T, dt, A, B)
    gt_terminal = gt_paths[:, -1, :]

    def ref_drift(x, t):
        return torch.zeros_like(x)

    def J_fn(x, t):
        if abs(t - T + dt) < dt:
            return (1.0 / (2 * sigma_obs ** 2)) * ((x - B) ** 2).sum(dim=-1)
        return torch.zeros(x.shape[0], device=x.device)

    # Default configuration
    default_config = {
        'hidden_dims': (20, 30),
        'n_annealing': 10,
        'n_paths': 500,
        'n_opt_steps': 200
    }

    ablation_results = {}

    # --- Vary network size ---
    print("\n--- Varying network size ---")
    network_sizes = [(10, 10), (20, 30), (40, 60), (80, 120)]
    results_network = []
    for hdims in network_sizes:
        print(f"  Network: {hdims}")
        ct = ControlledTransport(d, ref_drift, J_fn, T, dt,
                                 hidden_dims=hdims, lr=1e-3)
        paths, timing = ct.run(n_annealing_steps=default_config['n_annealing'],
                               n_paths=default_config['n_paths'],
                               n_opt_steps=default_config['n_opt_steps'])
        terminal = paths[:, -1, :]
        metrics = compute_all_metrics(terminal, gt_terminal)
        metrics['timing'] = timing['total']
        metrics['memory_mb'] = timing['peak_memory_mb']
        metrics['config'] = str(hdims)
        results_network.append(metrics)
        print(f"    MMD²={metrics['mmd2']:.6f}, SW2={metrics['sw2']:.6f}, "
              f"Time={timing['total']:.1f}s")
    ablation_results['network_size'] = results_network

    # --- Vary annealing steps ---
    print("\n--- Varying annealing steps ---")
    annealing_counts = [5, 10, 20, 40]
    results_anneal = []
    for n_ann in annealing_counts:
        print(f"  Annealing steps: {n_ann}")
        ct = ControlledTransport(d, ref_drift, J_fn, T, dt,
                                 hidden_dims=default_config['hidden_dims'], lr=1e-3)
        paths, timing = ct.run(n_annealing_steps=n_ann,
                               n_paths=default_config['n_paths'],
                               n_opt_steps=default_config['n_opt_steps'])
        terminal = paths[:, -1, :]
        metrics = compute_all_metrics(terminal, gt_terminal)
        metrics['timing'] = timing['total']
        metrics['memory_mb'] = timing['peak_memory_mb']
        metrics['config'] = n_ann
        results_anneal.append(metrics)
        print(f"    MMD²={metrics['mmd2']:.6f}, SW2={metrics['sw2']:.6f}, "
              f"Time={timing['total']:.1f}s")
    ablation_results['annealing_steps'] = results_anneal

    # --- Vary number of paths ---
    print("\n--- Varying ensemble size (paths) ---")
    path_counts = [100, 250, 500, 1000]
    results_paths = []
    for n_p in path_counts:
        print(f"  Paths: {n_p}")
        ct = ControlledTransport(d, ref_drift, J_fn, T, dt,
                                 hidden_dims=default_config['hidden_dims'], lr=1e-3)
        paths, timing = ct.run(n_annealing_steps=default_config['n_annealing'],
                               n_paths=n_p,
                               n_opt_steps=default_config['n_opt_steps'])
        terminal = paths[:, -1, :]
        metrics = compute_all_metrics(terminal, gt_terminal)
        metrics['timing'] = timing['total']
        metrics['memory_mb'] = timing['peak_memory_mb']
        metrics['config'] = n_p
        results_paths.append(metrics)
        print(f"    MMD²={metrics['mmd2']:.6f}, SW2={metrics['sw2']:.6f}, "
              f"Time={timing['total']:.1f}s")
    ablation_results['ensemble_size'] = results_paths

    # --- Vary optimization steps ---
    print("\n--- Varying optimization steps ---")
    opt_counts = [50, 100, 200, 400]
    results_opt = []
    for n_opt in opt_counts:
        print(f"  Opt steps: {n_opt}")
        ct = ControlledTransport(d, ref_drift, J_fn, T, dt,
                                 hidden_dims=default_config['hidden_dims'], lr=1e-3)
        paths, timing = ct.run(n_annealing_steps=default_config['n_annealing'],
                               n_paths=default_config['n_paths'],
                               n_opt_steps=n_opt)
        terminal = paths[:, -1, :]
        metrics = compute_all_metrics(terminal, gt_terminal)
        metrics['timing'] = timing['total']
        metrics['memory_mb'] = timing['peak_memory_mb']
        metrics['config'] = n_opt
        results_opt.append(metrics)
        print(f"    MMD²={metrics['mmd2']:.6f}, SW2={metrics['sw2']:.6f}, "
              f"Time={timing['total']:.1f}s")
    ablation_results['opt_steps'] = results_opt

    # Save
    with open(os.path.join(RESULTS_DIR, "ablation_algorithm1.json"), 'w') as f:
        json.dump(ablation_results, f, indent=2, default=str)

    # Plot ablation results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ablation_axes = [
        ('network_size', 'Network Size', [str(s) for s in network_sizes]),
        ('annealing_steps', 'Annealing Steps', [str(s) for s in annealing_counts]),
        ('ensemble_size', 'Ensemble Size (K)', [str(s) for s in path_counts]),
        ('opt_steps', 'Optimization Steps', [str(s) for s in opt_counts]),
    ]
    for ax, (key, xlabel, labels) in zip(axes.flat, ablation_axes):
        data = ablation_results[key]
        mmd_vals = [d['mmd2'] for d in data]
        sw2_vals = [d['sw2'] for d in data]
        x_pos = range(len(labels))
        ax.bar([p - 0.15 for p in x_pos], mmd_vals, 0.3, label='MMD²', color='steelblue')
        ax.bar([p + 0.15 for p in x_pos], sw2_vals, 0.3, label='SW₂', color='coral')
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(labels, rotation=45)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Metric Value')
        ax.legend()
        ax.set_title(f'Ablation: {xlabel}')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "ablation_algorithm1.pdf"), dpi=150)
    plt.close()

    return ablation_results


# =============================================================================
# Experiment 4: ICNN vs MLP for Algorithm 3
# =============================================================================

def experiment_icnn_vs_mlp():
    """
    Compare ICNN and standard MLP as pushforward map in Algorithm 3
    on the OU process example from Section G.2.
    """
    from algorithm3 import WassersteinJKO
    print("\n" + "="*70)
    print("EXPERIMENT 4: ICNN vs MLP for Algorithm 3")
    print("="*70)

    d = 1
    T = 1.0
    h = 0.2
    beta = 0.25
    sigma_fi = 0.4
    n_particles = 200
    n_opt_steps = 200
    x_fixed_mid = -1.0
    x_fixed_end = 1.0
    gamma_big = 0.5
    gamma_small = 0.1

    ou = OrnsteinUhlenbeck(d=d, beta=beta)

    def grad_V(x):
        return ou.grad_potential(x)

    def J_fn(x, t):
        """Soft constraint at midpoint and endpoint."""
        J = torch.zeros(x.shape[0], device=x.device)
        t_mid = T / 2
        if abs(t - t_mid) < h / 2:
            J += (1.0 / (2 * gamma_big ** 2)) * ((x[:, 0] - x_fixed_mid) ** 2)
        if abs(t - T) < h / 2:
            J += (1.0 / (2 * gamma_small ** 2)) * ((x[:, 0] - x_fixed_end) ** 2)
        return J

    results = {}

    for use_icnn, label in [(True, "ICNN"), (False, "MLP")]:
        print(f"\n--- Running Algorithm 3 with {label} ---")
        x0 = ou.sample_stationary(n_particles)

        jko = WassersteinJKO(
            d=d, grad_V_fn=grad_V, J_fn_at_t=J_fn, T=T, h=h,
            sigma_fi=sigma_fi, m_perturbations=30,
            use_icnn=use_icnn,
            hidden_dims=(64, 64, 64, 64),
            lr=1e-3
        )
        particles_history, timing = jko.run(x0, n_opt_steps=n_opt_steps)

        # Get terminal particles
        terminal = particles_history[T]

        # Compute metrics against OU stationary conditioned on constraints
        # (approximate ground truth via long-run MCMC is expensive;
        #  here we report the distribution statistics)
        results[label] = {
            'terminal_mean': terminal.mean(dim=0).tolist(),
            'terminal_std': terminal.std(dim=0).tolist(),
            'timing_total': timing['total'],
            'timing_per_step': timing['per_step'],
            'peak_memory_mb': timing['peak_memory_mb'],
        }
        print(f"  {label}: mean={terminal.mean().item():.3f}, "
              f"std={terminal.std().item():.3f}, "
              f"time={timing['total']:.1f}s, mem={timing['peak_memory_mb']:.1f}MB")

    # Cross-compare: compute MMD between ICNN and MLP outputs
    # (They should ideally agree if both are correct)
    with open(os.path.join(RESULTS_DIR, "icnn_vs_mlp.json"), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, ylabel in zip(axes, ['timing_total', 'peak_memory_mb'],
                                   ['Wall-clock Time (s)', 'Peak Memory (MB)']):
        vals = [results[label][metric] for label in ['ICNN', 'MLP']]
        ax.bar(['ICNN', 'MLP'], vals, color=['steelblue', 'coral'])
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel}: ICNN vs MLP')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "icnn_vs_mlp.pdf"), dpi=150)
    plt.close()

    return results


# =============================================================================
# Experiment 5: Higher-dimensional Scaling
# =============================================================================

def experiment_scaling_ou(dimensions=None):
    """
    Scale Algorithm 3 (interacting particle system) to higher dimensions
    using the OU process with soft constraints.
    """
    from algorithm3 import WassersteinJKO
    print("\n" + "="*70)
    print("EXPERIMENT 5a: Higher-dimensional Scaling (OU process)")
    print("="*70)

    if dimensions is None:
        dimensions = [1, 2, 5, 10, 20, 50]

    T = 1.0
    h = 0.2
    beta = 0.25
    sigma_fi = 0.4
    n_particles = 200
    n_opt_steps = 150
    gamma = 0.5

    results = {}

    for d in dimensions:
        print(f"\n--- d = {d} ---")
        ou = OrnsteinUhlenbeck(d=d, beta=beta)

        def grad_V(x, _ou=ou):
            return _ou.grad_potential(x)

        # Constraint: pin first coordinate to +1 at endpoint
        def J_fn(x, t, _T=T, _h=h):
            if abs(t - _T) < _h / 2:
                return (1.0 / (2 * gamma ** 2)) * ((x[:, 0] - 1.0) ** 2)
            return torch.zeros(x.shape[0], device=x.device)

        x0 = ou.sample_stationary(n_particles)

        # Use interacting particle system (MLP) for high-d
        use_icnn = d <= 10  # ICNN works better in low-d

        jko = WassersteinJKO(
            d=d, grad_V_fn=grad_V, J_fn_at_t=J_fn, T=T, h=h,
            sigma_fi=sigma_fi, m_perturbations=max(30, d),
            use_icnn=use_icnn,
            hidden_dims=(64, 64, 64, 64),
            lr=1e-3
        )

        try:
            particles_history, timing = jko.run(x0, n_opt_steps=n_opt_steps)
            terminal = particles_history[T]

            # For OU, the true FI is tr(Sigma^{-1}) = d * beta
            fi_true = d * beta
            fi_est = fisher_information_estimator(
                terminal, sigma_fi, m_perturbations=50
            ).item()
            fi_rel_error = abs(fi_est - fi_true) / fi_true

            results[d] = {
                'terminal_mean_x1': terminal[:, 0].mean().item(),
                'terminal_std_x1': terminal[:, 0].std().item(),
                'fi_true': fi_true,
                'fi_estimated': fi_est,
                'fi_relative_error': fi_rel_error,
                'timing_total': timing['total'],
                'peak_memory_mb': timing['peak_memory_mb'],
                'use_icnn': use_icnn,
            }
            print(f"  d={d}: x1_mean={terminal[:, 0].mean().item():.3f}, "
                  f"FI_err={fi_rel_error:.3f}, "
                  f"time={timing['total']:.1f}s, mem={timing['peak_memory_mb']:.1f}MB")
        except Exception as e:
            print(f"  d={d}: FAILED - {e}")
            results[d] = {'error': str(e)}

    with open(os.path.join(RESULTS_DIR, "scaling_ou.json"), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Plot
    valid_dims = [d for d in dimensions if d in results and 'error' not in results[d]]
    if valid_dims:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # FI relative error vs d
        axes[0].plot(valid_dims,
                     [results[d]['fi_relative_error'] for d in valid_dims],
                     'bo-', linewidth=2)
        axes[0].set_xlabel('Dimension d')
        axes[0].set_ylabel('Relative FI Error')
        axes[0].set_title('Fisher Information Scaling')
        axes[0].grid(True, alpha=0.3)

        # Time vs d
        axes[1].plot(valid_dims,
                     [results[d]['timing_total'] for d in valid_dims],
                     'ro-', linewidth=2)
        axes[1].set_xlabel('Dimension d')
        axes[1].set_ylabel('Wall-clock Time (s)')
        axes[1].set_title('Computational Time Scaling')
        axes[1].grid(True, alpha=0.3)

        # Memory vs d
        axes[2].plot(valid_dims,
                     [results[d]['peak_memory_mb'] for d in valid_dims],
                     'go-', linewidth=2)
        axes[2].set_xlabel('Dimension d')
        axes[2].set_ylabel('Peak Memory (MB)')
        axes[2].set_title('Memory Scaling')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "scaling_ou.pdf"), dpi=150)
        plt.close()

    return results


def experiment_scaling_double_well(dimensions=None):
    """
    Scale Algorithm 1 to higher dimensions for double-well TPS.
    V(x) = 5(x_1^2 - 1)^2, transition from x_1=-1 to x_1=+1.

    Uses the stabilized configuration from run_dw_d12.py:
      - ResBlock DriftNet with LayerNorm + output clamping
      - Cosine annealing schedule
      - 5-step terminal J ramp
      - 30 annealing steps, 200 opt steps
      - Per-dimension network width and sigma_obs
    """
    print("\n" + "="*70)
    print("EXPERIMENT 5b: Higher-dimensional Scaling (Double-well TPS)")
    print("="*70)

    if dimensions is None:
        dimensions = [1, 2, 5, 10]

    # Import the stabilized implementation from run_dw_d12
    from run_dw_d12 import (
        run_experiment as dw_run_experiment,
        plot_results as dw_plot_results,
        get_device,
    )
    device = get_device()

    # Per-dimension configs matching run_dw_d12.py
    configs = {
        1: dict(
            n_annealing=30, n_opt=200, n_paths=200,
            width=128, depth=4, lr=2e-3,
            sigma_obs=0.15, dt=0.02,
        ),
        2: dict(
            n_annealing=30, n_opt=200, n_paths=200,
            width=192, depth=4, lr=2e-3,
            sigma_obs=0.15, dt=0.02,
        ),
        5: dict(
            n_annealing=30, n_opt=200, n_paths=300,
            width=192, depth=4, lr=1.5e-3,
            sigma_obs=0.2, dt=0.02,
        ),
        10: dict(
            n_annealing=30, n_opt=200, n_paths=400,
            width=256, depth=4, lr=1e-3,
            sigma_obs=0.25, dt=0.02,
        ),
    }

    results = {}
    all_paths = []

    for d in dimensions:
        if d not in configs:
            print(f"\n--- d = {d}: no config defined, skipping ---")
            continue

        print(f"\n--- d = {d} ---")
        config = configs[d]

        try:
            result, paths = dw_run_experiment(d, config, device)
            results[d] = result
            all_paths.append(paths)

            print(f"  d={d}: x1_start={result['initial_x1_mean']:.3f}, "
                  f"x1_end={result['terminal_x1_mean']:.3f}, "
                  f"crossing={result['n_crossing']}/{result['n_total']}, "
                  f"time={result['timing_total']:.1f}s")
        except Exception as e:
            print(f"  d={d}: FAILED - {e}")
            results[d] = {'error': str(e)}

    # Save results
    for d, result in results.items():
        if 'error' not in result:
            fname = f"dw_d{d}_stable.json"
            with open(os.path.join(RESULTS_DIR, fname), 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"  Saved {fname}")

    with open(os.path.join(RESULTS_DIR, "scaling_double_well.json"), 'w') as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2, default=str)

    # Plot
    valid_dims = [d for d in dimensions if d in results and 'error' not in results[d]]
    if valid_dims and all_paths:
        results_list = [results[d] for d in valid_dims]
        try:
            dw_plot_results(results_list, all_paths, RESULTS_DIR)
        except Exception as e:
            print(f"  Plot failed: {e}")

        # Also make a simple scaling summary plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Crossing rate
        crossing_rates = [results[d]['n_crossing'] / results[d]['n_total'] for d in valid_dims]
        axes[0].bar([str(d) for d in valid_dims], crossing_rates, color='steelblue', alpha=0.8)
        axes[0].set_xlabel('Dimension d')
        axes[0].set_ylabel('Crossing Rate')
        axes[0].set_title('Barrier Crossing Rate')
        axes[0].set_ylim(0, 1.1)
        axes[0].axhline(y=1.0, color='r', ls='--', alpha=0.3)
        axes[0].grid(True, alpha=0.3)

        # Terminal x₁ mean
        axes[1].plot(valid_dims,
                     [results[d]['terminal_x1_mean'] for d in valid_dims],
                     'bo-', linewidth=2, label='Terminal x₁ mean')
        axes[1].axhline(y=1.0, color='r', linestyle='--', label='Target (x₁=1)')
        axes[1].set_xlabel('Dimension d')
        axes[1].set_ylabel('Terminal x₁')
        axes[1].legend()
        axes[1].set_title('Terminal Constraint Satisfaction')
        axes[1].grid(True, alpha=0.3)

        # Time
        axes[2].plot(valid_dims,
                     [results[d]['timing_total'] for d in valid_dims],
                     'ro-', linewidth=2)
        axes[2].set_xlabel('Dimension d')
        axes[2].set_ylabel('Wall-clock Time (s)')
        axes[2].set_title('Computational Time vs Dimension')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "scaling_double_well.pdf"), dpi=150)
        plt.close()

    return results


# =============================================================================
# Experiment 6: Step-size h Sweep for Algorithm 3
# =============================================================================

def experiment_h_sweep():
    """
    h-sweep for Algorithm 3 under a setup that is *hard* for the JKO
    decomposition: dense observations with extreme zig-zag targets
    that force rapid marginal reversals.

    Setup: OU process (β=0.25, d=1), equilibrium init N(0, 1/β) = N(0, 4).
    5 observations at t=0.2, 0.4, 0.6, 0.8, 1.0 with zig-zag targets
    [5, -4, 6, -3, 7] and σ_obs = 0.2.  These targets are 2–3.5σ from
    equilibrium and alternate sign, creating the most stressful scenario
    for Algorithm 3: at large h, a single JKO step must bridge two or
    more observation times with opposite-sign targets, violating the
    smooth-marginal-evolution assumption.

    Reference: analytical Gaussian posterior via Kalman smoother.
    Sweep h ∈ {T/20, T/10, T/5, T/4, T/2, T}.

    At small h the JKO grid is fine enough that each interval contains
    at most one observation → error is small.  At large h the grid is
    too coarse → error grows, demonstrating the O(h²) degradation
    predicted by Section 4.3.3.
    """
    import gc, math

    print("\n" + "="*70)
    print("EXPERIMENT A: h-Sweep for Algorithm 3 (dense zig-zag)")
    print("="*70)

    # Import OU utilities and Algorithm 3 from run_alg_comparison
    from run_alg_comparison import (
        Likelihood, Algorithm3,
        ou_conditional_moments, sample_ou_conditional,
        compute_metrics, sliced_wasserstein_1d,
        BETA, T, DT, N_STEPS, EQ_STD, DEVICE,
    )

    obs_times = [0.2, 0.4, 0.6, 0.8, 1.0]
    obs_targets = [5.0, -4.0, 6.0, -3.0, 7.0]
    sigma_obs = 0.2

    lik = Likelihood(obs_times, obs_targets, sigma_obs)
    print(f"  Obs times:  {obs_times}")
    print(f"  Targets:    {obs_targets}")
    print(f"  σ_obs = {sigma_obs}")
    print(f"  Penalty strength: 1/(2σ²) = {0.5/sigma_obs**2:.1f}")

    # Analytical reference via Kalman smoother
    print("  Computing analytical Gaussian posterior...")
    ref_paths, eval_times, cond_mean, cond_var = sample_ou_conditional(
        500, obs_times, obs_targets, sigma_obs
    )
    print(f"  Reference: {ref_paths.shape[0]} analytical samples")
    for t_obs, target in zip(obs_times, obs_targets):
        t_idx = int(round(t_obs / DT))
        print(f"    t={t_obs:.1f}: cond_mean={cond_mean[t_idx]:.3f}, "
              f"cond_std={math.sqrt(max(cond_var[t_idx], 0)):.3f} (target={target})")

    h_values = [T / 20, T / 10, T / 5, T / 4, T / 2, T]
    results = {
        "obs_times": obs_times, "obs_targets": obs_targets,
        "sigma_obs": sigma_obs,
        "analytical_cond_mean": {
            f"t={t:.2f}": float(cond_mean[int(round(t/DT))])
            for t in obs_times
        },
        "analytical_cond_std": {
            f"t={t:.2f}": float(math.sqrt(max(cond_var[int(round(t/DT))], 0)))
            for t in obs_times
        },
        "h_sweep": {},
    }

    for h in h_values:
        n_jko = int(round(T / h))
        h_key = f"h={h:.4f}"
        print(f"\n--- h = {h:.4f} ({n_jko} JKO steps) ---")

        alg3 = Algorithm3(lik, h=h, width=256, depth=4, lr=1e-3)
        paths, elapsed = alg3.run(n_particles=300, n_opt=400)
        metrics = compute_metrics(paths, ref_paths, lik, cond_mean, cond_var, eval_times)
        metrics["time_s"] = elapsed
        metrics["n_jko_steps"] = n_jko
        results["h_sweep"][h_key] = metrics

        print(f"  avg_sw2={metrics['avg_sw2']:.5f}, "
              f"path_sw2={metrics['path_sw2']:.5f}, "
              f"traj_rmse={metrics['traj_rmse']:.5f}, "
              f"time={elapsed:.1f}s")

        del alg3, paths
        gc.collect()

    # Save
    with open(os.path.join(RESULTS_DIR, "hsweep_dense_abrupt.json"), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary table
    print("\n  h-sweep summary:")
    print(f"  {'h':>8} {'T/h':>5} {'Avg SW₂':>10} {'Traj RMSE':>10} {'Time(s)':>8}")
    print("  " + "-" * 50)
    for h_key, m in sorted(results["h_sweep"].items()):
        print(f"  {h_key:>8} {m['n_jko_steps']:>5} "
              f"{m['avg_sw2']:>10.4f} {m['traj_rmse']:>10.4f} "
              f"{m['time_s']:>8.1f}")

    # Plot: 3 panels
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        h_vals_plot = []
        avg_sw2_plot = []
        path_sw2_plot = []
        traj_rmse_plot = []
        times_plot = []
        per_obs_data = {}

        for h_key, m in sorted(results["h_sweep"].items()):
            h = float(h_key.split("=")[1])
            h_vals_plot.append(h)
            avg_sw2_plot.append(m["avg_sw2"])
            path_sw2_plot.append(m.get("path_sw2", m["avg_sw2"]))
            traj_rmse_plot.append(m["traj_rmse"])
            times_plot.append(m["time_s"])
            for j, t_obs in enumerate(obs_times):
                key = f"t={t_obs:.2f}"
                if key not in per_obs_data:
                    per_obs_data[key] = []
                per_obs_data[key].append(m[key]["sw2"])

        # Left: SW₂ + RMSE vs h (log-log)
        ax = axes[0]
        ax.loglog(h_vals_plot, avg_sw2_plot, 'bo-', lw=2, ms=8, label='Avg marginal SW₂')
        ax.loglog(h_vals_plot, path_sw2_plot, 'mv--', lw=2, ms=8, label='Path-space SW₂')
        ax.loglog(h_vals_plot, traj_rmse_plot, 'rs--', lw=2, ms=8, label='Traj RMSE vs analytical')
        h_arr = np.array(h_vals_plot)
        if avg_sw2_plot[0] > 0 and h_vals_plot[0] > 0:
            c = avg_sw2_plot[0] / h_vals_plot[0]**2
            ax.loglog(h_arr, c * h_arr**2, 'k:', alpha=0.5, lw=1, label='O(h²) reference')
        ax.set_xlabel('JKO step size h')
        ax.set_ylabel('Error')
        ax.set_title('Algorithm 3: Error vs h\n(dense extreme zig-zag)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Middle: per-observation SW₂ breakdown
        ax = axes[1]
        colors = plt.cm.tab10(np.linspace(0, 0.5, len(per_obs_data)))
        for (key, sw2_list), color in zip(per_obs_data.items(), colors):
            idx = obs_times.index(float(key.split("=")[1]))
            target = obs_targets[idx]
            ax.loglog(h_vals_plot, sw2_list, 'o-', color=color, lw=1.5,
                      ms=6, label=f'{key} (y={target:.0f})')
        ax.set_xlabel('JKO step size h')
        ax.set_ylabel('Marginal SW₂')
        ax.set_title('Per-observation SW₂ breakdown')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Right: wall-clock time vs h
        ax = axes[2]
        ax.semilogx(h_vals_plot, times_plot, 'go-', lw=2, ms=8)
        ax.set_xlabel('JKO step size h')
        ax.set_ylabel('Wall-clock time (s)')
        ax.set_title('Computation time vs h')
        ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"OU β={BETA}, d=1 | Obs: t=[0.2..1.0], "
            f"targets=[{','.join(str(int(y)) for y in obs_targets)}], σ={sigma_obs}\n"
            f"Reference: analytical Gaussian posterior (Kalman smoother)",
            fontsize=11, y=1.04
        )
        plt.tight_layout()
        fig_path = os.path.join(RESULTS_DIR, "hsweep_dense_abrupt.pdf")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nFigure saved to {fig_path}")
    except Exception as e:
        print(f"  Plot failed: {e}")

    return results


# =============================================================================
# Experiment 8: Comprehensive Computational Cost Table
# =============================================================================

def experiment_computational_costs():
    """
    Generate a comprehensive table of computational costs for both algorithms
    across different problem sizes and dimensions.

    Includes:
    - Alg 1 on Brownian bridge (simple, original algorithm1.py)
    - Alg 1 on Double-well TPS (stabilized, from run_dw_d12.py results)
    - Alg 3 (ICNN/MLP) on OU process
    """
    from algorithm1 import ControlledTransport
    from algorithm3 import WassersteinJKO
    print("\n" + "="*70)
    print("EXPERIMENT 8: Computational Cost Summary")
    print("="*70)

    cost_table = []

    # Brownian bridge (1D) with Algorithm 1 (original simple implementation)
    for n_paths in [100, 500, 1000]:
        d = 1
        T = 1.0
        dt = 0.01

        def ref_drift(x, t):
            return torch.zeros_like(x)

        def J_fn(x, t):
            if abs(t - T + dt) < dt:
                return (1.0 / (2 * 0.1 ** 2)) * ((x - 2.0) ** 2).sum(dim=-1)
            return torch.zeros(x.shape[0])

        start = time.time()
        tracemalloc.start()
        ct = ControlledTransport(d, ref_drift, J_fn, T, dt, hidden_dims=(20, 30))
        paths, timing = ct.run(n_annealing_steps=10, n_paths=n_paths, n_opt_steps=100)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        cost_table.append({
            'algorithm': 'Alg 1 (Controlled Transport)',
            'problem': f'BB d={d}',
            'n_samples': n_paths,
            'wall_time_s': timing['total'],
            'peak_memory_mb': peak / 1024 / 1024,
            'time_per_step_s': np.mean(timing['per_step']),
        })

    # Double-well TPS with stabilized Alg 1 (read from saved results)
    for d_dw in [1, 2, 5, 10]:
        fname = os.path.join(RESULTS_DIR, f"dw_d{d_dw}_stable.json")
        if os.path.exists(fname):
            with open(fname) as f:
                dw_res = json.load(f)
            cost_table.append({
                'algorithm': 'Alg 1 (Stabilized DW)',
                'problem': f'DW d={d_dw}',
                'n_samples': dw_res.get('n_total', dw_res.get('config', {}).get('n_paths', '?')),
                'wall_time_s': dw_res.get('timing_total', -1),
                'peak_memory_mb': -1,  # not tracked in run_dw_d12
                'time_per_step_s': dw_res.get('timing_total', 0) / 30,  # approx: 30 annealing steps
                'n_params': dw_res.get('n_params', '?'),
                'crossing_rate': f"{dw_res.get('frac_crossing', 0):.0%}",
            })
            print(f"  Loaded DW d={d_dw} results from {fname}")
        else:
            print(f"  DW d={d_dw} results not found at {fname}")

    # OU process with Algorithm 3 (ICNN and MLP)
    for use_icnn, label in [(True, "ICNN"), (False, "MLP")]:
        for d in [1, 5, 10]:
            T = 1.0
            h = 0.2
            beta = 0.25
            ou = OrnsteinUhlenbeck(d=d, beta=beta)

            def grad_V(x, _ou=ou):
                return _ou.grad_potential(x)

            def J_fn(x, t, _h=h):
                if abs(t - T) < _h / 2:
                    return (1.0 / (2 * 0.5 ** 2)) * ((x[:, 0] - 1.0) ** 2)
                return torch.zeros(x.shape[0])

            x0 = ou.sample_stationary(200)

            start = time.time()
            tracemalloc.start()
            jko = WassersteinJKO(d=d, grad_V_fn=grad_V, J_fn_at_t=J_fn,
                                 T=T, h=h, sigma_fi=0.4,
                                 use_icnn=use_icnn,
                                 hidden_dims=(64, 64, 64, 64), lr=1e-3)
            try:
                particles, timing = jko.run(x0, n_opt_steps=100)
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                cost_table.append({
                    'algorithm': f'Alg 3 ({label})',
                    'problem': f'OU d={d}',
                    'n_samples': 200,
                    'wall_time_s': timing['total'],
                    'peak_memory_mb': peak / 1024 / 1024,
                    'time_per_step_s': np.mean(timing['per_step']),
                })
            except Exception as e:
                tracemalloc.stop()
                cost_table.append({
                    'algorithm': f'Alg 3 ({label})',
                    'problem': f'OU d={d}',
                    'n_samples': 200,
                    'wall_time_s': -1,
                    'peak_memory_mb': -1,
                    'time_per_step_s': -1,
                    'error': str(e),
                })

    # Print table
    print("\n" + "-"*100)
    print(f"{'Algorithm':<30} {'Problem':<12} {'N':<6} "
          f"{'Time(s)':<10} {'Mem(MB)':<10} {'Time/step(s)':<12} {'Extra':<20}")
    print("-"*100)
    for row in cost_table:
        extra = ""
        if 'crossing_rate' in row:
            extra = f"cross={row['crossing_rate']}"
        if 'n_params' in row:
            extra += f" params={row['n_params']}"
        wt = row['wall_time_s']
        mem = row['peak_memory_mb']
        tps = row['time_per_step_s']
        print(f"{row['algorithm']:<30} {row['problem']:<12} {str(row['n_samples']):<6} "
              f"{wt:<10.1f} {mem:<10.1f} {tps:<12.2f} {extra}")
    print("-"*100)

    with open(os.path.join(RESULTS_DIR, "computational_costs.json"), 'w') as f:
        json.dump(cost_table, f, indent=2, default=str)

    return cost_table


# =============================================================================
# Fisher Information Scaling (improved Table 1 with std)
# =============================================================================

def experiment_fi_scaling():
    """
    Improved version of Table 1: Fisher information scaling with dimension.
    Reports mean ± std over 20 trials.
    Also varies n and m to show the error can be reduced.
    """
    print("\n" + "="*70)
    print("EXPERIMENT: Improved Fisher Information Scaling (Table 1)")
    print("="*70)

    dimensions = list(range(1, 11))
    n_trials = 20
    sigma_bw = 0.4
    m_pert = 50

    results = {}

    for d in dimensions:
        # Sigma = diag(1, ..., d) => Sigma_inv = diag(1, 1/2, ..., 1/d)
        Sigma_diag = torch.arange(1, d + 1, dtype=torch.float32)
        Sigma_inv_diag = 1.0 / Sigma_diag
        fi_true = Sigma_inv_diag.sum().item()

        errors = []
        for trial in range(n_trials):
            # Sample from N(0, Sigma)
            n_samples = 500 * d  # Scale linearly with d
            samples = torch.randn(n_samples, d) * Sigma_diag.sqrt().unsqueeze(0)
            fi_est = fisher_information_estimator(samples, sigma_bw, m_pert).item()
            rel_error = abs(fi_est - fi_true) / fi_true
            errors.append(rel_error)

        results[d] = {
            'fi_true': fi_true,
            'mean_rel_error': np.mean(errors),
            'std_rel_error': np.std(errors),
            'n_samples': 500 * d,
        }
        print(f"  d={d}: FI_true={fi_true:.2f}, "
              f"rel_error={np.mean(errors):.4f} ± {np.std(errors):.4f}")

    with open(os.path.join(RESULTS_DIR, "fi_scaling_improved.json"), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Print as table
    print("\n  Table: Fisher Information Scaling (n=500*d, m=50, 20 trials)")
    print(f"  {'d':<5} {'FI_true':<10} {'Rel Error (mean±std)':<25} {'n_samples':<10}")
    for d in dimensions:
        r = results[d]
        print(f"  {d:<5} {r['fi_true']:<10.2f} "
              f"{r['mean_rel_error']:.4f} ± {r['std_rel_error']:.4f}       "
              f"{r['n_samples']:<10}")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run experiments for path measure sampling paper"
    )
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', 'brownian_bridge_metrics', 'ablation',
                                 'icnn_vs_mlp', 'scaling_ou', 'scaling_dw',
                                 'h_sweep', 'costs', 'fi_scaling'],
                        help='Which experiment to run')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'])
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"PyTorch version: {torch.__version__}")

    all_results = {}

    if args.experiment in ['all', 'brownian_bridge_metrics']:
        all_results['brownian_bridge'] = experiment_brownian_bridge_metrics()

    if args.experiment in ['all', 'ablation']:
        all_results['ablation'] = experiment_ablation_algorithm1()

    if args.experiment in ['all', 'icnn_vs_mlp']:
        all_results['icnn_vs_mlp'] = experiment_icnn_vs_mlp()

    if args.experiment in ['all', 'scaling_ou']:
        all_results['scaling_ou'] = experiment_scaling_ou()

    if args.experiment in ['all', 'scaling_dw']:
        all_results['scaling_dw'] = experiment_scaling_double_well()

    if args.experiment in ['all', 'h_sweep']:
        all_results['h_sweep'] = experiment_h_sweep()

    if args.experiment in ['all', 'costs']:
        all_results['costs'] = experiment_computational_costs()

    if args.experiment in ['all', 'fi_scaling']:
        all_results['fi_scaling'] = experiment_fi_scaling()

    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results saved to: {os.path.abspath(RESULTS_DIR)}/")
    print("="*70)


if __name__ == "__main__":
    main()
