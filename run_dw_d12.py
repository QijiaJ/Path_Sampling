#!/usr/bin/env python3
"""
Double-well TPS for d=1,2,5,10 

Run:
  python run_dw_d12.py          # runs d=1,2,5,10
  python run_dw_d12.py 1        # d=1 only
  python run_dw_d12.py 5 10     # d=5 and d=10
"""

import json, os, sys, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---- Device selection ----
def get_device():
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("Using Apple MPS GPU")
        return torch.device('mps')
    elif torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device('cuda')
    else:
        print("WARNING: No GPU found, falling back to CPU")
        return torch.device('cpu')


# ---- Residual drift network (moderate size) ----
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        h = self.norm(x)
        h = F.silu(self.fc1(h))
        h = self.fc2(h)
        return x + h


class DriftNet(nn.Module):
    """
    Residual drift network with moderate size.
    d=1: width=128, depth=4 -> ~100K params (vs original 900 params)
    d=2: width=192, depth=4 -> ~225K params
    Output is scaled by a learnable factor initialized near 0.
    """
    def __init__(self, d, width=128, depth=4):
        super().__init__()
        self.d = d
        self.input_proj = nn.Linear(d + 2, width)
        self.blocks = nn.ModuleList([ResBlock(width) for _ in range(depth)])
        self.output_norm = nn.LayerNorm(width)
        self.output_proj = nn.Linear(width, d)
        # Zero-init output layer so the network starts as identity (zero drift update)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x_t, t, s):
        B = x_t.shape[0]
        device, dtype = x_t.device, x_t.dtype
        if isinstance(t, (int, float)):
            t = torch.full((B, 1), t, device=device, dtype=dtype)
        elif t.dim() == 0:
            t = t.unsqueeze(0).expand(B, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)
        if isinstance(s, (int, float)):
            s = torch.full((B, 1), s, device=device, dtype=dtype)
        elif s.dim() == 0:
            s = s.unsqueeze(0).expand(B, 1)
        elif s.dim() == 1:
            s = s.unsqueeze(1)

        inp = torch.cat([x_t, t, s], dim=1)
        h = F.silu(self.input_proj(inp))
        for block in self.blocks:
            h = block(h)
        h = self.output_norm(h)
        out = self.output_proj(h)
        # Clamp output to prevent explosive drift updates
        return out.clamp(-10.0, 10.0)


# ---- Double-well with drift clipping ----
class DoubleWell:
    def __init__(self, d, scale=5.0):
        self.d = d
        self.scale = scale

    def potential(self, x):
        return self.scale * (x[:, 0] ** 2 - 1) ** 2

    def grad_potential(self, x):
        g = torch.zeros_like(x)
        g[:, 0] = 4 * self.scale * x[:, 0] * (x[:, 0] ** 2 - 1)
        return g

    def drift(self, x, t):
        return (-self.grad_potential(x)).clamp(-10.0, 10.0)


# ---- Controlled Transport with stability ----
class ControlledTransport:
    def __init__(self, d, ref_drift_fn, J_fn, T, dt, device,
                 width=128, depth=4, lr=1e-3):
        self.d = d
        self.ref_drift = ref_drift_fn
        self.J_fn = J_fn
        self.T = T
        self.dt = dt
        self.n_time_steps = int(T / dt)
        self.device = device
        self.width = width
        self.depth = depth
        self.lr = lr
        self.drift_increments = []

    def current_drift(self, x, t, s):
        """Evaluate accumulated drift. Previous networks are detached from
        the computation graph to prevent quadratic memory growth."""
        with torch.no_grad():
            b = self.ref_drift(x, t)
            for net, ds_val, s_val in self.drift_increments:
                b = b + net(x, t, s_val) * ds_val
        # Re-attach to graph so downstream ops (phi, divergence) can
        # differentiate w.r.t. x, but the previous-net parameters are
        # not in the graph.
        b = b.detach().requires_grad_(x.requires_grad)
        return b.clamp(-15.0, 15.0)  # Safety clamp on total drift

    def simulate_paths(self, n_paths, s, x0=None):
        if x0 is not None:
            x = x0.clone().to(self.device)
        else:
            x = torch.randn(n_paths, self.d, device=self.device)
        paths = [x.clone()]
        sqrt_2dt = math.sqrt(2.0 * self.dt)

        for i in range(self.n_time_steps):
            t = i * self.dt
            drift = self.current_drift(x, t, s)
            noise = torch.randn_like(x) * sqrt_2dt
            x = x + drift * self.dt + noise
            x = x.clamp(-5.0, 5.0)  # Prevent path divergence
            x = x.detach()
            paths.append(x.clone())

        return torch.stack(paths, dim=1)

    def train_step(self, s, delta_s, n_paths, n_opt_steps, x0_fn):
        phi = DriftNet(self.d, width=self.width, depth=self.depth).to(self.device)
        optimizer = optim.AdamW(phi.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_opt_steps, eta_min=self.lr * 0.05
        )

        best_loss = float('inf')
        nan_count = 0

        for opt_step in range(n_opt_steps):
            x0 = x0_fn(n_paths, self.d).to(self.device)
            paths = self.simulate_paths(n_paths, s, x0=x0)

            K = paths.shape[0]
            h_vals = torch.zeros(K, device=self.device)

            for i in range(self.n_time_steps):
                t_val = i * self.dt
                x = paths[:, i].detach().requires_grad_(True)
                dx_dt = (paths[:, i + 1] - paths[:, i]) / self.dt
                b_s = self.current_drift(x, t_val, s)
                phi_out = phi(x, t_val, s)

                # Compute divergence: exact for d<=5, Hutchinson for d>5
                if self.d <= 5:
                    div_phi = torch.zeros(K, device=self.device)
                    for dim_idx in range(self.d):
                        grad_phi_dim = torch.autograd.grad(
                            phi_out[:, dim_idx].sum(), x, create_graph=True
                        )[0][:, dim_idx]
                        div_phi += grad_phi_dim
                else:
                    # Hutchinson trace estimator: E[v^T (J_phi) v] = tr(J_phi)
                    # Uses 1 autograd call instead of d, cutting memory ~d×
                    v = torch.randn_like(x)  # Rademacher also works
                    vJ = torch.autograd.grad(
                        (phi_out * v).sum(), x, create_graph=True
                    )[0]
                    div_phi = (vJ * v).sum(dim=1)

                residual = (b_s.detach() - dx_dt.detach())
                h_vals = h_vals + (-self.dt / 2) * (residual * phi_out).sum(dim=1) \
                          - (self.dt / 4) * div_phi

            # J values
            J_vals = torch.zeros(K, device=self.device)
            for i in range(self.n_time_steps + 1):
                t_val = i * self.dt
                J_vals = J_vals + self.J_fn(paths[:, i].detach(), t_val)  # point-sum, not time integral — no dt factor

            J_mean = J_vals.mean()
            h_bar = h_vals.mean()
            residuals = (h_vals - h_bar) + (J_vals - J_mean)
            loss = (residuals ** 2).sum()

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
            # Aggressive gradient clipping
            torch.nn.utils.clip_grad_norm_(phi.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            if loss.item() < best_loss:
                best_loss = loss.item()

        self.drift_increments.append((phi.eval(), delta_s, s))
        return best_loss

    def run(self, n_annealing_steps, n_paths, n_opt_steps, x0_fn,
            annealing_schedule='cosine'):
        if annealing_schedule == 'cosine':
            s_vals = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n_annealing_steps + 1)))
        else:
            s_vals = np.linspace(0, 1, n_annealing_steps + 1)

        start_time = time.time()
        all_losses = []

        for step in range(n_annealing_steps):
            step_start = time.time()
            s = s_vals[step]
            delta_s = s_vals[step + 1] - s_vals[step]

            loss = self.train_step(s, delta_s, n_paths, n_opt_steps, x0_fn)
            all_losses.append(loss)

            # Free intermediate tensors to limit memory growth
            import gc
            gc.collect()
            if self.device.type == 'mps':
                torch.mps.empty_cache()
            elif self.device.type == 'cuda':
                torch.cuda.empty_cache()

            step_time = time.time() - step_start
            print(f"  Annealing {step+1}/{n_annealing_steps}, "
                  f"s={s_vals[step+1]:.4f}, Δs={delta_s:.4f}, "
                  f"loss={loss:.2f}, time={step_time:.1f}s")

        total_time = time.time() - start_time

        # Final paths (average over 3 runs for stability)
        all_final = []
        for _ in range(3):
            x0 = x0_fn(n_paths, self.d).to(self.device)
            fp = self.simulate_paths(n_paths, 1.0, x0=x0)
            all_final.append(fp)
        final_paths = torch.cat(all_final, dim=0)

        return final_paths, {'total': total_time, 'losses': all_losses}


# ---- Initialization ----
def dw_init(n, d, device='cpu'):
    """Initialize x₁ ~ N(-1, 0.01), other dims ~ N(0, 0.09)."""
    x = torch.randn(n, d) * 0.3
    x[:, 0] = -1.0 + torch.randn(n) * 0.1
    return x


# ---- Main experiment ----
def run_experiment(d, config, device):
    T = 1.0
    dt = config['dt']
    sigma_obs = config['sigma_obs']
    dw = DoubleWell(d=d)

    def ref_drift(x, t):
        return dw.drift(x, t)

    def J_fn(x, t):
        """Terminal constraint only: x₁(T) → +1."""
        J = torch.zeros(x.shape[0], device=x.device)
        # Apply terminal constraint in the last few time steps
        # Using a ramp that activates in [T-5dt, T]
        if t >= T - 5 * dt:
            weight = min(1.0, (t - (T - 5*dt)) / (5*dt))
            J = J + weight * (1.0 / (2 * sigma_obs ** 2)) * ((x[:, 0] - 1.0) ** 2)
        return J

    def x0_fn(n, d_):
        return dw_init(n, d_, device)

    ct = ControlledTransport(
        d=d, ref_drift_fn=ref_drift, J_fn=J_fn, T=T, dt=dt, device=device,
        width=config['width'], depth=config['depth'], lr=config['lr']
    )

    n_params = sum(p.numel() for p in DriftNet(d, config['width'], config['depth']).parameters())
    print(f"\n  Network: width={config['width']}, depth={config['depth']}, "
          f"params={n_params:,}")
    print(f"  Annealing steps: {config['n_annealing']}, "
          f"opt steps: {config['n_opt']}, paths: {config['n_paths']}")
    print(f"  σ_obs: {sigma_obs}, dt: {dt}, lr: {config['lr']}")

    paths, timing = ct.run(
        n_annealing_steps=config['n_annealing'],
        n_paths=config['n_paths'],
        n_opt_steps=config['n_opt'],
        x0_fn=x0_fn,
        annealing_schedule='cosine'
    )

    # Analyze
    n_total = paths.shape[0]
    terminal = paths[:, -1, :].cpu()
    initial = paths[:, 0, :].cpu()
    mid_idx = int(0.5 * T / dt)
    midpoint = paths[:, mid_idx, :].cpu()
    q1 = paths[:, int(0.25 * T / dt), :].cpu()
    q3 = paths[:, int(0.75 * T / dt), :].cpu()

    result = {
        'd': d,
        'initial_x1_mean': initial[:, 0].mean().item(),
        'initial_x1_std': initial[:, 0].std().item(),
        'q1_x1_mean': q1[:, 0].mean().item(),
        'q1_x1_std': q1[:, 0].std().item(),
        'midpoint_x1_mean': midpoint[:, 0].mean().item(),
        'midpoint_x1_std': midpoint[:, 0].std().item(),
        'q3_x1_mean': q3[:, 0].mean().item(),
        'q3_x1_std': q3[:, 0].std().item(),
        'terminal_x1_mean': terminal[:, 0].mean().item(),
        'terminal_x1_std': terminal[:, 0].std().item(),
        'n_crossing': int((terminal[:, 0] > 0).sum().item()),
        'n_total': n_total,
        'frac_crossing': (terminal[:, 0] > 0).float().mean().item(),
        'n_params': n_params,
        'timing_total': timing['total'],
        'config': config,
        'device': str(device),
    }
    return result, paths.cpu()


def plot_results(results_list, all_paths, save_dir):
    n = len(results_list)
    fig, axes = plt.subplots(2, n, figsize=(6*n, 9))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, (res, paths) in enumerate(zip(results_list, all_paths)):
        d = res['d']
        dt = res['config']['dt']
        n_steps = paths.shape[1]
        times = np.linspace(0, 1, n_steps)

        # Top: trajectories
        ax = axes[0, col]
        n_show = min(80, paths.shape[0])
        for i in range(n_show):
            color = 'steelblue' if paths[i, -1, 0] <= 0 else 'darkorange'
            ax.plot(times, paths[i, :, 0].numpy(), alpha=0.12, color=color, lw=0.5)

        mean_x1 = paths[:, :, 0].mean(dim=0).numpy()
        ax.plot(times, mean_x1, color='darkblue', lw=2, label='Mean x₁')
        ax.axhline(y=-1, color='red', ls='--', alpha=0.5, label='Wells (±1)')
        ax.axhline(y=+1, color='red', ls='--', alpha=0.5)
        ax.axhline(y=0, color='gray', ls=':', alpha=0.3)
        ax.set_xlabel('Time t', fontsize=12)
        ax.set_ylabel('x₁', fontsize=12)
        ax.set_title(
            f'd={d}: x₁(T)={res["terminal_x1_mean"]:.3f}±{res["terminal_x1_std"]:.3f}\n'
            f'{res["n_crossing"]}/{res["n_total"]} crossing ({100*res["frac_crossing"]:.0f}%)',
            fontsize=11
        )
        ax.legend(fontsize=9, loc='upper left')
        ax.set_ylim(-3, 3)

        # Bottom: terminal histogram
        ax2 = axes[1, col]
        term_x1 = paths[:, -1, 0].numpy()
        ax2.hist(term_x1, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='navy')
        ax2.axvline(x=-1, color='red', ls='--', lw=1.5, label='Wells')
        ax2.axvline(x=+1, color='red', ls='--', lw=1.5)
        ax2.axvline(x=0, color='gray', ls=':', alpha=0.5, label='Barrier')
        ax2.axvline(x=res['terminal_x1_mean'], color='darkblue', lw=2, label='Mean')
        ax2.set_xlabel('x₁(T)', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title(f'd={d}: Terminal distribution', fontsize=11)
        ax2.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, 'double_well_d12_gpu.pdf')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {path}")
    plt.close()


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    device = get_device()
    RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(RESULTS_DIR, 'results'), exist_ok=True)

    # Configurations — scaled by dimension
    configs = {
        1: dict(
            n_annealing=30,     # 30 cosine-spaced steps
            n_opt=200,          # 200 opt steps per annealing step
            n_paths=200,        # 200 paths per batch
            width=128,          # ~100K params
            depth=4,
            lr=2e-3,
            sigma_obs=0.15,     # moderate constraint (not 0.05!)
            dt=0.02,
        ),
        2: dict(
            n_annealing=30,
            n_opt=200,
            n_paths=200,
            width=192,          # ~225K params
            depth=4,
            lr=2e-3,
            sigma_obs=0.15,
            dt=0.02,
        ),
        5: dict(
            n_annealing=30,
            n_opt=200,
            n_paths=300,        # more paths for higher-d gradient estimation
            width=192,          # ~225K params (input is 7-dim: 5+t+s)
            depth=4,
            lr=1.5e-3,          # slightly lower lr for stability
            sigma_obs=0.2,      # slightly looser — penalty still 1/(2*0.04)=12.5 per obs
            dt=0.02,
        ),
        10: dict(
            n_annealing=20,     # fewer steps to fit in memory (20 nets vs 30)
            n_opt=200,
            n_paths=300,        # reduced from 400 to save memory
            width=192,          # reduced from 256 (~225K vs ~400K params)
            depth=4,
            lr=1e-3,            # lower lr for stability at high d
            sigma_obs=0.25,     # looser — penalty 1/(2*0.0625)=8 per obs
            dt=0.02,
        ),
    }

    dims = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [1, 2, 5, 10]
    results_list = []
    all_paths = []

    for d in dims:
        config = configs[d]
        print(f"\n{'='*70}")
        print(f"Double-Well TPS d={d} — Stabilized GPU Run")
        print(f"{'='*70}")

        result, paths = run_experiment(d, config, device)
        results_list.append(result)
        all_paths.append(paths)

        print(f"\n  Results for d={d}:")
        print(f"    x₁(0)    = {result['initial_x1_mean']:.4f} ± {result['initial_x1_std']:.4f}")
        print(f"    x₁(T/4)  = {result['q1_x1_mean']:.4f} ± {result['q1_x1_std']:.4f}")
        print(f"    x₁(T/2)  = {result['midpoint_x1_mean']:.4f} ± {result['midpoint_x1_std']:.4f}")
        print(f"    x₁(3T/4) = {result['q3_x1_mean']:.4f} ± {result['q3_x1_std']:.4f}")
        print(f"    x₁(T)    = {result['terminal_x1_mean']:.4f} ± {result['terminal_x1_std']:.4f}")
        print(f"    Barrier crossings: {result['n_crossing']}/{result['n_total']} "
              f"({100*result['frac_crossing']:.1f}%)")
        print(f"    Time: {result['timing_total']:.1f}s")
        print(f"    Params: {result['n_params']:,}")

        json_path = os.path.join(RESULTS_DIR, 'results', f'dw_d{d}_stable.json')
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"    Saved to {json_path}")

    if results_list:
        plot_results(results_list, all_paths, RESULTS_DIR)

    print(f"\nDone!")
