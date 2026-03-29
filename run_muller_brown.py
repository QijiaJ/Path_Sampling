#!/usr/bin/env python3
"""
Head-to-head comparison on the 2D Müller-Brown potential:
  Algorithm 1 (Controlled Transport)  vs
  Algorithm 3 (JKO / Wasserstein)     vs
  Vanilla SPDE (Eq. 6 of the paper)

Key features:
  - 2D Müller-Brown potential with 3 minima and 2 saddle points
  - Multi-point likelihood: observations at t=0, t=T/3, t=2T/3, t=T
    defining a transition path from well A → saddle → well B
  - Algorithm 3 uses equilibrium reference ν ∝ exp(−V_MB)
  - SPDE uses Crank-Nicolson discretization from Eq. 6

Run:
  python run_muller_brown.py
"""

import json, os, sys, time, math, gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ===========================================================================
# Device
# ===========================================================================
def get_device():
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("[device] Apple MPS GPU"); return torch.device('mps')
    elif torch.cuda.is_available():
        print("[device] CUDA GPU"); return torch.device('cuda')
    else:
        print("[device] CPU"); return torch.device('cpu')

DEVICE = get_device()

# ===========================================================================
# Müller-Brown Potential
# ===========================================================================
# Standard parameters
_A  = torch.tensor([-200., -100., -170.,  15.], device=DEVICE)
_a  = torch.tensor([  -1.,   -1.,  -6.5,  0.7], device=DEVICE)
_b  = torch.tensor([   0.,    0.,  11.,   0.6], device=DEVICE)
_c  = torch.tensor([ -10.,  -10.,  -6.5,  0.7], device=DEVICE)
_x0 = torch.tensor([   1.,    0.,  -0.5, -1. ], device=DEVICE)
_y0 = torch.tensor([   0.,    0.5,  1.5,  1. ], device=DEVICE)

# Approximate minima (well A deepest, well B second)
WELL_A = torch.tensor([-0.558, 1.442], device=DEVICE)   # deepest
WELL_B = torch.tensor([ 0.623, 0.028], device=DEVICE)   # second
WELL_C = torch.tensor([-0.050, 0.467], device=DEVICE)   # shallowest
# Approximate saddle between A and B (via C)
SADDLE_AC = torch.tensor([-0.822, 0.624], device=DEVICE)
SADDLE_CB = torch.tensor([ 0.212, 0.293], device=DEVICE)


def muller_brown_potential(xy):
    """
    V(x,y) = Σ_i A_i exp(a_i(x-x̄_i)² + b_i(x-x̄_i)(y-ȳ_i) + c_i(y-ȳ_i)²)
    xy: (..., 2) tensor  →  (...,) potential values
    """
    x = xy[..., 0:1]  # (..., 1)
    y = xy[..., 1:2]
    dx = x - _x0  # (..., 4)
    dy = y - _y0
    exponent = _a * dx**2 + _b * dx * dy + _c * dy**2
    return (_A * torch.exp(exponent)).sum(dim=-1)


def muller_brown_grad(xy):
    """
    Analytical gradient ∇V of the Müller-Brown potential.
    No autograd — works inside torch.no_grad() contexts.

    V = Σ_i A_i exp(E_i),  E_i = a_i(x-x̄)² + b_i(x-x̄)(y-ȳ) + c_i(y-ȳ)²
    dV/dx = Σ_i A_i exp(E_i) * (2a_i(x-x̄) + b_i(y-ȳ))
    dV/dy = Σ_i A_i exp(E_i) * (b_i(x-x̄) + 2c_i(y-ȳ))
    """
    x = xy[..., 0:1]  # (..., 1)
    y = xy[..., 1:2]
    dx = x - _x0  # (..., 4)
    dy = y - _y0
    exponent = _a * dx**2 + _b * dx * dy + _c * dy**2
    Aexp = _A * torch.exp(exponent)  # (..., 4)

    dVdx = (Aexp * (2 * _a * dx + _b * dy)).sum(dim=-1, keepdim=True)
    dVdy = (Aexp * (_b * dx + 2 * _c * dy)).sum(dim=-1, keepdim=True)

    return torch.cat([dVdx, dVdy], dim=-1)


def muller_brown_drift(xy, t=None):
    """Reference drift u^ref = −∇V, clamped for stability."""
    return (-muller_brown_grad(xy)).clamp(-50., 50.)


# ===========================================================================
# Observation model  (multi-point likelihood)
# ===========================================================================
class MultiPointLikelihood:
    """
    J(x; y) = (1/(2σ²)) Σ_k ‖y_k − x(t_k)‖²
    Applied at discrete observation times t_k.
    """
    def __init__(self, obs_points, obs_times, sigma_obs=0.3, T=1.0, dt=0.02):
        """
        obs_points: list of (2,) tensors — target positions
        obs_times:  list of floats — observation times
        sigma_obs:  observation noise
        """
        self.obs = [(t, y.to(DEVICE)) for t, y in zip(obs_times, obs_points)]
        self.sigma_obs = sigma_obs
        self.T = T
        self.dt = dt

    def J_at_time(self, x, t):
        """
        Return per-sample J contribution at physical time t.
        x: (batch, 2)
        Applies observation penalty if t is close to an observation time.
        """
        J = torch.zeros(x.shape[0], device=x.device)
        for t_obs, y_obs in self.obs:
            if abs(t - t_obs) < self.dt * 1.5:  # snap to nearest dt
                diff = x - y_obs.unsqueeze(0)
                J = J + (1.0 / (2 * self.sigma_obs**2)) * (diff**2).sum(dim=1)
        return J

    def J_at_time_ramped(self, x, t):
        """
        Ramped J: for each observation at t_k, apply a 5-step linear ramp
        in [t_k - 5*dt, t_k] so the gradient signal is spread over multiple
        timesteps rather than concentrated on a single delta.
        """
        J = torch.zeros(x.shape[0], device=x.device)
        ramp_width = 5 * self.dt
        for t_obs, y_obs in self.obs:
            if t_obs - ramp_width - self.dt * 0.5 <= t <= t_obs + self.dt * 0.5:
                if t >= t_obs - self.dt * 0.5:
                    weight = 1.0
                else:
                    weight = max(0.0, (t - (t_obs - ramp_width)) / ramp_width)
                diff = x - y_obs.unsqueeze(0)
                J = J + weight * (1.0 / (2 * self.sigma_obs**2)) * (diff**2).sum(dim=1)
        return J

    def J_at_time_single(self, x, t):
        """Single-point version: x is (2,) tensor."""
        J = 0.0
        for t_obs, y_obs in self.obs:
            if abs(t - t_obs) < self.dt * 1.5:
                diff = x - y_obs
                J = J + (1.0 / (2 * self.sigma_obs**2)) * (diff**2).sum()
        return J


# ===========================================================================
# Neural networks (moderate size for d=2)
# ===========================================================================
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
    def forward(self, x):
        h = self.norm(x)
        h = F.silu(self.fc1(h))
        return x + self.fc2(h)


class DriftNet(nn.Module):
    """Drift network for Algorithm 1: (x, t, s) → db/ds ∈ R²."""
    def __init__(self, width=128, depth=4):
        super().__init__()
        self.proj = nn.Linear(4, width)   # x(2) + t(1) + s(1)
        self.blocks = nn.ModuleList([ResBlock(width) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, 2)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
    def forward(self, x, t, s):
        B = x.shape[0]
        dev, dt = x.device, x.dtype
        if isinstance(t, (int, float)):
            t = torch.full((B,1), t, device=dev, dtype=dt)
        elif t.dim() == 0: t = t.view(1,1).expand(B,1)
        elif t.dim() == 1: t = t.unsqueeze(1)
        if isinstance(s, (int, float)):
            s = torch.full((B,1), s, device=dev, dtype=dt)
        elif s.dim() == 0: s = s.view(1,1).expand(B,1)
        elif s.dim() == 1: s = s.unsqueeze(1)
        h = F.silu(self.proj(torch.cat([x, t, s], 1)))
        for blk in self.blocks: h = blk(h)
        return self.out(self.out_norm(h)).clamp(-10., 10.)


class PushNet(nn.Module):
    """Pushforward network for Algorithm 3: (x, t) → x + Δx."""
    def __init__(self, width=128, depth=4):
        super().__init__()
        self.proj = nn.Linear(3, width)   # x(2) + t(1)
        self.blocks = nn.ModuleList([ResBlock(width) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, 2)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
    def forward(self, x, t=None):
        B = x.shape[0]
        dev, dt_ = x.device, x.dtype
        if t is None: t = 0.0
        if isinstance(t, (int, float)):
            t = torch.full((B,1), t, device=dev, dtype=dt_)
        elif t.dim() == 0: t = t.view(1,1).expand(B,1)
        elif t.dim() == 1: t = t.unsqueeze(1)
        h = F.silu(self.proj(torch.cat([x, t], 1)))
        for blk in self.blocks: h = blk(h)
        return x + self.out(self.out_norm(h))
    def gradient(self, x, t=None):
        return self.forward(x, t)


# ===========================================================================
# Algorithm 1: Controlled Transport
# ===========================================================================
class Alg1_MB:
    """Algorithm 1 on Müller-Brown with multi-point likelihood."""

    def __init__(self, lik: MultiPointLikelihood, T=1.0, dt=0.02,
                 width=128, depth=4, lr=2e-3):
        self.lik = lik
        self.T, self.dt = T, dt
        self.n_steps = int(T / dt)
        self.width, self.depth, self.lr = width, depth, lr
        self.increments = []  # (net, Δs, s)

    def _drift(self, x, t, s):
        """Accumulated drift. Previous networks detached to prevent
        quadratic graph growth across annealing steps."""
        with torch.no_grad():
            b = muller_brown_drift(x)
            for net, ds, sv in self.increments:
                b = b + net(x, t, sv) * ds
        b = b.detach().requires_grad_(x.requires_grad)
        return b.clamp(-30., 30.)

    def _simulate(self, n, s, x0):
        x = x0.clone()
        paths = [x.clone()]
        sq = math.sqrt(2 * self.dt)
        for i in range(self.n_steps):
            t = i * self.dt
            dr = self._drift(x, t, s)
            x = x + dr * self.dt + torch.randn_like(x) * sq
            x[:, 0].clamp_(-2.0, 2.0)   # MB x-range
            x[:, 1].clamp_(-1.0, 3.0)   # MB y-range (Well A at y=1.44)
            x = x.detach()
            paths.append(x.clone())
        return torch.stack(paths, 1)  # (n, T/dt+1, 2)

    def train_step(self, s, ds_val, K, n_opt, x0_fn):
        phi = DriftNet(self.width, self.depth).to(DEVICE)
        opt = optim.AdamW(phi.parameters(), lr=self.lr, weight_decay=1e-4)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, n_opt, eta_min=self.lr*0.05)
        best = float('inf')
        for _ in range(n_opt):
            x0 = x0_fn(K)
            paths = self._simulate(K, s, x0)
            B = paths.shape[0]
            h_vals = torch.zeros(B, device=DEVICE)
            for i in range(self.n_steps):
                tv = i * self.dt
                x = paths[:, i].detach().requires_grad_(True)
                dx = (paths[:, i+1] - paths[:, i]) / self.dt
                bs = self._drift(x, tv, s)
                po = phi(x, tv, s)
                div_phi = torch.zeros(B, device=DEVICE)
                for d_ in range(2):
                    g = torch.autograd.grad(po[:, d_].sum(), x, create_graph=True)[0][:, d_]
                    div_phi = div_phi + g
                res = (bs.detach() - dx.detach())
                h_vals = h_vals + (-self.dt/2)*(res*po).sum(1) - (self.dt/4)*div_phi
            # J across all observation times — ramped (5-step ramp near each obs)
            # Spreads gradient signal over ~20 timesteps instead of 4 deltas
            J_vals = torch.zeros(B, device=DEVICE)
            for i in range(self.n_steps + 1):
                tv = i * self.dt
                J_vals = J_vals + self.lik.J_at_time_ramped(paths[:, i].detach(), tv)
            hbar = h_vals.mean(); Jm = J_vals.mean()
            loss = ((h_vals - hbar) + (J_vals - Jm)).pow(2).sum()
            if not torch.isfinite(loss): continue
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(phi.parameters(), 5.0)
            opt.step(); sch.step()
            if loss.item() < best: best = loss.item()
        self.increments.append((phi.eval(), ds_val, s))
        return best

    def run(self, n_anneal=30, K=200, n_opt=200, x0_fn=None):
        s_vals = 0.5*(1 - np.cos(np.linspace(0, np.pi, n_anneal+1)))
        t0 = time.time()
        for step in range(n_anneal):
            s = s_vals[step]; ds = s_vals[step+1] - s_vals[step]
            loss = self.train_step(s, ds, K, n_opt, x0_fn)
            # Free intermediate tensors between annealing steps
            gc.collect()
            if DEVICE.type == 'mps':
                torch.mps.empty_cache()
            elif DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
            print(f"  Alg1 anneal {step+1}/{n_anneal}, s={s_vals[step+1]:.3f}, loss={loss:.1f}")
        # final paths (3 batches for better statistics)
        all_p = []
        for _ in range(3):
            all_p.append(self._simulate(K, 1.0, x0_fn(K)))
        return torch.cat(all_p, 0), time.time() - t0


# ===========================================================================
# Algorithm 3: JKO with equilibrium reference
# ===========================================================================
class Alg3_MB:
    """
    Algorithm 3 on Müller-Brown. Requires equilibrium reference ν ∝ exp(−V).
    Evolves marginals q_t via JKO pushforward steps.
    """

    def __init__(self, lik: MultiPointLikelihood, T=1.0, h=0.25,
                 width=128, depth=4, lr=1e-3):
        self.lik = lik
        self.T, self.h = T, h
        self.width, self.depth, self.lr = width, depth, lr

    @staticmethod
    def kde_score(samples, bw=None):
        n, d = samples.shape
        if bw is None: bw = n**(-1./(d+4)) * samples.std()
        diffs = samples.unsqueeze(0) - samples.unsqueeze(1)
        dsq = (diffs**2).sum(2)
        mask = 1. - torch.eye(n, device=samples.device)
        w = torch.exp(-dsq / (2*bw**2)) * mask
        w = w / w.sum(1, keepdim=True).clamp(min=1e-10)
        return -(1./bw**2) * (w.unsqueeze(2) * diffs).sum(1)

    def jko_step(self, particles, t, n_opt):
        n = particles.shape[0]
        phi = PushNet(self.width, self.depth).to(DEVICE)
        opt = optim.AdamW(phi.parameters(), lr=self.lr, weight_decay=1e-4)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, n_opt, eta_min=self.lr*0.05)
        for _ in range(n_opt):
            xn = phi(particles.detach(), t)
            # transport
            tc = (1./(2*self.h**2*n)) * ((xn - particles.detach())**2).sum()
            # Fisher via KDE score
            sc = self.kde_score(xn.detach())
            fi = 0.5 * (sc**2).sum(1).mean()
            # likelihood at t+h
            Jv = self.lik.J_at_time(xn, t + self.h)
            lc = (2./self.h) * Jv.mean()
            # potential from reference
            gV = muller_brown_grad(xn)
            pc = 0.5 * (gV**2).sum(1).mean()
            loss = tc + fi + lc + pc
            if not torch.isfinite(loss): continue
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(phi.parameters(), 5.0)
            opt.step(); sch.step()
        with torch.no_grad():
            return phi(particles, t)

    def run(self, x0, n_opt=200):
        nj = int(self.T / self.h)
        p = x0.clone().to(DEVICE)
        hist = {0.: p.clone().cpu()}
        t0 = time.time()
        for step in range(nj):
            t = step * self.h
            p = self.jko_step(p, t, n_opt)
            hist[round(t+self.h, 6)] = p.clone().cpu()
            print(f"  Alg3 JKO step {step+1}/{nj}, t={t+self.h:.2f}")
        return hist, time.time() - t0


# ===========================================================================
# Vanilla SPDE  (Eq. 6 of the paper)
# ===========================================================================
class VanillaSPDE_MB:
    """
    Langevin SPDE on path space targeting Q(x) ∝ ρ₀(x₀) exp(−I(x) − J(x;y)).
    Uses Crank-Nicolson discretization.

    δ_x log Q at time index i includes:
      (1) 0.5 * d²x/dt²  (Laplacian, handled implicitly by CN)
      (2) −0.5 * Jac(u)·u  (drift-drift coupling)
      (3) −0.5 * ∇(div u)  (divergence correction)
      (4) observation forces at t_k
    """

    def __init__(self, lik: MultiPointLikelihood, T=1.0, dt=0.02,
                 n_spde_steps=500, sigma_obs=0.3, use_mh=True,
                 drift_clamp=2.0, ds_scale=0.1):
        self.lik = lik
        self.T, self.dt = T, dt
        self.n_t = int(T / dt) + 1  # include endpoint: t=0, dt, ..., T
        self.n_spde = n_spde_steps
        self.sigma_obs = sigma_obs
        self.use_mh = use_mh
        self.drift_clamp = drift_clamp  # clamp M1, M2 per-component
        # ds = ds_scale * dt² for CN stability
        # Default ds_scale=0.1 gives noise scale sqrt(2*ds/dt)≈0.063
        # vs the old ds_scale=1.0 which gave 0.2 (way too large for MB)
        self.ds = ds_scale * dt ** 2

        # Build CN matrices (applied per spatial dimension)
        # n_t points: t_0=0, t_1=dt, ..., t_{n_t-1}=T
        L = torch.zeros(self.n_t, self.n_t, device=DEVICE)
        for i in range(self.n_t):
            L[i, i] = -2.0
            if i > 0: L[i, i-1] = 1.0
            if i < self.n_t - 1: L[i, i+1] = 1.0
        L = L / dt**2
        I = torch.eye(self.n_t, device=DEVICE)
        self.L_inv = torch.linalg.inv(I - 0.25 * self.ds * L)
        self.R_mat = I + 0.25 * self.ds * L

    def _drift_terms(self, xts):
        """
        Compute M₁, M₂ for a single path xts: (n_t, 2).
        M₁[i] = −0.5·ds · Jac(u)(x_i) · u(x_i)
        M₂[i] = −0.5·ds · ∇(div u)(x_i)

        Uses ANALYTICAL Hessian and third derivatives of the Müller-Brown
        potential to avoid nested autograd, which causes OOM over many steps.

        V = Σ_i A_i exp(a_i(x-x̄)² + b_i(x-x̄)(y-ȳ) + c_i(y-ȳ)²)
        grad V = Σ_i A_i exp(...) * [2a(x-x̄)+b(y-ȳ), b(x-x̄)+2c(y-ȳ)]
        Hessian V = Σ_i A_i exp(...) * {outer(g_i, g_i) + H_quad_i}
        where g_i = [2a(x-x̄)+b(y-ȳ), b(x-x̄)+2c(y-ȳ)] and
        H_quad_i = [[2a, b],[b, 2c]]
        """
        with torch.no_grad():
            n = xts.shape[0]
            M1 = torch.zeros_like(xts)
            M2 = torch.zeros_like(xts)

            # Vectorised over all time points at once
            x = xts[:, 0:1]  # (n, 1)
            y = xts[:, 1:2]
            dx = x - _x0  # (n, 4)
            dy = y - _y0

            exponent = _a * dx**2 + _b * dx * dy + _c * dy**2  # (n, 4)
            exp_val = torch.exp(exponent)       # (n, 4)
            Aexp = _A * exp_val                 # (n, 4)

            # First derivatives of exponent w.r.t. x, y
            gx = 2*_a*dx + _b*dy    # (n, 4)
            gy = _b*dx + 2*_c*dy    # (n, 4)

            # grad V
            dVdx = (Aexp * gx).sum(dim=1)   # (n,)
            dVdy = (Aexp * gy).sum(dim=1)

            # u = -grad V
            ux = -dVdx  # (n,)
            uy = -dVdy

            # Hessian V: Σ_i Aexp_i * (g_i ⊗ g_i + H_quad_i)
            # H_quad_i = [[2a_i, b_i], [b_i, 2c_i]]
            Hxx = (Aexp * (gx**2 + 2*_a)).sum(dim=1)
            Hxy = (Aexp * (gx*gy + _b)).sum(dim=1)
            Hyy = (Aexp * (gy**2 + 2*_c)).sum(dim=1)

            # Jac(u) = -Hessian(V)
            # Jac(u) @ u:
            # [[-Hxx, -Hxy], [-Hxy, -Hyy]] @ [-dVdx, -dVdy] = [Hxx*dVdx + Hxy*dVdy, Hxy*dVdx + Hyy*dVdy]
            Ju_x = Hxx * dVdx + Hxy * dVdy
            Ju_y = Hxy * dVdx + Hyy * dVdy
            M1[:, 0] = -0.5 * self.ds * Ju_x
            M1[:, 1] = -0.5 * self.ds * Ju_y

            # div(u) = -(Hxx + Hyy) = -trace(Hessian V)
            # ∇(div u) = -∇(Hxx + Hyy)
            # Need third derivatives of V for ∇(Hxx + Hyy)
            # Hxx = Σ Aexp*(gx² + 2a)
            # d(Hxx)/dx = Σ Aexp*(gx*(gx² + 2a) + 2*gx*(2a))  -- chain rule
            # More carefully:
            # d/dx[Aexp*(gx²+2a)] = Aexp*gx*(gx²+2a) + Aexp*2*gx*(2a)
            # Wait, let me be precise:
            # d/dx[exp(E)] = exp(E) * dE/dx = exp(E) * gx
            # d/dx[gx] = 2a (since gx = 2a*(x-x0) + b*(y-y0))
            # d/dx[gx²+2a] = 2*gx * 2a = 4a*gx  (2a is constant)
            # So d(Hxx)/dx = Σ A*exp(E)*gx*(gx²+2a) + Σ A*exp(E)*4a*gx
            #             = Σ Aexp * gx * (gx² + 2a + 4a)
            #             = Σ Aexp * gx * (gx² + 6a)
            dHxx_dx = (Aexp * gx * (gx**2 + 6*_a)).sum(dim=1)
            # d(Hxx)/dy: d/dy[exp(E)] = exp(E)*gy, d/dy[gx]=b, d/dy[gx²+2a]=2*gx*b
            dHxx_dy = (Aexp * (gy*(gx**2 + 2*_a) + 2*gx*_b)).sum(dim=1)

            # Hyy = Σ Aexp*(gy²+2c)
            # d(Hyy)/dx: d/dx[exp(E)]=exp(E)*gx, d/dx[gy]=b, d/dx[gy²+2c]=2*gy*b
            dHyy_dx = (Aexp * (gx*(gy**2 + 2*_c) + 2*gy*_b)).sum(dim=1)
            # d(Hyy)/dy = Σ Aexp * gy * (gy²+2c) + Σ Aexp * 2*gy*2c
            #           = Σ Aexp * gy * (gy² + 6c)
            dHyy_dy = (Aexp * gy * (gy**2 + 6*_c)).sum(dim=1)

            # ∇(div u) = -∇(Hxx+Hyy)
            M2[:, 0] = -0.5 * self.ds * (-(dHxx_dx + dHyy_dx))
            M2[:, 1] = -0.5 * self.ds * (-(dHxx_dy + dHyy_dy))

        return M1, M2

    def _log_target(self, xts, s):
        """
        Log-density of the path-space target (up to constants):
          log Q(x) = -I(x) - s*J(x;y)
        where I(x) = ∫ ½|ẋ - u(x)|² dt  (Onsager-Machlup action)
        and   J(x;y) = Σ_k (1/(2σ²)) ||x(t_k) - y_k||²
        Used for Metropolis-Hastings acceptance.
        """
        # OM action: I = Σ_i 0.5 * |dx/dt - u(x_i)|² * dt
        # Need enable_grad() because muller_brown_drift uses autograd internally
        om = 0.0
        with torch.enable_grad():
            for i in range(self.n_t - 1):
                dx_dt = (xts[i+1] - xts[i]) / self.dt
                u = muller_brown_drift(xts[i:i+1]).squeeze(0)  # -∇V, clamped
                om += 0.5 * ((dx_dt - u)**2).sum().item() * self.dt
        with torch.no_grad():
            # Likelihood
            J = 0.0
            for t_obs, y_obs in self.lik.obs:
                idx = int(round(t_obs / self.dt))
                if 0 <= idx < self.n_t:
                    J += (1.0 / (2 * self.sigma_obs**2)) * ((xts[idx] - y_obs)**2).sum().item()
            return -om - s * J

    def _obs_force(self, xts, s):
        """
        Observation forces: s/σ² * (y_k − x(t_k)) at observation times.
        xts: (n_t, 2), s: annealing parameter.
        Returns: (n_t, 2) force tensor.
        """
        force = torch.zeros_like(xts)
        for t_obs, y_obs in self.lik.obs:
            idx = int(round(t_obs / self.dt))
            if 0 <= idx < self.n_t:
                force[idx] += self.ds * (s / self.sigma_obs**2) * (y_obs - xts[idx].detach())
        return force

    def spde_step(self, xts, s, diag=False):
        """Single CN step for a path xts: (n_t, 2), with optional MH correction."""
        M1, M2 = self._drift_terms(xts)
        obs = self._obs_force(xts, s)
        noise = torch.randn_like(xts) * math.sqrt(2 * self.ds / self.dt)

        # Clamp M1, M2 to prevent explosive drift from steep MB regions
        if self.drift_clamp > 0:
            M1 = M1.clamp(-self.drift_clamp * self.ds, self.drift_clamp * self.ds)
            M2 = M2.clamp(-self.drift_clamp * self.ds, self.drift_clamp * self.ds)

        # Laplacian contribution (implicit via CN): R @ x part
        lap_contrib = torch.zeros_like(xts)
        for d_ in range(2):
            lap_contrib[:, d_] = self.R_mat @ xts[:, d_] - xts[:, d_]  # net change from Laplacian

        if diag:
            print(f"    [SPDE term magnitudes at s={s:.3f}]")
            print(f"      |Laplacian (R@x - x)|  = {lap_contrib.abs().mean().item():.6f}")
            print(f"      |M1 (Jac·u) clamped|   = {M1.abs().mean().item():.6f}")
            print(f"      |M2 (∇ div u) clamped| = {M2.abs().mean().item():.6f}")
            print(f"      |obs force|             = {obs.abs().mean().item():.6f}  (nonzero entries: {(obs.abs() > 1e-10).sum().item()})")
            print(f"      |noise|                 = {noise.abs().mean().item():.6f}")
            print(f"      ds={self.ds:.6f}, dt={self.dt}, ds/dt={self.ds/self.dt:.6f}")
            for t_obs, y_obs in self.lik.obs:
                idx = int(round(t_obs / self.dt))
                if 0 <= idx < self.n_t:
                    dist = (xts[idx] - y_obs).norm().item()
                    frc = obs[idx].norm().item()
                    print(f"      obs @ t={t_obs:.3f} (idx={idx}): |x-y|={dist:.4f}, |force|={frc:.6f}")

        xts_new = torch.zeros_like(xts)
        for d_ in range(2):
            rhs = self.R_mat @ xts[:, d_] + M1[:, d_] + M2[:, d_] + obs[:, d_] + noise[:, d_]
            xts_new[:, d_] = self.L_inv @ rhs

        # Soft clamp: keep paths in the physically meaningful MB region
        xts_new[:, 0].clamp_(-2.0, 2.0)
        xts_new[:, 1].clamp_(-1.0, 3.0)

        # Metropolis-Hastings correction: accept/reject based on log-target
        if self.use_mh:
            log_p_old = self._log_target(xts, s)
            log_p_new = self._log_target(xts_new, s)
            log_alpha = log_p_new - log_p_old
            # Clamp to avoid overflow in exp
            log_alpha = max(min(log_alpha, 0.0), -20.0)
            if math.log(max(np.random.rand(), 1e-30)) < log_alpha:
                return xts_new  # accept
            else:
                return xts  # reject, keep old path
        return xts_new

    def init_path(self, obs_points, obs_times):
        """Initialize path by linear interpolation between observation points."""
        path = torch.zeros(self.n_t, 2, device=DEVICE)
        # Build piecewise linear interpolation
        # obs_times already includes t=0 and t=T in our setup, but be safe
        all_t = sorted(set([0.] + list(obs_times) + [self.T]))
        # Match points to times
        time_to_pt = {t: p for t, p in zip(obs_times, obs_points)}
        if 0. not in time_to_pt: time_to_pt[0.] = obs_points[0]
        if self.T not in time_to_pt: time_to_pt[self.T] = obs_points[-1]
        all_p = [time_to_pt[t] for t in all_t]
        for i in range(self.n_t):
            t = i * self.dt
            # Find interval
            for k in range(len(all_t) - 1):
                if t <= all_t[k+1] + 1e-8:
                    frac = (t - all_t[k]) / max(all_t[k+1] - all_t[k], 1e-8)
                    frac = min(max(frac, 0.), 1.)
                    path[i] = (1 - frac) * all_p[k] + frac * all_p[k+1]
                    break
        # Add noise
        path += 0.05 * torch.randn_like(path)
        return path

    def run(self, n_paths=30, n_anneal_steps=None):
        if n_anneal_steps is None:
            n_anneal_steps = self.n_spde
        s_schedule = np.linspace(0, 1, n_anneal_steps + 1)

        obs_pts = [y for _, y in self.lik.obs]
        obs_ts = [t for t, _ in self.lik.obs]

        paths = [self.init_path(obs_pts, obs_ts) for _ in range(n_paths)]
        t0 = time.time()

        n_accept = 0
        n_total = 0

        for step_idx in range(1, len(s_schedule)):
            s = s_schedule[step_idx]
            is_diag_step = (step_idx % max(1, n_anneal_steps // 10) == 0)
            for pi in range(n_paths):
                do_diag = is_diag_step and (pi == 0)
                old_path = paths[pi]
                paths[pi] = self.spde_step(paths[pi], s, diag=do_diag).detach()
                n_total += 1
                # Track MH acceptance (if MH on, check if path changed)
                if self.use_mh:
                    if not torch.equal(paths[pi], old_path):
                        n_accept += 1
                else:
                    n_accept += 1  # no MH = always accept
            if is_diag_step:
                acc_rate = n_accept / max(n_total, 1)
                gc.collect()
                if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                print(f"  SPDE step {step_idx}/{n_anneal_steps}, s={s:.3f}, "
                      f"accept={acc_rate:.3f} ({n_accept}/{n_total})")

        elapsed = time.time() - t0
        acc_rate = n_accept / max(n_total, 1)
        print(f"  SPDE final acceptance rate: {acc_rate:.3f}")
        return paths, elapsed, acc_rate


# ===========================================================================
# Stationary sampling for Algorithm 3 init (overdamped Langevin MCMC)
# ===========================================================================
def sample_mb_stationary(n, n_mcmc=20000, dt_mcmc=1e-4):
    """
    Sample from ν ∝ exp(−V_MB) via overdamped Langevin.
    Initialize from all 3 wells (weighted by depth) for better mixing,
    and run long enough to equilibrate.
    """
    # Start from mixture of wells (proportions roughly matching Boltzmann weights)
    # Well A is deepest, so it gets the most particles
    n_a = n // 2
    n_b = n // 4
    n_c = n - n_a - n_b
    x_a = WELL_A.unsqueeze(0).repeat(n_a, 1) + 0.05 * torch.randn(n_a, 2, device=DEVICE)
    x_b = WELL_B.unsqueeze(0).repeat(n_b, 1) + 0.05 * torch.randn(n_b, 2, device=DEVICE)
    x_c = WELL_C.unsqueeze(0).repeat(n_c, 1) + 0.05 * torch.randn(n_c, 2, device=DEVICE)
    x = torch.cat([x_a, x_b, x_c], 0)
    # Shuffle
    perm = torch.randperm(n, device=DEVICE)
    x = x[perm]

    sq = math.sqrt(2 * dt_mcmc)
    for _ in range(n_mcmc):
        g = muller_brown_grad(x)
        x = x - g * dt_mcmc + sq * torch.randn_like(x)
        x[:, 0].clamp_(-2., 2.); x[:, 1].clamp_(-1., 3.)
    return x.detach()


# ===========================================================================
# Head-to-head comparison
# ===========================================================================
def run_comparison():
    T = 1.0
    dt = 0.02
    sigma_obs = 0.3

    # Observation waypoints: A → saddle_AC → saddle_CB → B
    obs_times = [0.0, T/3, 2*T/3, T]
    obs_points = [WELL_A.clone(), SADDLE_AC.clone(), SADDLE_CB.clone(), WELL_B.clone()]

    lik = MultiPointLikelihood(obs_points, obs_times, sigma_obs=sigma_obs, T=T, dt=dt)

    print("\n" + "="*70)
    print("Müller-Brown Head-to-Head Comparison")
    print(f"  Observations at t = {obs_times}")
    print(f"  Waypoints: A={obs_points[0].cpu().numpy()} → {obs_points[1].cpu().numpy()}")
    print(f"             → {obs_points[2].cpu().numpy()} → B={obs_points[3].cpu().numpy()}")
    print(f"  σ_obs = {sigma_obs}, T = {T}, dt = {dt}")
    print("="*70)

    results = {}

    # ------ Algorithm 1 ------
    print("\n--- Algorithm 1 (Controlled Transport) ---")
    K1 = 200
    def x0_fn(n):
        return WELL_A.unsqueeze(0).repeat(n, 1).to(DEVICE) + 0.1*torch.randn(n, 2, device=DEVICE)

    alg1 = Alg1_MB(lik, T=T, dt=dt, width=128, depth=4, lr=2e-3)
    a1_paths, a1_time = alg1.run(n_anneal=30, K=K1, n_opt=200, x0_fn=x0_fn)
    a1_paths_cpu = a1_paths.cpu()

    # Evaluate at observation times
    a1_obs_eval = {}
    for t_obs, y_obs in zip(obs_times, obs_points):
        idx = int(round(t_obs / dt))
        pts = a1_paths_cpu[:, idx, :]
        err = (pts - y_obs.cpu().unsqueeze(0)).norm(dim=1).mean().item()
        a1_obs_eval[f"t={t_obs:.2f}"] = {"mean": pts.mean(0).tolist(), "err": err}

    results['alg1'] = {
        'time': a1_time,
        'n_paths': a1_paths_cpu.shape[0],
        'obs_eval': a1_obs_eval,
        'terminal_mean': a1_paths_cpu[:, -1].mean(0).tolist(),
        'terminal_std': a1_paths_cpu[:, -1].std(0).tolist(),
    }
    print(f"  Time: {a1_time:.1f}s, Paths: {a1_paths_cpu.shape[0]}")
    for k, v in a1_obs_eval.items():
        print(f"  {k}: mean={v['mean']}, err={v['err']:.4f}")

    # ------ Algorithm 3 ------
    print("\n--- Algorithm 3 (JKO / Wasserstein) ---")
    N3 = 200
    # Initialize near Well A (the t=0 observation), NOT from the full equilibrium.
    # The full equilibrium spreads particles across all 3 wells, but the t=0
    # observation pins the path at Well A — starting from a mixture causes
    # the JKO pushforward to average over wells rather than follow the A→B path.
    # We use a short Langevin burn-in from Well A to get a local equilibrium sample
    # in the Well A basin.
    x0_eq = WELL_A.unsqueeze(0).repeat(N3, 1) + 0.1 * torch.randn(N3, 2, device=DEVICE)
    # Short burn-in within Well A basin (small step size to stay local)
    sq_mcmc = math.sqrt(2 * 1e-4)
    for _ in range(2000):
        g = muller_brown_grad(x0_eq)
        x0_eq = x0_eq - g * 1e-4 + sq_mcmc * torch.randn_like(x0_eq)
        x0_eq[:, 0].clamp_(-1.5, 0.5)  # stay in Well A basin
        x0_eq[:, 1].clamp_(0.5, 2.0)
    x0_eq = x0_eq.detach()
    print(f"  Sampled {N3} particles near Well A, mean={x0_eq.mean(0).cpu().numpy()}")

    # h = T/3 so JKO grid {0, T/3, 2T/3, T} aligns with observation times
    alg3 = Alg3_MB(lik, T=T, h=T/3, width=128, depth=4, lr=1e-3)
    a3_hist, a3_time = alg3.run(x0_eq, n_opt=200)

    a3_obs_eval = {}
    for t_obs, y_obs in zip(obs_times, obs_points):
        # Find nearest JKO time
        nearest_t = min(a3_hist.keys(), key=lambda x: abs(x - t_obs))
        pts = a3_hist[nearest_t]
        err = (pts - y_obs.cpu().unsqueeze(0)).norm(dim=1).mean().item()
        a3_obs_eval[f"t={t_obs:.2f}"] = {"mean": pts.mean(0).tolist(), "err": err,
                                          "jko_t": nearest_t}

    results['alg3'] = {
        'time': a3_time,
        'n_particles': N3,
        'h': round(T/3, 6),
        'obs_eval': a3_obs_eval,
    }
    print(f"  Time: {a3_time:.1f}s, Particles: {N3}")
    for k, v in a3_obs_eval.items():
        print(f"  {k} (JKO t={v['jko_t']:.2f}): mean={v['mean']}, err={v['err']:.4f}")

    # ------ Vanilla SPDE (without MH) ------
    print("\n--- Vanilla SPDE (Eq. 6, no MH) ---")
    N_spde = 50
    spde_nomh = VanillaSPDE_MB(lik, T=T, dt=dt, n_spde_steps=1500,
                                sigma_obs=sigma_obs, use_mh=False,
                                drift_clamp=2.0, ds_scale=0.1)
    spde_nomh_paths, spde_nomh_time, _ = spde_nomh.run(n_paths=N_spde)

    def _eval_spde_paths(paths, label):
        obs_eval = {}
        for t_obs, y_obs in zip(obs_times, obs_points):
            idx = int(round(t_obs / dt))
            idx = min(idx, paths[0].shape[0] - 1)
            pts = torch.stack([p[idx].cpu() for p in paths])
            err = (pts - y_obs.cpu().unsqueeze(0)).norm(dim=1).mean().item()
            obs_eval[f"t={t_obs:.2f}"] = {"mean": pts.mean(0).tolist(), "err": err}
        print(f"  {label}:")
        for k, v in obs_eval.items():
            print(f"    {k}: mean={v['mean']}, err={v['err']:.4f}")
        return obs_eval

    spde_nomh_eval = _eval_spde_paths(spde_nomh_paths, "SPDE (no MH)")

    # ------ Vanilla SPDE (with MH) ------
    print("\n--- Vanilla SPDE (Eq. 6, with MH) ---")
    spde_mh = VanillaSPDE_MB(lik, T=T, dt=dt, n_spde_steps=1500,
                              sigma_obs=sigma_obs, use_mh=True,
                              drift_clamp=2.0, ds_scale=0.1)
    spde_mh_paths, spde_mh_time, spde_mh_acc = spde_mh.run(n_paths=N_spde)
    spde_mh_eval = _eval_spde_paths(spde_mh_paths, "SPDE (with MH)")

    # Pick the better SPDE result for the main comparison
    nomh_avg = np.mean([spde_nomh_eval[f't={t:.2f}']['err'] for t in obs_times])
    mh_avg = np.mean([spde_mh_eval[f't={t:.2f}']['err'] for t in obs_times])
    print(f"\n  SPDE no-MH avg err: {nomh_avg:.4f}, MH avg err: {mh_avg:.4f}")

    # Use no-MH as the primary "spde" result (raw Langevin), add MH as separate entry
    results['spde'] = {
        'time': spde_nomh_time,
        'n_paths': N_spde,
        'n_spde_steps': 1500,
        'use_mh': False,
        'drift_clamp': 2.0,
        'ds_scale': 0.1,
        'obs_eval': spde_nomh_eval,
    }
    results['spde_mh'] = {
        'time': spde_mh_time,
        'n_paths': N_spde,
        'n_spde_steps': 1500,
        'use_mh': True,
        'drift_clamp': 2.0,
        'ds_scale': 0.1,
        'acceptance_rate': spde_mh_acc,
        'obs_eval': spde_mh_eval,
    }

    # Use whichever SPDE has lower avg error for plotting
    if mh_avg < nomh_avg:
        spde_paths = spde_mh_paths
        print("  Using SPDE (MH) for plotting (lower avg error)")
    else:
        spde_paths = spde_nomh_paths
        print("  Using SPDE (no MH) for plotting (lower avg error)")

    return results, a1_paths_cpu, a3_hist, spde_paths, obs_times, obs_points


# ===========================================================================
# Plotting
# ===========================================================================
def plot_comparison(results, a1_paths, a3_hist, spde_paths,
                    obs_times, obs_points, save_dir):
    """6-panel figure: potential + 3 methods with paths + summary."""

    # Grid for potential contour
    xg = np.linspace(-1.8, 1.5, 200)
    yg = np.linspace(-0.5, 2.2, 200)
    X, Y = np.meshgrid(xg, yg)
    xy_grid = torch.tensor(np.stack([X.ravel(), Y.ravel()], 1), dtype=torch.float32,
                           device=DEVICE)
    V_grid = muller_brown_potential(xy_grid).cpu().numpy().reshape(X.shape)
    V_grid = np.clip(V_grid, -200, 100)  # clip for visualization

    obs_np = [p.cpu().numpy() for p in obs_points]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    def plot_potential(ax, title):
        cs = ax.contourf(X, Y, V_grid, levels=30, cmap='viridis', alpha=0.7)
        ax.contour(X, Y, V_grid, levels=15, colors='k', alpha=0.3, linewidths=0.5)
        for i, (t, p) in enumerate(zip(obs_times, obs_np)):
            ax.plot(p[0], p[1], 'r*', markersize=14, zorder=10)
            ax.annotate(f't={t:.2f}', (p[0]+0.05, p[1]+0.05), fontsize=9, color='red',
                       fontweight='bold')
        ax.set_xlabel('x', fontsize=11); ax.set_ylabel('y', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(-1.8, 1.5); ax.set_ylim(-0.5, 2.2)
        return cs

    # (0,0) Potential landscape
    cs = plot_potential(axes[0, 0], 'Müller-Brown V(x,y)\n& observation waypoints')
    plt.colorbar(cs, ax=axes[0, 0], shrink=0.8)

    # (0,1) Algorithm 1 paths
    plot_potential(axes[0, 1], f'Algorithm 1 (Controlled Transport)\nTime: {results["alg1"]["time"]:.1f}s')
    n_show = min(60, a1_paths.shape[0])
    for i in range(n_show):
        axes[0, 1].plot(a1_paths[i, :, 0].numpy(), a1_paths[i, :, 1].numpy(),
                       'w-', alpha=0.15, lw=0.5)
    # mean path
    mp = a1_paths[:n_show].mean(0)
    axes[0, 1].plot(mp[:, 0].numpy(), mp[:, 1].numpy(), 'cyan', lw=2, label='Mean')
    axes[0, 1].legend(fontsize=9)

    # (0,2) Algorithm 3 marginals
    plot_potential(axes[0, 2], f'Algorithm 3 (JKO / Wasserstein)\nTime: {results["alg3"]["time"]:.1f}s')
    colors_jko = plt.cm.cool(np.linspace(0, 1, len(a3_hist)))
    for idx, (t, pts) in enumerate(sorted(a3_hist.items())):
        axes[0, 2].scatter(pts[:, 0].numpy(), pts[:, 1].numpy(),
                          c=[colors_jko[idx]], s=8, alpha=0.4, zorder=5)
    # Draw mean trajectory through JKO marginal means
    sorted_times = sorted(a3_hist.keys())
    jko_means = np.array([a3_hist[t].mean(0).numpy() for t in sorted_times])
    axes[0, 2].plot(jko_means[:, 0], jko_means[:, 1], 'cyan', lw=2.5,
                   marker='s', markersize=7, zorder=8, label='Mean')
    axes[0, 2].legend(fontsize=9)

    # (1,0) SPDE paths
    plot_potential(axes[1, 0], f'Vanilla SPDE (Eq. 6)\nTime: {results["spde"]["time"]:.1f}s')
    for i, p in enumerate(spde_paths):
        pc = p.cpu()
        axes[1, 0].plot(pc[:, 0].detach().numpy(), pc[:, 1].detach().numpy(), 'w-', alpha=0.2, lw=0.5)
    # mean
    spde_stack = torch.stack([p.cpu() for p in spde_paths])
    sp_mean = spde_stack.mean(0)
    axes[1, 0].plot(sp_mean[:, 0].numpy(), sp_mean[:, 1].numpy(), 'cyan', lw=2, label='Mean')
    axes[1, 0].legend(fontsize=9)

    # (1,1) Observation error comparison — include SPDE+MH if available
    ax = axes[1, 1]
    if 'spde_mh' in results:
        methods = ['Alg 1', 'Alg 3', 'SPDE', 'SPDE+MH']
        keys = ['alg1', 'alg3', 'spde', 'spde_mh']
        colors = ['steelblue', 'darkorange', 'forestgreen', 'mediumpurple']
        bar_width = 0.18
    else:
        methods = ['Alg 1', 'Alg 3', 'SPDE']
        keys = ['alg1', 'alg3', 'spde']
        colors = ['steelblue', 'darkorange', 'forestgreen']
        bar_width = 0.22
    x_pos = np.arange(len(obs_times))
    for mi, (method, key, col) in enumerate(zip(methods, keys, colors)):
        errs = []
        for t_obs in obs_times:
            k = f"t={t_obs:.2f}"
            errs.append(results[key]['obs_eval'][k]['err'])
        ax.bar(x_pos + mi*bar_width, errs, bar_width, label=method, color=col, alpha=0.8)
    ax.set_xticks(x_pos + len(methods)/2*bar_width - bar_width/2)
    ax.set_xticklabels([f't={t:.2f}' for t in obs_times], fontsize=10)
    ax.set_ylabel('Mean distance to observation', fontsize=11)
    ax.set_title('Constraint satisfaction at each waypoint', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # (1,2) Summary table
    ax = axes[1, 2]
    ax.axis('off')
    n_methods = len(methods)
    header = [''] + methods
    row_type = ['Type', 'Path-space\nannealing', 'Marginal\nJKO push']
    row_type += ['Path-space\nLangevin'] * (n_methods - 2)
    row_time = ['Time (s)'] + [f'{results[k]["time"]:.1f}' for k in keys]
    row_np = ['Paths/Ptcls']
    for k in keys:
        if 'n_paths' in results[k]:
            row_np.append(str(results[k]['n_paths']))
        else:
            row_np.append(str(results[k]['n_particles']))

    table_data = [header, row_type, row_time, row_np]

    # Per-waypoint errors
    for t_obs in obs_times:
        row = [f'Err @ t={t_obs:.2f}']
        for k in keys:
            row.append(f'{results[k]["obs_eval"][f"t={t_obs:.2f}"]["err"]:.3f}')
        table_data.append(row)

    # Avg error row
    avg_row = ['Avg obs error']
    for k in keys:
        errs = [results[k]['obs_eval'][f't={t:.2f}']['err'] for t in obs_times]
        avg_row.append(f'{np.mean(errs):.3f}')
    table_data.append(avg_row)

    # MH acceptance rate if available
    if 'spde_mh' in results and 'acceptance_rate' in results['spde_mh']:
        acc_row = ['MH accept'] + ['—'] * (n_methods - 1) + [f'{results["spde_mh"]["acceptance_rate"]:.3f}']
        table_data.append(acc_row)

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for j in range(n_methods + 1):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')
    # Highlight avg error row
    n_rows = len(table_data)
    avg_row_idx = n_rows - (2 if 'spde_mh' in results else 1)
    for j in range(n_methods + 1):
        table[avg_row_idx, j].set_facecolor('#E6F0FF')
        table[avg_row_idx, j].set_text_props(fontweight='bold')
    ax.set_title('Summary', fontsize=12, pad=20)

    plt.tight_layout()
    path = os.path.join(save_dir, 'muller_brown_comparison.pdf')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {path}")
    plt.close()


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(RESULTS_DIR, 'results'), exist_ok=True)

    results, a1_paths, a3_hist, spde_paths, obs_times, obs_points = run_comparison()

    # Save JSON
    json_path = os.path.join(RESULTS_DIR, 'results', 'muller_brown.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    # Plot
    plot_comparison(results, a1_paths, a3_hist, spde_paths,
                    obs_times, obs_points, RESULTS_DIR)

    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'Time(s)':>8} {'Avg Obs Err':>12}")
    print(f"{'-'*70}")
    for name, key in [('Alg 1 (Ctrl Transport)', 'alg1'),
                      ('Alg 3 (JKO/Wass)', 'alg3'),
                      ('SPDE (Eq. 6)', 'spde')]:
        errs = [results[key]['obs_eval'][f't={t:.2f}']['err']
                for t in [0.0, 1/3, 2/3, 1.0]]
        print(f"{name:<25} {results[key]['time']:>8.1f} {np.mean(errs):>12.4f}")
    print(f"{'='*70}")
    print("Done!")
