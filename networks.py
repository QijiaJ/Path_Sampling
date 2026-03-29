"""
Neural network architectures for Algorithm 1 and Algorithm 3.

DriftNetwork    — drift increment db/ds for Algorithm 1 (Controlled Transport)
ICNN            — input-convex neural network for Algorithm 3 (JKO pushforward)
PushforwardMLP  — standard MLP pushforward for Algorithm 3 (non-convex alternative)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DriftNetwork(nn.Module):
    """
    Network for Algorithm 1: maps (x, t, s) -> db/ds in R^d.

    Architecture: MLP with configurable hidden layers.
    Input: concatenation of x (d), t (1), s (1) -> d+2 dimensional.
    Output: d dimensional drift increment, clamped for stability.
    """

    def __init__(self, d, hidden_dims=(20, 30)):
        super().__init__()
        self.d = d
        layers = []
        in_dim = d + 2  # x (d) + t (1) + s (1)
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.SiLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, d))
        self.net = nn.Sequential(*layers)

        # Zero-initialize output layer for stable start
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, t, s):
        """
        x: (batch, d)
        t: scalar or (batch,) or (batch, 1)
        s: scalar or (batch,) or (batch, 1)
        Returns: (batch, d)
        """
        B = x.shape[0]
        dev, dtype = x.device, x.dtype

        if isinstance(t, (int, float)):
            t = torch.full((B, 1), t, device=dev, dtype=dtype)
        elif t.dim() == 0:
            t = t.view(1, 1).expand(B, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)

        if isinstance(s, (int, float)):
            s = torch.full((B, 1), s, device=dev, dtype=dtype)
        elif s.dim() == 0:
            s = s.view(1, 1).expand(B, 1)
        elif s.dim() == 1:
            s = s.unsqueeze(1)

        inp = torch.cat([x, t, s], dim=1)
        return self.net(inp)


class ICNN(nn.Module):
    """
    Input-Convex Neural Network for Algorithm 3 pushforward maps.

    The network is convex in x (the spatial input), which guarantees that
    grad_x phi(x, t) defines a valid optimal transport map.

    Architecture follows Amos et al. (2017):
    - z-path: nonneg weights, quadratic input skip connections
    - context (t): injected via bias modulation at each layer

    forward(x, t)  -> scalar phi(x, t)
    gradient(x, t)  -> grad_x phi(x, t), the pushforward map
    """

    def __init__(self, d, hidden_dims=(64, 64), context_dim=1):
        super().__init__()
        self.d = d
        self.context_dim = context_dim
        h_dims = list(hidden_dims)

        # Context embedding (time -> bias modulation)
        self.context_net = nn.Sequential(
            nn.Linear(context_dim, h_dims[0]),
            nn.SiLU(),
            nn.Linear(h_dims[0], h_dims[0]),
        )

        # First layer (unconstrained)
        self.fc0 = nn.Linear(d, h_dims[0])

        # Hidden layers: z-path (nonneg weights) + x skip connections
        self.wz = nn.ModuleList()  # z -> z (nonneg)
        self.wx = nn.ModuleList()  # x -> z (unconstrained, quadratic)
        for i in range(len(h_dims) - 1):
            self.wz.append(nn.Linear(h_dims[i], h_dims[i + 1], bias=True))
            self.wx.append(nn.Linear(d, h_dims[i + 1], bias=False))

        # Output layer
        self.wz_out = nn.Linear(h_dims[-1], 1, bias=False)
        self.wx_out = nn.Linear(d, 1, bias=False)

        # Initialize nonneg weights to be positive
        for layer in list(self.wz) + [self.wz_out]:
            nn.init.uniform_(layer.weight, 0.0, 0.1)

    def _clamp_nonneg(self):
        """Clamp z-path weights to be nonneg (called before forward)."""
        for layer in list(self.wz) + [self.wz_out]:
            layer.weight.data.clamp_(min=0.0)

    def forward(self, x, t=None):
        """
        x: (batch, d), t: scalar or tensor
        Returns: (batch,) scalar convex function values
        """
        self._clamp_nonneg()
        B = x.shape[0]
        dev, dtype = x.device, x.dtype

        # Context embedding
        if t is None:
            t = torch.zeros(B, self.context_dim, device=dev, dtype=dtype)
        elif isinstance(t, (int, float)):
            t = torch.full((B, self.context_dim), t, device=dev, dtype=dtype)
        elif t.dim() == 0:
            t = t.view(1, 1).expand(B, self.context_dim)
        elif t.dim() == 1:
            t = t.unsqueeze(1)

        ctx = self.context_net(t)  # (B, h0)

        # First layer
        z = F.softplus(self.fc0(x) + ctx)

        # Hidden layers
        for wz_l, wx_l in zip(self.wz, self.wx):
            z = F.softplus(wz_l(z) + wx_l(x))

        # Output: scalar
        out = self.wz_out(z) + self.wx_out(x)
        # Add quadratic term for strong convexity
        out = out.squeeze(-1) + 0.5 * (x ** 2).sum(dim=1)
        return out

    def gradient(self, x, t=None):
        """
        Compute grad_x phi(x, t) — the optimal transport map.
        x must have requires_grad=True (or it will be set).
        """
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        phi = self.forward(x, t)
        grad = torch.autograd.grad(phi.sum(), x, create_graph=True)[0]
        return grad


class PushforwardMLP(nn.Module):
    """
    Standard (non-convex) MLP pushforward for Algorithm 3.

    Maps x -> x + f(x, t), i.e. a residual architecture.
    Does not guarantee convexity, but is more expressive in practice.
    """

    def __init__(self, d, hidden_dims=(64, 64), context_dim=1):
        super().__init__()
        self.d = d
        self.context_dim = context_dim

        layers = []
        in_dim = d + context_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.SiLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, d))
        self.net = nn.Sequential(*layers)

        # Zero-init output for identity-start
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, t=None):
        """
        x: (batch, d), t: scalar or tensor
        Returns: (batch, d) = x + f(x, t)
        """
        B = x.shape[0]
        dev, dtype = x.device, x.dtype

        if t is None:
            t = torch.zeros(B, self.context_dim, device=dev, dtype=dtype)
        elif isinstance(t, (int, float)):
            t = torch.full((B, self.context_dim), t, device=dev, dtype=dtype)
        elif t.dim() == 0:
            t = t.view(1, 1).expand(B, self.context_dim)
        elif t.dim() == 1:
            t = t.unsqueeze(1)

        inp = torch.cat([x, t], dim=1)
        return x + self.net(inp)

    def gradient(self, x, t=None):
        """For API compatibility with ICNN. Just calls forward."""
        return self.forward(x, t)
