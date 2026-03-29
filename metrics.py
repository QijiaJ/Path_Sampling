"""
Quantitative evaluation metrics for path measure sampling.
Implements MMD, sliced Wasserstein-2, marginal KL divergence,
and relative Fisher information error.
"""

import torch
import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist


def gaussian_kernel(x, y, sigma=1.0):
    """RBF kernel between two sets of samples."""
    # x: (n, d), y: (m, d)
    xx = (x * x).sum(dim=1, keepdim=True)  # (n, 1)
    yy = (y * y).sum(dim=1, keepdim=True)  # (m, 1)
    xy = x @ y.t()  # (n, m)
    dists = xx - 2 * xy + yy.t()  # (n, m)
    return torch.exp(-dists / (2 * sigma ** 2))


def mmd_squared(samples_p, samples_q, sigma=1.0):
    """
    Compute MMD^2 between two sets of samples using Gaussian kernel.
    samples_p: (n, d), samples_q: (m, d)
    Returns scalar MMD^2.
    """
    Kpp = gaussian_kernel(samples_p, samples_p, sigma)
    Kqq = gaussian_kernel(samples_q, samples_q, sigma)
    Kpq = gaussian_kernel(samples_p, samples_q, sigma)
    n = samples_p.shape[0]
    m = samples_q.shape[0]
    # Unbiased estimator
    mmd2 = (Kpp.sum() - Kpp.diag().sum()) / (n * (n - 1)) \
         + (Kqq.sum() - Kqq.diag().sum()) / (m * (m - 1)) \
         - 2 * Kpq.mean()
    return mmd2


def mmd_multi_sigma(samples_p, samples_q, sigmas=None):
    """MMD with multiple kernel bandwidths (median heuristic + multiples)."""
    if sigmas is None:
        # Median heuristic
        with torch.no_grad():
            dists = torch.cdist(samples_p, samples_q)
            median = dists.median()
        sigmas = [median * f for f in [0.5, 1.0, 2.0]]
    return sum(mmd_squared(samples_p, samples_q, s) for s in sigmas) / len(sigmas)


def sliced_wasserstein_2(samples_p, samples_q, n_projections=100):
    """
    Sliced Wasserstein-2 distance between two sets of samples.
    Much cheaper than full W2 in higher dimensions.
    """
    d = samples_p.shape[1]
    # Random projections on unit sphere
    projections = torch.randn(n_projections, d, device=samples_p.device)
    projections = projections / projections.norm(dim=1, keepdim=True)

    # Project samples
    proj_p = samples_p @ projections.t()  # (n, n_projections)
    proj_q = samples_q @ projections.t()  # (m, n_projections)

    # Sort along sample dimension
    proj_p_sorted = proj_p.sort(dim=0).values
    proj_q_sorted = proj_q.sort(dim=0).values

    # If different sizes, interpolate
    n, m = proj_p_sorted.shape[0], proj_q_sorted.shape[0]
    if n != m:
        # Subsample the larger one
        if n > m:
            idx = torch.linspace(0, n - 1, m).long()
            proj_p_sorted = proj_p_sorted[idx]
        else:
            idx = torch.linspace(0, m - 1, n).long()
            proj_q_sorted = proj_q_sorted[idx]

    # W2 in 1D is just sorted difference
    sw2 = ((proj_p_sorted - proj_q_sorted) ** 2).mean()
    return sw2


def marginal_kl_kde(samples_p, samples_q, d=1):
    """
    Estimate KL(P || Q) using KDE for 1D marginals.
    For higher d, computes average over coordinate-wise 1D KLs.
    samples_p, samples_q: numpy arrays (n, d)
    """
    if isinstance(samples_p, torch.Tensor):
        samples_p = samples_p.detach().cpu().numpy()
    if isinstance(samples_q, torch.Tensor):
        samples_q = samples_q.detach().cpu().numpy()

    if samples_p.ndim == 1:
        samples_p = samples_p[:, None]
    if samples_q.ndim == 1:
        samples_q = samples_q[:, None]

    d = samples_p.shape[1]
    kl_total = 0.0

    for dim in range(d):
        p_data = samples_p[:, dim]
        q_data = samples_q[:, dim]

        try:
            kde_p = gaussian_kde(p_data)
            kde_q = gaussian_kde(q_data)

            # Evaluate on a grid
            x_min = min(p_data.min(), q_data.min()) - 1
            x_max = max(p_data.max(), q_data.max()) + 1
            x_grid = np.linspace(x_min, x_max, 1000)

            p_vals = kde_p(x_grid) + 1e-10
            q_vals = kde_q(x_grid) + 1e-10

            # Normalize
            dx = x_grid[1] - x_grid[0]
            p_vals = p_vals / (p_vals.sum() * dx)
            q_vals = q_vals / (q_vals.sum() * dx)

            kl = (p_vals * np.log(p_vals / q_vals) * dx).sum()
            kl_total += max(0, kl)  # Clip numerical negatives
        except Exception:
            kl_total += float('nan')

    return kl_total / d


def fisher_information_true_gaussian(Sigma_inv):
    """True Fisher information for N(mu, Sigma): tr(Sigma^{-1})."""
    if isinstance(Sigma_inv, torch.Tensor):
        return Sigma_inv.diag().sum().item()
    return np.trace(Sigma_inv)


def fisher_information_estimator(samples, sigma_bw=None, m_perturbations=50):
    """
    Kernel-based Fisher information estimator using leave-one-out KDE score.
    Uses a Gaussian KDE to estimate the score function at each sample point,
    then computes FI = (1/n) sum_i ||nabla log q_hat_{-i}(x_i)||^2.

    Bandwidth is chosen adaptively via Scott's rule if not provided:
      sigma_bw = n^{-1/(d+4)} * std(samples) per coordinate.

    For q_hat(x) = (1/n) sum_j K_H(x - x_j), the score is:
      nabla log q_hat(x) = [sum_j K_H(x-x_j) * H^{-1}(x_j - x)] / [sum_j K_H(x-x_j)]

    Uses leave-one-out to avoid self-score bias.
    m_perturbations is unused (kept for API compat).
    """
    n, d = samples.shape

    # Adaptive bandwidth per dimension (Scott's rule)
    if sigma_bw is None or sigma_bw <= 0:
        stds = samples.std(dim=0)  # (d,)
        scott_factor = n ** (-1.0 / (d + 4))
        bw = stds * scott_factor  # (d,)
    else:
        # Use provided sigma_bw, scaled by data std for each dim
        stds = samples.std(dim=0)
        scott_factor = n ** (-1.0 / (d + 4))
        bw = stds * scott_factor

    bw = bw.clamp(min=1e-6)  # avoid division by zero

    # Standardize samples by bandwidth
    scaled = samples / bw.unsqueeze(0)  # (n, d)

    # Pairwise squared distances in scaled space
    # diffs_scaled[i,j,:] = scaled[j] - scaled[i]
    diffs_scaled = scaled.unsqueeze(0) - scaled.unsqueeze(1)  # (n, n, d)
    sq_dists = (diffs_scaled ** 2).sum(dim=2)  # (n, n)

    # Gaussian kernel in product form
    K = torch.exp(-0.5 * sq_dists)  # (n, n)

    # Leave-one-out: zero out self-contributions
    mask = 1.0 - torch.eye(n, device=samples.device)
    K = K * mask

    # Score numerator: sum_j K[i,j] * (x_j - x_i) / bw^2
    diffs_orig = samples.unsqueeze(0) - samples.unsqueeze(1)  # (n,n,d) : [j]-[i]
    bw_sq = (bw ** 2).unsqueeze(0).unsqueeze(0)  # (1,1,d)
    weighted_diffs = K.unsqueeze(2) * diffs_orig / bw_sq  # (n,n,d)
    score_num = weighted_diffs.sum(dim=1)  # (n, d)

    # Score denominator
    score_den = K.sum(dim=1, keepdim=True).clamp(min=1e-30)  # (n, 1)

    scores = score_num / score_den  # (n, d)

    # FI = (1/n) sum_i ||score(x_i)||^2
    fi_est = (scores ** 2).sum(dim=1).mean()
    return fi_est


def compute_all_metrics(samples_learned, samples_true, sigmas_mmd=None):
    """
    Compute all metrics between learned and true samples.
    Returns dict with MMD^2, SW2, marginal KL.
    """
    results = {}
    results['mmd2'] = mmd_multi_sigma(samples_learned, samples_true, sigmas_mmd).item()
    results['sw2'] = sliced_wasserstein_2(samples_learned, samples_true).item()
    results['marginal_kl'] = marginal_kl_kde(samples_learned, samples_true)
    return results
