# Additional Experiments for "Machine-Learned Sampling of Conditioned Path Measures"

## Setup

```bash
pip install torch numpy matplotlib scipy
```

## Quick Start

### Main experiment runner

```bash
# Run all experiments 
python run_experiments.py --experiment all

# Run individual experiments
python run_experiments.py --experiment brownian_bridge_metrics  # Quantitative metrics
python run_experiments.py --experiment ablation                 # Alg 1 ablations
python run_experiments.py --experiment icnn_vs_mlp              # ICNN vs MLP
python run_experiments.py --experiment scaling_ou               # OU scaling 
python run_experiments.py --experiment scaling_dw               # DW scaling (delegates to run_dw_d12.py)
python run_experiments.py --experiment h_sweep                  # Algorithm 3 h-sweep (dense extreme zig-zag)
python run_experiments.py --experiment costs                    # Computational cost table
python run_experiments.py --experiment fi_scaling               # Improved Table 1 (Fisher information)
```

### Standalone scripts

```bash
# Double-well TPS per dimension — run individual dimensions
python run_dw_d12.py 1          # d=1
python run_dw_d12.py 2          # d=2
python run_dw_d12.py 5          # d=5
python run_dw_d12.py 10         # d=10
python run_dw_d12.py 1 2 5 10   # all dimensions sequentially

# Algorithm 1 vs Algorithm 3 across likelihood regimes 
python run_alg_comparison.py

# Baseline comparison on 2D double-well TPS
python run_baseline_2d_dw.py

# Müller-Brown 2D potential head-to-head (Alg 1 vs Alg 3 vs SPDE)
python run_muller_brown.py
```

## File Structure

### Core modules
- `networks.py` — DriftNetwork (Alg 1), ICNN + PushforwardMLP (Alg 3)
- `sde_utils.py` — SDE integrators, reference processes (BM, OU, double-well)
- `metrics.py` — MMD², sliced W₂, marginal KL, Fisher information estimator
- `algorithm1.py` — Controlled Transport (Section 3.1, Algorithm 1)
- `algorithm3.py` — Wasserstein/JKO with pushforward maps (Section 4.3, Algorithm 3)

### Experiment scripts
- `run_experiments.py` — Main runner for Experiments 
- `run_dw_d12.py` — Stabilized double-well TPS (d=1,2,5,10) with per-dimension configs
- `run_alg_comparison.py` — Alg 1 vs Alg 3 across 4 likelihood regimes (OU)
- `run_baseline_2d_dw.py` — Alg 1 (±SPDE) vs IPF on 2D double-well
- `run_muller_brown.py` — Alg 1 vs Alg 3 vs SPDE on Müller-Brown 2D potential


## Outputs

Results are saved to `results/` as JSON files and PDF figures.
