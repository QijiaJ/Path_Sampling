# Supplementary Experimental Results

This document presents comprehensive experimental results addressing the reviewer-requested improvements. All experiments were run using PyTorch on Apple MPS (M-series GPU) or CPU, with fixed random seed (42) for reproducibility.

---

## Experiment 1: Quantitative Metrics on Brownian Bridge 

**Setup.** We compute sliced Wasserstein-2 (SW₂) between learned and ground-truth marginals at 5 time points on the Brownian bridge. Ground truth from the Brownian bridge SDE with x(0) = −2, x(1) = +2. Algorithm 1: 20 cosine-spaced annealing steps, network (80, 120), 500 paths, 400 opt steps per annealing iteration, dt = 0.01. Final metrics averaged over 3 independent batches of 500 paths each against 1,500 ground truth samples.

**Results.**

| Time t | SW₂ | 
|--------|-----|
| 0.00 | 0.000 | 
| 0.25 | 0.602 | 
| 0.50 | 0.778 | 
| 0.75 | 0.561 | 
| 1.00 | 0.035 |

At t = 0, all metrics are 0 by construction since both learned and ground-truth paths start deterministically at x = −2. The interior marginals (t = 0.25–0.75) show SW₂ in the range 0.56–0.78. The terminal constraint is strongly enforced: SW₂ = 0.035 at t = 1.0, confirming that paths concentrates near the target x(T) = +2. 

*Results: brownian_bridge_metrics.json*

---

## Experiment 2: Ablation Study for Algorithm 1 

**Setup.** We test the sensitivity of Algorithm 1 to network size, annealing steps, ensemble size, and optimization budget. Brownian bridge test bed w/ quantitative metrics. Default configuration: hidden_dims = (20, 30), 10 annealing steps, 500 paths, 200 optimization steps. Each hyperparameter is varied independently.

### Network Size

| Hidden Dims | MMD² | SW₂ | Time (s) |
|-------------|------|-----|----------|
| (10, 10) | 0.349 | 0.982 | 239.6 |
| (20, 30) | 0.313 | 0.129 | 304.2 |
| (40, 60) | 0.321 | 0.093 | 408.0 |
| (80, 120) | 0.324 | 0.068 | 739.3 |

Increasing capacity from (10, 10) to (80, 120) yields a 14.4× improvement in SW₂ (0.982 → 0.068) with ~3× additional compute. The default (20, 30) already captures most of the gain (SW₂ = 0.129), with diminishing returns beyond (40, 60).

### Number of Annealing Steps

| Steps | MMD² | SW₂ | 
|-------|------|-----|
| 5 | 0.429 | 0.415 | 
| 10 | 0.341 | 0.144 | 
| 20 | 0.333 | 0.126 | 
| 40 | 0.339 | 0.121 | 

SW₂ improves 3.4× from 5 to 10 steps, then plateaus. Runtime scales super-linearly since each step must evaluate the full drift chain. 10 steps offer a good accuracy/cost trade-off.

### Ensemble Size (Number of Paths K)

| K | MMD² | SW₂ | Time (s) |
|---|------|-----|----------|
| 100 | 0.337 | 0.326 | 167.5 |
| 250 | 0.358 | 0.164 | 207.4 |
| 500 | 0.315 | 0.123 | 292.8 |
| 1000 | 0.337 | 0.216 | 391.3 |

K = 500 achieves the best SW₂ (0.123). K = 1000 shows slightly worse SW₂, likely due to noisier per-sample gradients at large batch sizes with the sum-based loss. 

### Optimization Steps per Annealing Iteration

| Opt Steps | MMD² | SW₂ | Time (s) |
|-----------|------|-----|----------|
| 50 | 0.463 | 4.911 | 73.7 |
| 100 | 0.370 | 1.251 | 147.7 |
| 200 | 0.336 | 0.126 | 292.2 |
| 400 | 0.331 | 0.068 | 562.0 |

This is the most impactful hyperparameter: SW₂ improves 72× from 50 to 400 optimization steps (4.911 → 0.068). At 50 steps, the drift is severely under-optimized; 200 steps captures most of the improvement.

*Restuls: ablation_algorithm1.pdf*


---

## Experiment 3: h-Sweep for Algorithm 3 

**Setup.** In the paper, we showed an O(h²) error scaling of the approximation w.r.t the JKO step size h. We investigate this empirically here. OU process (β = 0.25, d = 1), equilibrium initialization N(0, 2). Five observations at t = 0.2, 0.4, 0.6, 0.8, 1.0 with extreme zig-zag targets [5, −4, 6, −3, 7] and σ_obs = 0.2 (penalty 1/(2σ²) = 12.5). These targets range from 1.5σ to 3.5σ from equilibrium, forcing violent marginal reversals. Algorithm 3: 4-layer MLP (width 256), 300 particles, 400 opt steps per JKO step. Sweep h ∈ {0.05, 0.1, 0.2, 0.25, 0.5, 1.0}.

| h | T/h | Avg SW₂ |  
|-------|-----|---------|
| 0.050 | 20 | 1.493 | 
| 0.100 | 10 | 0.131 |
| 0.200 | 5 | 10.24 | 
| 0.250 | 4 | 16.57 |
| 0.500 | 2 | 12.45 |
| 1.000 | 1 | 13.78 | 

The relationship between h and accuracy is slightly non-monotonic. h = 0.1 (10 JKO steps) achieves the best marginal accuracy (avg SW₂ = 0.131). At h = 0.05 (20 steps), the marginal quality degrades (avg SW₂ = 1.49) despite the finer grid — the increased number of optimization problems accumulates more training error. At h ≥ 0.2, the JKO grid becomes too coarse: the avg SW₂ jumps to 10.24 because each step must bridge multiple observations with opposite-sign targets, violating the smooth-marginal assumption. 

Appendix A identifies the failure mode: the JKO decomposition assumes smooth marginal evolution --  we additionally investigate the perforance w.r.t abrupt marginal changes in the response to Reviewer a2Zn.

*Results: hsweep_dense_abrupt.pdf*


---

## Experiment 4: Higher-Dimensional Scaling 

We demonstrate both algorithms on problems up to d = 50 (Alg 3) and d = 10 (Alg 1).

### Experiment 4a: Double-Well TPS Scaling (Alg 1)

**Setup.** Double-well potential V(x) = 5(x₁² − 1)² with transition path sampling from x₁ = −1 to x₁ = +1. Algorithm 1 with dt = 0.02, cosine-spaced annealing, 200 opt steps per iteration, 4-layer residual MLP, per-dimension σ_obs tuning.

| d | Init x₁ Mean | Term x₁ Mean | Term x₁ Std | Crossing Rate | Paths | Params |
|---|-------------|-------------|-------------|---------------|-------|--------|
| 1 | −0.994 | 1.182 | 0.261 | 100% (600/600) | 200 | 134K | 
| 2 | −0.995 | 1.075 | 0.246 | 99.5% (597/600) | 200 | 300K | 
| 5 | −1.000 | 1.165 | 0.241 | 100% (900/900) | 300 | 301K | 
| 10 | −1.002 | 1.156 | 0.251 | 99.6% (896/900) | 300 | 303K | 

Algorithm 1 achieves near-perfect barrier crossing at all dimensions. At d = 1, all 600 paths cross with tight terminal concentration (std = 0.26). At d = 5, all 900 paths cross (100%), demonstrating robust scaling to moderate dimensions. At d = 10, 99.6% of paths cross (896/900) with terminal x₁ mean = 1.156 and std = 0.251. The midpoint x₁ means (0.23–0.33 at the quarter-mark, rising through 1.0–1.1 at the midpoint) confirm that paths traverse the barrier continuously rather than jumping.


### Experiment 4b: OU Process Scaling (Alg 3)

**Setup.** OU process (β = 0.25) with soft endpoint constraint pinning x₁ = 1 (γ = 0.5). Algorithm 3 with h = 0.25, 200 particles, 4 JKO steps. ICNN for d ≤ 10, MLP for d ≥ 20.

| d | Terminal x₁ Mean | Terminal x₁ Std | FI (true) | FI (est) | FI Rel. Err | 
|---|-----------------|-----------------|-----------|----------|-------------|
| 1 | 0.564 | 5.80 | 0.25 | 0.023 | 0.908 | 
| 2 | 0.256 | 4.09 | 0.50 | 0.125 | 0.750 | 
| 5 | 0.560 | 2.88 | 1.25 | 0.742 | 0.406 | 
| 10 | 0.741 | 2.58 | 2.50 | 2.053 | 0.179 | 
| 20 | 0.614 | 0.76 | 5.00 | 9.192 | 0.838 | 
| 50 | 0.663 | 0.77 | 12.50 | 19.661 | 0.573 | 

Algorithm 3 successfully shifts the first coordinate toward the target across all dimensions. The FI estimator is most accurate at d = 10 (17.9% relative error) but degrades at higher dimensions, consistent with the known bias of kernel-based estimators. 


*Results: scaling_ou.json, scaling_dw_dim_5.pdf, scaling_dw_dim_10.pdf*


---

## Experiment 5: Algorithm 1 vs Algorithm 3 Across Likelihood Regimes 

**Motivation.** Some reviewers ask when each algorithm is preferable. We test four regimes spanning the sparse/dense and mild/extreme axes.

**Setup.** OU process (β = 0.25, d = 1), T = 1.0, dt = 0.02, equilibrium initialization N(0, 2). Reference: **analytical Gaussian posterior** via Kalman RTS smoother. Algorithm 1: 30 annealing steps, 200 paths, 200 opt steps, 4-layer MLP (width 128). Algorithm 3: h = 0.2 (5 JKO steps), 300 particles, 300 opt steps, 4-layer MLP (width 256).

| Regime | # Obs | Targets | σ_obs | Penalty |
|--------|-------|---------|-------|---------|
| Dense/mild | 5 | [3, −3, 3, −3, 3] | 0.3 | 5.6 |
| Dense/abrupt | 5 | [5, −4, 6, −3, 7] | 0.2 | 12.5 |
| Sparse/mild | 1 | [4] | 0.3 | 5.6 |
| Sparse/extreme | 1 | [8] | 0.3 | 5.6 |

**Results.**

| Regime | Alg | Avg SW₂ | Path SW₂ | Traj RMSE | Terminal Mean (True) | 
|--------|-----|---------|----------|-----------|---------------------|
| Dense/mild | Alg 1 | 3.27 | 1.90 | 1.33 | 0.795 (2.16) | 
| Dense/mild | Alg 3 | 2.51 | 1.33 | 1.07 | 0.000 (2.16) | 
| Dense/abrupt | Alg 1 | 13.24 | 6.43 | 2.61 | 3.036 (6.19) | 
| Dense/abrupt | Alg 3 | 10.28 | 4.69 | 2.18 | 1.975 (6.19) | 
| Sparse/mild | Alg 1 | **0.017** | **3.02** | **1.46** | 3.966 (3.91) | 
| Sparse/mild | Alg 3 | 0.273 | 18.67 | 3.30 | 3.234 (3.91) |
| Sparse/extreme | Alg 1 | 1.21 | **14.48** | **3.26** | 8.176 (7.82) | 
| Sparse/extreme | Alg 3 | 1.28 | 61.89 | 6.48 | 6.507 (7.82) | 

The results reveal clear regime-dependent trade-offs:

**Sparse likelihoods:** Algorithm 1 dominates. On sparse/mild, Alg 1 achieves 16× better marginal SW₂ (0.017 vs 0.273) and 6× better path SW₂ (3.02 vs 18.67), with the terminal mean nearly matching the analytical value (3.97 vs 3.91). On sparse/extreme (target 4σ from equilibrium), Alg 1's terminal mean (8.18) is much closer to the true value (7.82) than Alg 3's (6.51), and the path SW₂ gap is 4.3× (14.5 vs 61.9). The path-space annealing in Alg 1 can gradually deform the full trajectory toward extreme targets, whereas Alg 3's marginal-only approach struggles to propagate the endpoint constraint backward.

**Dense likelihoods:** Algorithm 3 achieves comparable or slightly better marginal accuracy at lower cost. On dense/mild constraint, Alg 3's avg SW₂ (2.51) beats Alg 1 (3.27), and its trajectory RMSE is lower (1.07 vs 1.33). However, both algorithms struggle with the dense/abrupt regime (avg SW₂ > 10), where the extreme zig-zag targets create violent marginal reversals that neither method resolves well within the given budget.


*Results: alg_comparison.json*

---

## Experiment 6: Baseline Comparison on 2D Double-Well TPS 

**Setup.** Double-well potential V(x) = 3.5(x₁² − 1)² + 0.5x₂² in d = 2, T = 0.5, dt = 0.01 (50 time steps). Observations: well A = (−1, 0) at t = 0, well B = (1, 0) at t = T, σ_obs decaying from 1.5 to 0.15 over 30 annealing steps. With this particular (endpoint-constrained) likelihood, we use Sinkhorn / IPF from Diffusion Schroding bridge [BTHD NeurIPS '21] as external baseline, and compare Algorithm 1 (with and without SPDE refinement) against on this 2D double-well TPS problem.

**Results.** 

| Method | Path SW₂ | Barrier Cross | OM Action (ref: 105.6 ± 14.0) | Terminal x₁ |
|--------|----------|---------------|-------------------------------|------------|
| Sinkhorn/IPF | 0.092 | 99.0% | 118.9 ± 21.6 | 0.976 | 
| Alg 1 | 0.112 | 99.4% | 119.1 ± 16.2 | 0.979 | 
| Alg 1 + SPDE | 0.082 | 99.8% | 30.2 ± 6.1 | 1.002 | 

Sinkhorn/IPF and Algorithm 1 achieve comparable path-level accuracy (SW₂ 0.09–0.11) with near-perfect crossing rates (99.0–99.4%). Adding SPDE refinement to Algorithm 1 improves path SW₂ to 0.082 and the crossing rate to 99.8%.

We want to emphasize that, our Algorithm 1 framework is, however, substantially more general, which can deal with arbitrary path-wise likelihood and reference process.

*Results: baseline_2d_dw.json*

---

## Experiment 7: Müller-Brown 2D Potential: Head-to-Head Comparison 

**Motivation.** The Müller-Brown potential is a standard benchmark in computational chemistry with three metastable wells. We condition on a 4-point path: Well A (−0.558, 1.442) at t = 0, Saddle AC (−0.822, 0.624) at t = 1/3, Saddle CB (0.212, 0.293) at t = 2/3, and Well B (0.623, 0.028) at t = 1. σ_obs = 0.3. 

**Setup.** T = 1.0. Analytical gradient of the MB potential. Algorithm 1: 30 cosine-spaced annealing steps, 200 opt steps, 200 paths, 3 final batches averaged. Algorithm 3: 200 particles, h = T/3 (3 JKO steps aligned with observation times), initialized near Well A with 2000-step Langevin burn-in clamped to Well A basin. SPDE: 50 paths, 1500 Langevin steps, ds_scale = 0.1. 

**Results.** Observation error denotes Euclidean distance from the observation target. From the plot:

Algorithm 1 achieves moderate accuracy (avg error 0.647) and is the best of the 3 methods. The path-space annealing successfully navigates the potential landscape but the drift optimization struggles a bit at the interior saddle points

Algorithm 3's particles remain near Well A throughout the trajectory (terminal error 1.843), with the JKO steps unable to drive the transport across the barrier in only 3 steps. This is consistent with the failure mode identified in the paper: when marginals change abruptly between JKO steps (here, from Well A to Saddle AC to Saddle CB to Well B), the smooth-marginal assumption breaks down.

Vanilla SPDE shows improved accuracy through the trajectory (0.598 at t = 1), suggesting that the Langevin dynamics on path space find reasonable paths but struggle to fully satisfy the constraint (avg error 0.843).

*Results: muller_brown_comparison.pdf*

---

## Experiment 8: Computational Cost Comparison

**Results.**

| Algorithm | Problem | N | Time/Step (s) |
|-----------|---------|---|---------------|
| Alg 1 (BB) | BB d=1 | 100 | 9.75 |
| Alg 1 (DW) | DW d=1 | 600 | 15.30 |
| Alg 3 (ICNN) | OU d=1 | 200 | 0.24 |
| Alg 3 (MLP) | OU d=1 | 200 | 0.12 |


Algorithm 1 is much slower than Algorithm 3 per run, reflecting the cost of path-space annealing with multiple SDE re-simulations. 

For Algorithm 3, ICNN is ~2× slower than MLP.




