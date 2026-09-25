# Proposal: Gram-Cached Orthant-Wise Limited-memory Quasi-Newton (GC-OWL-QN)

## Mechanism
We propose a fundamentally different inner-loop solver for the Lasso active set: the Orthant-Wise Limited-memory Quasi-Newton (OWL-QN) algorithm, fully integrated with incremental Gram matrix caching. 
Instead of optimizing one coordinate at a time (Coordinate Descent) or factoring the full active block Hessian exactly (PDAS/Newton), OWL-QN uses an L-BFGS approximation of the Hessian to take perfectly scaled, dense Quasi-Newton steps across the entire active set simultaneously. 
1. **L-BFGS Curvature**: We maintain a history (size $m=10$) of gradients to implicitly model the inverse Hessian $H$. 
2. **Orthant Projection**: To handle the L1 non-differentiability without Coordinate Descent, we construct a pseudo-gradient that identifies the correct orthant, compute the L-BFGS direction, and project the line search strictly into this orthant (snapping zero-crossings exactly to zero).
3. **Vectorized Operations**: Because it performs full matrix-vector multiplications ($O(|A|^2)$) rather than scalar loops, it fully utilizes CPU SIMD/AVX instructions.

## Evidence from history
Reviewing past attempts:
- `attempt_b001` (CD with caching) proved $O(1)$ memory updates are crucial but remained bound by CD's sequential zigzagging on highly correlated features.
- `attempt_b002` and `attempt_b003` (PDAS Exact Direct Solves) replaced CD with an exact Cholesky LDLT factorization ($O(|A|^3)$). While robust to collinearity, exact inversion becomes a heavy bottleneck as $|A|$ grows, and orthant tracking causes excessive Cholesky restarts.

## Why it's not a repeat
Neither the baseline nor any past attempt has utilized a Limited-Memory Quasi-Newton (L-BFGS) approach. It sits in the algorithmic "sweet spot" between Coordinate Descent (first-order, slow convergence on correlation) and PDAS Exact Solves (second-order, $O(|A|^3)$ cost). OWL-QN provides near-Newton convergence on correlated features while preserving strictly $O(m |A|)$ per-step complexity, making it a structurally distinct, untried paradigm.

## Expected benefit/risk
**Benefit**: On dense or highly correlated subproblems, OWL-QN converges in $\sim 10-20$ iterations (vastly outperforming CD), and requires NO matrix factorizations (vastly outperforming PDAS LDLT). Coupled with the Gram cache, evaluating the objective and gradient becomes virtually free $O(|A|^2)$, eliminating all outer-loop bottlenecks.
**Risk**: For completely uncorrelated, highly sparse data, pure Coordinate Descent is theoretically unbeatable in its simplicity. The line-search and L-BFGS history maintenance might introduce a small constant overhead compared to CD when the problem is trivially easy.
