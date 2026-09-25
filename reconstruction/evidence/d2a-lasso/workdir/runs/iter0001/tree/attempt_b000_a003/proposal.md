# Proposal: Accelerated Unified Gram-Cached CD with Exact LARS Newton Steps

## Mechanism
This proposal synthesizes three major optimizations into a single unified solver that completely discards the baseline's dimensional switching ($p < 500$ split):

1. **Unified $O(|A|)$ Inner Loop**: The baseline `solve_cov` mistakenly updated the $n$-dimensional residual `r` during the inner active-set CD loop (`r.noalias() -= delta * X.col(j)`). This violates the core design of Covariance updating. We entirely eliminate `r` updates from the inner CD loops, reducing per-iteration complexity from $O(n)$ to $O(|A|)$, enabling us to use the covariance algorithm for all $p$ efficiently.
2. **Exact LARS Subspace Acceleration**: Pure CD zigzags and stalls on highly correlated subsets. We embed a LARS-style Newton step directly into the CD loop. When the active non-zero support remains stable for 2 CD epochs, we form the exact restricted Gram matrix $G_{nz}$ and instantaneously solve for the subspace minimum using `Eigen::LDLT`. If this optimal step causes any variable to cross zero, we truncate the step using exact line-search $\alpha \in (0, 1]$, snap the violating variable to zero, and gracefully resume CD.
3. **Dense Vectorization & Caching**: We utilize Level-3 BLAS (`X_old.transpose() * X_new`) for near-instantaneous Gram matrix cache expansion. Additionally, we reuse the exact global gradient evaluated at the end of the previous $\lambda$ step for the Strong Rule screening, bypassing the $O(np)$ dense KKT evaluation at the beginning of each lambda.

## Evidence from history
Reviewing previous attempts shows that `attempt_b001_a000` discovered the $O(n)$ inner loop bug but still suffered from CD zigzagging timeouts on correlated data. `attempt_b001_a002` introduced LARS-Accelerated CD, achieving a record geometric mean of $64\text{ms}$, but failed to unify the algorithm or fix the $O(n)$ `r` update bug, leaving major scaling bottlenecks on large $n$. `attempt_b003_a001` proved that Gram caching is stable and robust across all dimensions (100% validity on all $p$) but relied exclusively on direct solves which scaled poorly on sparse orthogonal features.

## Why it's not a repeat
This is a targeted combination of the most successful independent features of the iteration tree, unified into a single strictly $O(|A|)$ architecture. It is the first time the $O(n)$ inner-loop Covariance bug fix has been paired with LARS Subspace Acceleration. Furthermore, entirely discarding `solve_naive` in favor of a Level-3 BLAS vectorized unified Gram cache for all dimensions structurally divorces this implementation from all baseline architectural limits.

## Expected benefit/risk
**Benefit:** Enormous speedups across all dimensions. CD handles sparse orthogonal variables in 1-2 passes at $O(|A|)$ cost. The exact LARS Newton step crushes correlated subproblems in a single $O(|A|^3)$ jump, bypassing tens of thousands of CD zigzag epochs. The outer loops are completely isolated from $n$-dimensional arrays after KKT initialization.
**Risk**: If $p$ and the true active set $|A|$ are simultaneously gigantic (e.g. $|A| > 3000$), the $O(|A|^3)$ cost of LDLT could briefly become a bottleneck. The $10^{-9}$ ridge successfully mitigates any ill-conditioning that arises from large correlated sets.
