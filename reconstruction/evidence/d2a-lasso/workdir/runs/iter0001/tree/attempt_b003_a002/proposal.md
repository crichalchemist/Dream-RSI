# Proposal: Dual-Engine Subspace Acceleration (DESA) with Exact Line Search

## Mechanism
We propose a Dual-Engine algorithm that aggressively accelerates Coordinate Descent (CD) via Exact Subspace Newton steps, tailored optimally to the problem dimensions to avoid scaling bottlenecks. 
Instead of forcing a single active-set solver on all dimensions (which caused previous exact methods to stall on large $p$ or CD to stall on high correlation), we split the logic:
1. **Low-Dimension ($p < 500$):** We maintain the $O(|A|^2)$ Gram matrix cache. When the non-zero support stabilizes, we execute an exact direct `Eigen::LDLT` solve on the restricted block.
2. **High-Dimension ($p \ge 500$):** Gram caches become memory/compute bottlenecks ($O(n|A|^2)$). We instead use a Hessian-free **Preconditioned Conjugate Gradient (PCG)** subspace acceleration. PCG solves the restricted Newton system using only $O(n|A|)$ matrix-vector multiplications, bypassing dense inversions completely.
Crucially, for both engines, we replace the naive sign-rejection mechanism of past attempts with a rigorous **Exact Orthant Line Search**. If a Newton step crosses zero, we take the maximal step $\alpha \in (0, 1]$ to the orthant boundary, snap the violating variable to exactly 0.0, and resume CD. This guarantees monotonic objective decrease along the Newton direction and completely eliminates step rejections and infinite loops.

## Evidence from history
- `attempt_b000_a001` (CD-Newton Hybrid) successfully hit 123ms but suffered severely on Problem 12 (`858ms`). By forcing a dense Gram Cache and LDLT inversion on large $p$, it hit an $O(|A|^3)$ bottleneck. Additionally, it completely rejected Newton steps if signs crossed, wasting the computation.
- `attempt_b001_a001` (Exact Active Set) timed out because its exact line search lacked strict precision snapping, causing floating-point infinite loops.
- `attempt_b003_a001` achieved 155ms using a Gram cache for everything, validating exact line search but proving that $O(|A|^3)$ inversion limits overall speed.

## Why it's not a repeat
This combines the fast scaling of Naive/Covariance dimensional splitting with exact second-order optimization. Past attempts either abandoned the $p=500$ threshold entirely, forcing Gram caching on huge datasets, or relied on unified direct solvers that scaled poorly. Furthermore, PCG (Hessian-free) Subspace Acceleration has never been attempted in this codebase; it uniquely bypasses the dense active-set inversions that bottlenecked all previous Newton-based attempts.

## Expected benefit/risk
**Benefit:** Problem 12 and other large $p$ cases will see massive speedups because PCG converges to the Newton step in $O(n|A|)$ without forming the Gram matrix. Highly correlated dense cases (like $p < 500$ `corr_high`) will be instantly crushed by the LDLT exact solver + orthant line search, which avoids the thousands of epochs CD zigzagging requires.
**Risk:** If the high-dimension cases ($p \ge 500$) are poorly conditioned, PCG might take many iterations. We cap PCG iterations to $\min(|A|, 20)$ and use diagonal preconditioning to ensure it always acts as a fast truncated-Newton step rather than stalling.
