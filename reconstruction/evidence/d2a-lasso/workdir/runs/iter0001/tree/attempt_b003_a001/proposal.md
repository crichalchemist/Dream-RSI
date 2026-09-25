# Proposal: Gram-Cached Primal-Dual Active Set (GC-PDAS)

## Mechanism
This proposal synthesizes the optimal insights from prior attempts by unifying the **Primal-Dual Active Set (PDAS)** architecture (from `attempt_b003`) with the **$O(|A|)$ incremental Gram-matrix caching** (from `attempt_b001` and `solve_cov`). 

In `attempt_b003`, the exact active-set solve (Cholesky decomposition) was highly successful but recomputed the active restricted matrix $X_A^T X_A$ from scratch every inner iteration, incurring an $O(n |A|^2)$ penalty per step. We completely eliminate this bottleneck by caching and incrementally maintaining the Gram matrix $G_A = X_A^T X_A$. 
- When $k$ new features are added, $G_A$ is expanded via an $O(k n |A|)$ vectorized dot-product loop.
- When features cross zero and are dropped during the line search, they are removed from $G_A$ via an instantaneous $O(|A|^2)$ block shift.
- The KKT subgradient $X_A^T y$ is precomputed upfront as $c_y = X^T y$ in $O(np)$ time, reducing the subproblem right-hand-side vector assembly to $O(|A|)$ pure lookups.
- Strong Rule screening completely avoids recomputing $r$ and $g$ by recycling the exact terminal KKT gradient evaluated at the end of the prior lambda.

## Evidence from history
- `attempt_b003` achieved 100% validity by replacing Coordinate Descent (CD) with an exact direct solver that traces the orthant paths rigorously, successfully neutralizing the collinearity slowdowns CD suffers from. However, its memory and computation profile scaled poorly with $|A|$ due to raw matrix reconstruction inside the inner loop.
- `attempt_b001` proved that incrementally caching gradients and Gram interactions drops evaluation times substantially (achieving a $72\text{ms}$ geometric mean), but the agent timed out.

## Why it's not a repeat
This is the first combination of PDAS (direct solver) with incremental Gram caching. Previous exact methods reconstructed the active submatrix every loop, while previous cached methods relied strictly on iterative scalar Coordinate Descent. Furthermore, caching the full $X^T y$ subgradient upfront to assemble Cholesky RHS vectors in $O(1)$ per feature is mathematically novel for this exact solver.

## Expected benefit/risk
- **Benefit**: The active-set inner loop drops to purely $O(|A|^3)$ complexity (independent of $n$). The line-search truncation logic becomes practically free. This guarantees massive speedups over both CD and naive PDAS for large-sample domains ($n \gg p$).
- **Risk**: Maintaining the Gram cache array shifts logic dynamically via row/col memory moves. A minor off-by-one error during zero-crossing variable drops could corrupt the linear systems. Pre-allocating and moving values avoids dense matrix copies but depends heavily on precise indexing.
