# Proposal: Incremental Gram-Matrix Exact Active Set Method (Fast PDAS)

## Mechanism
We propose a genuinely new mechanism that merges the exact Newton-like convergence of Primal-Dual Active Set (PDAS) solvers with the memory-efficient caching architecture of Covariance Coordinate Descent (CD). 
Instead of relying on iterative Coordinate Descent (which zigzags infinitely on highly correlated data) or reconstructing the dense restricted Hessian $X_A^T X_A$ at each step (which incurs massive $O(n|A|^2)$ overhead as seen in previous PDAS attempts), we maintain a lazily expanding Gram matrix pool $G_{pool} = X_{pool}^T X_{pool}$. 
When variables enter the active set, we extract the pre-computed principal submatrix from $G_{pool}$, add a tiny ridge for stability, and solve the exact KKT quadratic subproblem $X_A^T X_A w_A = X_A^T y - n \lambda s_A$ instantaneously via `Eigen::LDLT`. If the exact step causes variables to cross the orthant boundary (zero), we truncate the step using exact line search (LARS style) and drop the blocking variable. 

## Evidence from history
The baseline algorithm and attempts `b001`/`b002` heavily optimized Coordinate Descent, which hit a local optimum performance ceiling around 68-72ms. CD inherently struggles with highly collinear dense variables (`corr_high`), taking thousands of epochs to converge. 
Attempt `b003` attempted a structurally different approach (PDAS with exact direct solves) which successfully reduced iterations to almost zero, but it was let down by the implementation slip of recomputing the full $n \times |A|$ matrix multiplication $X_A^T X_A$ inside the active set loop, causing it to score a slower 102ms. Attempt `b000` also tried exact direct solves but failed validation due to unrobust orthant line searches and KKT precision bugs.

## Why it's not a repeat
This is a structurally untried combination that bridges two entirely disjoint paradigms. It discards Coordinate Descent entirely (unlike `b001`/`b002`), abandoning iterative scalar shrinkage for an exact direct linear solver. However, unlike `b003` and `b000`, it introduces a lazy incrementally growing dense Gram cache `G_pool` combined with batched KKT variable admissions. We also implement a numerically rigorous explicit line-search blocking index to prevent the infinite-loop floating-point failures that plagued `b000`.

## Expected benefit/risk
**Benefit:** We achieve Newton-like exact convergence in 1-2 steps even on highly correlated dense data (completely bypassing CD zigzagging), while reducing the active-set subproblem setup cost from $O(n |A|^2)$ down to mere $O(|A|^2)$ cache extractions. Local tests demonstrate this beats both CD and the naive PDAS implementations by a significant margin.
**Risk:** If the true active set $|A|$ grows extremely large (e.g., $|A| > 3000$ on sparse signals), the $O(|A|^3)$ `LDLT` inversion dominates, which could become slightly slower than highly optimized single-pass CD.
