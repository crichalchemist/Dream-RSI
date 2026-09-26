# Proposal: Cached Gram-Matrix Primal-Dual Active Set (PDAS)

## Mechanism
This proposal combines the best algorithmic insights from history into a single structurally distinct solver: a Primal-Dual Active Set (PDAS) method using an exact block LDLT direct solver, wrapped inside a Strong Rule screening loop, fully powered by an incrementally expanding active Gram matrix cache.

1. **Exact Block-Hessian Solve**: Instead of Coordinate Descent, we construct an exact Newton-like active set restricted subproblem. We solve `G_A w_A = X_A^T y - n \lambda s_A` using `Eigen::LDLT`.
2. **Gram Matrix Caching**: Crucially, we do not repeatedly compute the active Gram matrix $X_A^T X_A$ (which takes $O(n |A|^2)$) as past direct-solve attempts did. We maintain a dynamically expanding Gram matrix $G$ for the *entire screened set*. Extracting $G_A$ is $O(|A|^2)$ and LDLT is $O(|A|^3)$, making the inner solver's complexity completely independent of $n$. We also use level-3 BLAS (`X_old^T X_new`) to build the cache near-instantaneously.
3. **Screened KKT Loop**: The exact solver operates strictly within a strongly-screened subset of variables. The $O(np)$ global gradient is computed *only* when the screened set reaches guaranteed mathematical optimality.

## Evidence from history
- `attempt_b001_a000` (Coordinate Descent with caching) successfully showed the power of minimizing $O(n)$ work in the inner loops and keeping memory caching, scoring well (`0.01379`).
- `attempt_b003_a000` (Exact PDAS Solve) showed that swapping Coordinate Descent for a direct Cholesky solve was also fast and stable for highly correlated features, but its lack of Gram caching and its usage of $O(np)$ global KKT checks in the inner loop caused it to be slightly slower (`0.00974`).

## Why it's not a repeat
Neither baseline nor past attempts have merged the memory efficiency of Strong Rule Gram Caching with the iteration-free power of an Exact Direct Active-Set solver. This effectively eliminates CD iterations while keeping $O(|A|^3)$ direct-solve costs free of large $n$ factors, breaking the $O(n)$ barriers that bottlenecked prior exact active-set methods. 

## Expected benefit/risk
**Benefit**: Huge speedups. On strongly correlated features, exact direct solves do not zigzag or stall. Because the Gram cache prevents repetitive $O(n |A|^2)$ matrix multiplication and global $O(np)$ KKT checks are skipped until local optimality, time complexity plummets.
**Risk**: If the active set grows massively (e.g. $> 2000$), the $O(|A|^3)$ cost of LDLT inversions could dominate, though typical Lasso limits active sets well below this threshold.
