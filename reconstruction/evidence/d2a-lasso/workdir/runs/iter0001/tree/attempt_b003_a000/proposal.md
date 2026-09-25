# Proposal: Primal-Dual Active Set (PDAS) with Exact Direct Solves

## Mechanism
We propose a Primal-Dual Active Set (PDAS) algorithm, also related to Semismooth Newton methods, paired with an Active Set QP inner loop that uses an exact direct solver. Instead of performing Coordinate Descent (CD) on the active set, which can suffer from extremely slow convergence in regimes with high feature correlation (a known weakness of GLMNET), this method solves the restricted block Hessian exactly using `Eigen::LDLT`.
At each lambda in the path, it:
1. Employs a Sequential Strong Rule to initialize the active set efficiently.
2. Forms the restricted Gram matrix $X_A^T X_A$ and solves the exact system for the current active orthant.
3. If variables step outside their valid orthant (cross zero), the step size is truncated (similar to LARS) and multiple violating variables can be dropped simultaneously.
4. KKT conditions are evaluated on the inactive set, and batches of top violators are added back to the active set until global optimality is reached.

## Evidence from History
There is no prior history to read for `iter0001` since this is the first iteration. However, analysis of the provided GLMNET baseline reveals that it relies exclusively on Coordinate Descent (both naive updates and Gram-cached `solve_cov` updates). CD performs well for highly sparse or uncorrelated features, but is algorithmically bottlenecked by Gauss-Seidel convergence rates when variables are highly correlated (as in the `corr_high` evaluation tests). 

## Why it's not a repeat
The baseline uses Coordinate Descent. This method explicitly abandons Coordinate Descent for the inner solver in favor of an exact direct solve ($X_A^T X_A w_A = X_A^T y - n \lambda s_A$) using Cholesky decomposition. It leverages LARS-like step truncation and orthant constraints to rigorously enforce L1 non-differentiability without requiring scalar iterations. This fundamentally transforms the active set subproblem from an iterative coordinate-wise optimization to a sequence of direct linear system solves.

## Expected benefit / risk
- **Benefit**: Massive speedups on highly correlated datasets (`corr_high`), dense solution sets, and settings where CD stalls. The exact solver ensures Newton-like convergence (often 1 or 2 steps per lambda). Matrix operations leverage highly vectorized Eigen BLAS capabilities, fully avoiding scalar loops.
- **Risk**: For extremely large active sets (e.g. $|A| > 2000$), $O(|A|^3)$ direct inversion becomes the bottleneck, potentially making it slower than CD. Additionally, adding too many correlated variables at once might cause ill-conditioning in $X_A^T X_A$, which is mitigated here by a small $10^{-9}$ ridge.
