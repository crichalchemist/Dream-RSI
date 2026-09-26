## Mechanism
We propose an Orthant-Wise Direct Active Set Method (also known as a Subspace Optimization or Lasso Homotopy step). Instead of relying on iterative cyclic Coordinate Descent to solve the unconstrained lasso subproblem on the active set, we formulate it as an exact quadratic system by assuming the signs of the active coefficients match the gradient directions. This exact linear system is solved directly using Cholesky factorization (`Eigen::LDLT`) in $O(|A|^3)$ time. If the proposed step causes any active variable to cross zero, we perform a line search to the first zero-crossing, drop the violating variable from the active set, and re-solve. 

To maintain efficiency, we wrap this exact active-set solver in a Strong Rule screening loop:
1. Use strong rules to establish a small `screened` candidate set.
2. Form the Gram matrix $G_A$ explicitly for the active variables and solve directly.
3. KKT checks during active set growth are restricted entirely to the `screened` set, achieving $O(n \cdot |screened|)$ gradient updates rather than $O(np)$. 
4. A full $O(np)$ global KKT check is only performed once the `screened` subproblem has reached a guaranteed mathematical optimum.

## Evidence from history
This is the first evaluation in the iteration tree (`iteration 0`).

## Why it's not a repeat
The baseline algorithm is a direct C++ port of `glmnet`, which heavily optimizes a pure Coordinate Descent (CD) architecture. This proposal discards CD entirely in favor of an exact direct-solve (Newton-like) Active Set approach. The mathematical algorithm stepping through active-set combinations via exact linear solves (LARS-style homotopy) is structurally disjoint from cyclic coordinate-wise shrinkage.

## Expected benefit/risk
**Benefit:** For datasets with high collinearity/correlation, Coordinate Descent is notoriously slow because the variables zigzag, taking thousands of passes to converge. A direct linear solve evaluates the exact minimum of the active subspace instantaneously, bypassing collinearity slowdowns.
**Risk:** If the true active set $|A|$ grows extremely large (e.g., highly dense signals with $n, p > 5000$), the $O(|A|^3)$ cost of `LDLT` factorizations and resolving upon zero-crossings can exceed the $O(|A|^2)$ per-epoch cost of Coordinate Descent.
