# Proposal: Strong Rule and Inner Loop Optimizations

## Mechanism
This proposal introduces a set of deeply impactful algorithmic optimizations for the CD-based lasso path solver:
1. **$O(|A|)$ Inner Loop for Covariance Updates:** In `solve_cov`, the algorithm correctly utilized Gram caching ($c$ and $G$) for fast active set gradients, but mistakenly maintained the residual $r$ inside the inner CD loops ($r \mathrel{-}= \Delta X_j$), doing $O(n)$ work per iteration! This completely defeated the complexity gains of covariance updates for $p < n$. We completely eliminated $r$ updates from the inner loops, making them purely $O(|A|)$ and re-synchronizing $c$ cleanly via a fresh residual evaluated solely right before the $O(np)$ KKT check.
2. **Dense Vectorization of KKT check:** In `solve_naive`, the KKT check was performing an unvectorized matrix-vector multiplication by iterating $j$ from $0 \dots p-1$ and doing inner products via `X.col(j).dot(r)`. We replaced this with a densely vectorized, BLAS-accelerated $X^T r$ evaluation.
3. **No-Cost Strong Rule (Caching KKT Gradients):** For both methods, Step 1 (the Strong Rule screening) inherently requires computing the gradient vector at the *previous* lambda iteration's solution. Because KKT checks effectively recompute the full gradient exactly at the end of each iteration, we cache this densely-vectorized `grad` from the end of the previous iteration and directly access `grad(j)` for the Strong Rule and for adding features into the active-set covariance buffer. This saves $O(np)$ operations at the beginning of *every* lambda step.

## Evidence from History
As this is the first iteration (`iter0001`), there is no previous failure or success history to build upon. However, visual inspection of the baseline codebase identified multiple algorithmic performance bugs, mainly revolving around redundant residual updates and missing dense vectorization logic.

## Why It's Not a Repeat
This is the first attempt, meaning it builds purely off logical and mathematical analysis of the baseline rather than varying past hypotheses. It addresses three distinctly separate algorithmic bottlenecks (inner-loop complexity for covariance method, missing BLAS routines, and KKT gradient caching for Strong Rules) simultaneously.

## Expected Benefit/Risk
**Benefits:** 
- The covariance path inner loops transition from bounded by $O(n |A|)$ back to their mathematically pure $O(|A|^2)$ complexity, granting an exponential speedup on datasets where $n \gg p$.
- For both branches, the $O(np)$ computation inside the strong-rule step is virtually eliminated (cached from prior checks), dropping total execution times noticeably. 
- Level-2 BLAS dense caching optimizes what few $O(n)$ bounds still remain inside the sparse-centric algorithms.

**Risks:**
- The $O(|A|)$ optimization skips residual updates for the covariance inner loop. A minor synchronization bug could hypothetically arise if `grad` caching mismatches the exact analytical formulation, but careful hoisting logic mitigates this precision error drift.
