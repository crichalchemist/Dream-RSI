# Proposal: Coordinate Descent with Exact Subspace Acceleration (CD-Newton Hybrid)

## Mechanism
We propose a structurally new hybrid algorithm that merges Coordinate Descent (CD) with Exact Active-Set Newton steps. CD is robust and handles L1 non-differentiability perfectly, but suffers from extreme zigzagging (slow convergence) on highly correlated features. Direct solvers (like PDAS or SSN) solve correlated subspaces instantly but are prone to $O(|A|^3)$ explosions or unstable sign-crossings. 
Our hybrid runs Covariance CD as the primary driver. However, we continuously track the non-zero support set. If the support stabilizes for a few epochs, we form the restricted exact Gram matrix $X_{nz}^T X_{nz}$ and execute a direct unconstrained Cholesky solve (`Eigen::LDLT`) to jump to the subspace minimum. If the resulting Newton step preserves the orthant signs, it is accepted and instantly terminates the inner loop, bypassing thousands of CD iterations. If it violates a sign constraint, it is safely rejected, falling back to CD.

## Evidence from history
Reviewing past attempts reveals a clear split: `b001` optimized pure Coordinate Descent (achieving 72ms), while `b002` and `b003` experimented with Exact Active-Set / Semi-Smooth Newton (scoring 68ms and 102ms). SSN showed that direct solves are powerful for this dataset, but managing the active set bounds natively in Newton is fragile and scale-limited. CD handles bounds robustly but is sequentially slow.

## Why it's not a repeat
This is the first combination of both foundational paradigms. Instead of choosing between first-order CD or second-order Active Set, we use CD as the active-set identifier and orthant-bounds enforcer, and use the second-order solver solely as a Subspace Accelerator. This mirrors advanced SOTA solvers like Blitz or Celer, but implements the exact Newton step natively within a continuously running Covariance CD loop.

## Expected benefit/risk
**Benefit**: For highly correlated features (e.g., `corr_high`), the algorithm will instantly snap to the exact mathematical minimum via the LDLT solve as soon as the support is identified, achieving $O(1)$ epochs instead of $O(1000)$.
**Risk**: Constructing and solving the $O(|nz|^3)$ LDLT system incurs overhead. We mitigate this by only triggering the acceleration when the support is stable, and capping the maximum subspace dimension.
