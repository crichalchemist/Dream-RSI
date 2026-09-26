# Proposal: Anderson-Accelerated Coordinate Descent (AACD)

## Mechanism
We propose a genuinely new optimization topological structure that replaces the heavy active-set direct solvers (LDLT / PCG) with **Anderson Acceleration (AA)** applied directly to the Coordinate Descent (CD) fixed-point iteration.
Instead of explicitly computing and inverting a restricted Gram matrix ($O(|A|^3)$) or running expensive inner Conjugate Gradient loops ($O(n|A|)$), AA extrapolates the limit of the CD sequence. By maintaining a short history ($m \approx 5$) of CD iterates, we solve a tiny $m \times m$ least-squares problem to find the optimal mixing weights. We then compute the extrapolated state, project it onto the current orthant (snapping zero-crossings strictly to zero), and seamlessly resume CD.
We embed this inside a dual-engine architecture:
1. Low-Dimension ($p < 500$): Covariance CD + AA.
2. High-Dimension ($p \ge 500$): Naive CD + AA.

## Evidence from history
`attempt_b001_a002` scored the best (64ms) but its exact block Newton step (LDLT) took 858ms on Problem 12 due to $O(|A|^3)$ inversions and full $O(|A|^2 n)$ Gram caching.
`attempt_b003_a002` tried to fix this with PCG, but 20 iterations of PCG still required $20 \times O(n|A|)$ matrix-vector operations per acceleration step, taking 851ms on Problem 12.
Direct solvers or their iterative equivalents (CG) are inherently too slow for large $|A|$. CD is $O(|A| n)$ per epoch, which is optimal, but it zigzags. AA perfectly cures zigzagging using only $O(|A| m^2)$ overhead, without ever evaluating a Hessian or taking a matrix-vector product.

## Why it's not a repeat
No previous attempt has used Anderson Acceleration (or any sequence extrapolation technique). All prior attempts either fell back to second-order direct solves (LDLT, PDAS, Active Set Newton) or quasi-Newton (OWL-QN) to escape CD stalling. AA is a first-order sequence transformation that accelerates the fixed-point mapping itself, making it a structurally distinct, untried paradigm.

## Expected benefit/risk
**Benefit**: Problem 12 will see massive speedups because AACD requires ZERO dense inversions ($O(|A|^3) \to 0$) and ZERO inner-loop matrix-vector products ($20 \times O(n|A|) \to 0$). AA simply blends the states we already computed, providing Newton-like acceleration at virtually zero cost.
**Risk**: AA can theoretically become unstable if the active set oscillates rapidly. We mitigate this by applying AA only when the active support has stabilized for 2 epochs, and by strictly projecting the extrapolated point to guarantee it never crosses orthant boundaries.
