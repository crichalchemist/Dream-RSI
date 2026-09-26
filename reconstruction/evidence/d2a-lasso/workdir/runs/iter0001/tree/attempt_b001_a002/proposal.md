# Proposal: LARS-Accelerated Covariance CD

## Mechanism
We introduce **LARS-Accelerated Covariance CD**, a hybrid algorithm that fuses the robustness of Covariance Coordinate Descent (CD) with the exact orthant-boundary tracking of Primal-Dual Active Set (PDAS). 
When Covariance CD stalls due to highly collinear features (the support stabilizes but iterations plateau), we form the exact restricted Gram matrix $G_{nz}$ and compute the direct block Newton step. In previous implementations (like `attempt_b000`), if this proposed Newton step crossed an orthant boundary (changed a variable's sign), the step was entirely rejected, wasting the $O(|A|^3)$ `ldlt()` solve and falling back to slow CD. 
Instead, we now treat the subspace step as a true LARS step: we compute the maximal step size $\alpha \in (0, 1]$ that keeps all variables in their valid orthants. We then step exactly to the orthant boundary, snap the violating variable precisely to zero, and instantaneously drop it from the active support. Because this truncated step guarantees monotonic progress without crossing bounds, it is NEVER rejected. CD then seamlessly continues from this exact subspace projection. 

## Evidence from history
Reviewing previous attempts shows that `attempt_b000_a001` (CD-Newton Hybrid) scored an excellent $67\text{ms}$ geometric mean but struggled specifically on highly collinear datasets (e.g., Problem 12 took $>850\text{ms}$). This slowness was caused by the algorithm repeatedly attempting exact Cholesky solves, finding sign violations, and aborting the solve to fall back to CD, which was already geometrically stalling. Conversely, `attempt_b003_a001` (GC-PDAS) proved that exact line-search step truncations (LARS logic) perfectly navigate collinearity and zero-crossings without stalling, but using it exclusively (without CD) was slow on sparse orthogonal data ($86\text{ms}$). 

## Why it's not a repeat
This is a structurally novel combination: it embeds the LARS/PDAS exact line-search logic (from `attempt_b003`) directly into the active-set Subspace Acceleration mechanism of a Covariance CD solver (from `attempt_b000`). It replaces the fragile binary "Accept/Reject" acceleration with a guaranteed-progress "Truncate and Drop" step, completely neutralizing the performance cliff on correlated datasets.

## Expected benefit/risk
**Benefit**: Enormous speedups on highly collinear datasets (e.g., Problem 12). Because exact direct solves are never rejected, every $O(|A|^3)$ calculation makes guaranteed structural progress along the piecewise-linear solution path. CD handles uncorrelated variables instantly, while the LARS acceleration instantly resolves correlated variables.
**Risk**: If variables bounce in and out of the support rapidly, we could take many $\alpha < 1$ steps. We mitigate this by only triggering the acceleration when the CD support has stabilized for 2 epochs, ensuring CD has smoothed out high-frequency fluctuations before taking macroscopic LARS steps.
