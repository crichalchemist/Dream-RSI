# Proposal: Active-Set Restricted Gram Caching (ASRGC) with Lazy Residual Sync

## Mechanism
We propose a genuinely new hybrid active-set architecture that completely decouples the expensive Gram-matrix cache from the Strong Rule screened set.
In prior attempts, the Covariance Gram matrix $G$ was built for the *entire screened set* $S$. On sparse or high-dimensional problems, $S$ can grow to thousands of features, causing the $O(n|S|^2)$ Gram matrix construction to bottleneck the entire program (taking $>10^9$ operations). 
Instead, we maintain $G_A$ **strictly for the ever-active set $A$** (features that have actually been non-zero). We alternate between two distinct states:
1. **Screened Epoch (Naive CD)**: We iterate over $S$ using the exact residual $r$, taking $O(n)$ per feature. Any feature that becomes non-zero is pushed to $A$, and $G_A$ expands lazily by one row/column ($O(n|A|)$).
2. **Active Inner Loop (Covariance CD + LARS)**: We descend purely on $A$ using $G_A$ and $c_A$, taking $O(|A|)$ per update. We accelerate this with Exact LDLT Subspace Line-Search steps to crush highly-correlated zigzagging.
When the Active loop completes, we perform a **Lazy Residual Sync**: $r = y - X_A \beta_A$ taking only $O(n|A|)$ operations, entirely avoiding the $O(np)$ dense matrix multiplications that throttled past naive methods. 

## Evidence from history
Reviewing past eval scores reveals `attempt_b001` and `attempt_b003` achieved incredible performance on dense highly-correlated datasets using Gram Cache Subspace Acceleration (~64ms geo-mean), but failed dramatically on Problem 12 ($p=3000$), taking $>850\text{ms}$. 
By analyzing the memory access patterns, it's clear why: past methods forced all $p \ge 500$ datasets to use un-cached Naive loops (zigzagging infinitely), OR they forced the Gram cache onto the *screened* set $S$. On Problem 12, $S$ grows massive while the true non-zero support $A$ remains small. Constructing $G_S$ triggered an $O(n|S|^2)$ memory/compute explosion.

## Why it's not a repeat
This is a structurally new optimization pipeline. It combines the $O(|A|)$ exact inner loop of Covariance CD with the $O(n|S|)$ memory footprint of Naive CD. By interleaving Naive and Covariance epochs, and utilizing an $O(n|A|)$ lazy residual synchronization, it merges the distinct advantages of both regimes into a unified, dimension-independent solver without the historic scaling cliffs.

## Expected benefit/risk
**Benefit**: Problem 12 and other high-dimensional datasets will see massive speedups because $G_A$ construction drops from $O(n|S|^2)$ to $O(n|A|^2)$, saving over $10^9$ operations. CD will no longer stall because the exact LARS acceleration operates identically on the restricted subset.
**Risk**: If the true active support $|A|$ grows extremely large ($|A| > 1500$), the Exact LDLT subproblem $O(|A|^3)$ might stall. We mitigate this by bounding the acceleration subset.
