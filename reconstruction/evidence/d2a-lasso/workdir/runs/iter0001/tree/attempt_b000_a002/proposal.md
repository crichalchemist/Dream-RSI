# Proposal: N-Independent CD via Active-to-All Gram Expansion

## Mechanism
We propose a genuinely new optimization topological structure that entirely eliminates the $O(np)$ residual and KKT bottleneck that bounds all prior methods. Instead of caching only the symmetric active set Gram matrix $G_A = X_A^T X_A$, we lazily build and cache the asymmetric Active-to-All Gram matrix $G_{all} = X_A^T X$. 
Because $X \beta = X_A \beta_A$, the exact full-dimensional gradient can be mathematically synthesized at any point via $g = g_0 - G_{all}^T \beta_A$, where $g_0 = X^T y / n$. This exact synthesis takes $O(|A| p)$ operations, entirely dropping the sample size $n$ from the loop. We pair this N-Independent outer shell with the Subspace-Accelerated CD (Exact LDLT steps) from prior attempts to instantly solve highly collinear active sets.

## Evidence from history
Previous attempts `b001` optimized the inner CD loop nicely, and `b000` introduced Subspace LDLT acceleration to stop CD zigzagging (scoring ~67ms). However, analyzing their code reveals *all* prior methods recompute $r = y - X \beta$ and $grad = X^T r$ from scratch to perform the global KKT check, locking their performance behind hundreds of $O(np)$ dense array sweeps.

## Why it's not a repeat
No previous solver in the tree has cached the $X_A^T X$ projection against the *inactive* features. All prior "cached" implementations were strictly $A \times A$ and kept the $O(np)$ global KKT checks. By introducing the $g_0$ offset and the asymmetric $G_{all}$ cache, we are the first to mathematically excise the $n$-dimensional residual $r$ entirely from the algorithm's lifecycle after initialization. 

## Expected benefit/risk
**Benefit**: Huge scaling advantages for $n \gg p$ domains. The global KKT check complexity drops from $O(n p)$ to $O(|A| p)$. Because we compute $G_{all}$ additions in batched BLAS 3 operations upon strong-rule admissions, cache building is nearly instantaneous, leading to drastic time reductions.
**Risk**: Maintaining the asymmetric Gram matrix requires $O(|A| p)$ memory, which for massive dimensions could stress CPU caches more than the compact symmetric cache, though typically $|A| \le \min(n, p)$ limits this.
