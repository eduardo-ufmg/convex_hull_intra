This function computes a metric that reflects the average spatial volume occupied by different classes of points in a multi-dimensional space, adjusted by the variability in those volumes.

***

### Mathematical Formulation

The procedure begins with a set of $M$ points $\{\mathbf{q}_1, \dots, \mathbf{q}_M\}$ in an $N$-dimensional space ($\mathbb{R}^N$). Each point $\mathbf{q}_i$ is assigned a class label from a set of $N$ unique classes. The calculation also uses a given scalar factor, denoted here as $f_k$.

---

### 1. Intra-Class Convex Hull Volume 🗺️

The core of the method involves analyzing each class independently.

* **Point Partitioning**: First, the $M$ points are grouped into $N$ distinct sets, $P_1, P_2, \dots, P_N$, based on their class labels. The set $P_k$ contains all points belonging to the $k$-th class.

* **Convex Hull and Volume**: For each set of points $P_k$, its **convex hull**, denoted $\text{conv}(P_k)$, is constructed. The convex hull is the smallest convex geometric shape that encloses all points in $P_k$. The $N$-dimensional volume of each of these hulls, $V_k = \text{Volume}(\text{conv}(P_k))$, is then computed.

For a hull to have a non-zero volume in $\mathbb{R}^N$, it must be formed from at least $N+1$ points that are not co-planar (or, more generally, not affinely dependent). If a class $k$ has fewer than $N+1$ points, its $N$-dimensional volume $V_k$ is treated as zero.

---

### 2. Statistical Aggregation and Final Score 📊

After calculating the volume $V_k$ for each of the $N$ classes, these values are aggregated into a final score.

* **Statistical Measures**: Let $\mathcal{V} = \{V_1, V_2, \dots, V_N\}$ be the set of all computed class volumes. The arithmetic **mean** ($\mu_{\mathcal{V}}$) and **standard deviation** ($\sigma_{\mathcal{V}}$) of the values in this set are calculated.
    
    $$
    \mu_{\mathcal{V}} = \frac{1}{N} \sum_{k=1}^{N} V_k
    $$
    
    $$
    \sigma_{\mathcal{V}} = \sqrt{\frac{1}{N} \sum_{k=1}^{N} (V_k - \mu_{\mathcal{V}})^2}
    $$

* **Final Score**: The final score is computed by adjusting the mean volume by its standard deviation and scaling the result by the input factor $f_k$.
    
    $$
    \text{Final Score} = (\mu_{\mathcal{V}} - \sigma_{\mathcal{V}}) \cdot (1 - f_k)
    $$