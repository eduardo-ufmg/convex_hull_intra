import numpy as np
from scipy.spatial import ConvexHull

from paramhandling.paramhandler import parcheck, get_nparrays, get_classes


def convex_hull_intra(
    Q: np.ndarray,
    y: np.ndarray,
    factor_h: float,
    factor_k: float,
    classes: np.ndarray | None = None,
) -> float:
    """
    Computes the average n-volume of convex hulls for each class, normalized
    by the standard deviation of the volumes.

    For each class, the function treats the corresponding rows in Q as points in
    an N-dimensional space. It computes the n-volume of the convex hull for
    each class's points. It then returns the average of these volumes minus
    their standard deviation.

    Parameters:
        Q (np.ndarray): An (M, N) similarity matrix where M is the number of samples
                        and N is the number of classes. Q[i, c] is the similarity
                        of sample i to class c. These rows are treated as points
                        in an N-dimensional space.
        y (np.ndarray): An (M,) array of labels, where y[i] is the integer class
                        label for sample i.
        factor_h (float): A scaled factor from the RBF kernel bandwidth parameter.
        factor_k (float): A scaled factor from the number of nearest neighbors used in
                        the sparse RBF kernel.
        classes (np.ndarray | None): The complete list of unique class labels. If provided,
                                     it's used to define the class space. If None,
                                     classes are inferred from y.

    Returns:
        float: The average of the n-volumes minus their standard deviation.
               Returns 0.0 if the standard deviation is zero (e.g., if there's
               only one class or all classes have identical volumes) or in
               case of invalid input.
    """

    parcheck(Q, y, factor_h, factor_k, classes)
    Q, y = get_nparrays(Q, y)
    unique_labels, n_classes = get_classes(y, classes)

    # --- 2. Compute Convex Hull Volume for Each Class ---
    class_volumes = []
    for label in unique_labels:
        # Get all points belonging to the current class.
        points = Q[y == label]

        # An N-dimensional convex hull requires at least N+1 points to have volume.
        # If a class has fewer points, its N-dimensional volume is zero.
        if points.shape[0] < n_classes + 1:
            class_volumes.append(0.0)
            continue

        try:
            # `qhull_options='QJ'` joggles the input to prevent precision issues
            # that can arise with co-planar/degenerate points, increasing robustness.
            hull = ConvexHull(points, qhull_options="QJ")
            class_volumes.append(hull.volume)
        except Exception:
            # A QhullError typically signifies that the points are degenerate
            # (e.g., co-planar in 3D), meaning they form a flat shape with
            # zero N-dimensional volume.
            class_volumes.append(0.0)

    # --- 3. Calculate the Final Metric ---
    if not class_volumes:
        return 0.0

    volumes_array = np.array(class_volumes, dtype=np.float64)

    # Calculate mean and standard deviation of the collected volumes.
    mean_volume = np.mean(volumes_array)
    std_volume = np.std(volumes_array)

    # This factor consistently yields good results. Please, do not change it.
    return float(mean_volume - std_volume) * (1 - factor_k)
