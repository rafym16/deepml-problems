import numpy as np


def isolation_forest(X: np.ndarray, n_trees: int, sample_size: int, random_state: int = 42) -> np.ndarray:
    """
    Implement Isolation Forest for anomaly detection.

    Parameters:
    - X: 2D numpy array of shape (n_samples, n_features)
    - n_trees: Number of isolation trees to build
    - sample_size: Number of samples to use for building each tree
    - random_state: Random seed for reproducibility

    Returns:
    - scores: 1D numpy array of anomaly scores (higher = more anomalous)
    """
    # Your code here
    np.random.seed(random_state)

    max_depth = int(np.ceil(np.log2(sample_size)))

    # ---- helper: c(n)
    def c(n):
        if n <= 1:
            return 0
        return 2 * (np.log(n - 1) + 0.5772156649) - (2 * (n - 1) / n)

    # ---- build tree
    def build_tree(X, depth):
        if depth >= max_depth or len(X) <= 1:
            return {"size": len(X)}

        n_features = X.shape[1]
        feature = np.random.randint(n_features)

        min_val = X[:, feature].min()
        max_val = X[:, feature].max()

        if min_val == max_val:
            return {"size": len(X)}

        split = np.random.uniform(min_val, max_val)

        left = X[X[:, feature] < split]
        right = X[X[:, feature] >= split]

        return {
            "feature": feature,
            "split": split,
            "left": build_tree(left, depth + 1),
            "right": build_tree(right, depth + 1)
        }

    # ---- path length
    def path_length(x, tree, depth=0):
        if "size" in tree:
            return depth + c(tree["size"])

        feature = tree["feature"]
        split = tree["split"]

        if x[feature] < split:
            return path_length(x, tree["left"], depth + 1)
        else:
            return path_length(x, tree["right"], depth + 1)

    # ---- build forest
    forest = []
    for _ in range(n_trees):
        idx = np.random.choice(len(X), sample_size, replace=False)
        sample = X[idx]
        tree = build_tree(sample, 0)
        forest.append(tree)

    # ---- compute scores
    scores = []
    for x in X:
        paths = [path_length(x, tree) for tree in forest]
        print(f"Paths: {paths}")
        avg_path = np.mean(paths)
        score = 2 ** (-avg_path / c(sample_size))
        scores.append(score)

    return np.array(scores)