import torch


def k_means_clustering(points, k, initial_centroids, max_iterations):
    """
    Perform k-means clustering on `points` into `k` clusters.
    points: tensor of shape (n_points, n_features)
    initial_centroids: tensor of shape (k, n_features)
    max_iterations: maximum number of iterations
    Returns a list of k centroids as tuples, rounded to 4 decimals.
    """
    # Convert to tensors
    points_t = torch.as_tensor(points, dtype=torch.float)
    centroids = torch.as_tensor(initial_centroids, dtype=torch.float)
    # Your implementation here
    updated_centroids = []
    for _ in range(max_iterations):
        distances = torch.cdist(points_t, centroids)

        cluster_ids = torch.argmin(distances, dim=1)
        new_centroids = []


        for i in range(k):
            cluster_points = points_t[cluster_ids == i]

            if len(cluster_points) == 0:
                new_centroids.append(centroids[i])
            else:
                new_centroids.append(cluster_points.mean(dim=0))

        updated_centroids = torch.stack(new_centroids)

    final_centroids = updated_centroids
    return final_centroids