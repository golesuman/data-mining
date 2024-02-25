import numpy as np
from typing import List, Tuple


def k_medoids_clustering(
    data: List[Tuple[float, float]], k: int, max_iter=100, random_seed=42
) -> Tuple[List[int], List[Tuple[float, float]]]:
    """
    k-medoid clustering with voronoi iteration
    """
    # Convert list of points to numpy array
    data = np.array(data)

    # Step 1: Initialization
    np.random.seed(random_seed)
    N = data.shape[0]
    medoids_idx = np.random.choice(N, k, replace=False)
    medoids = data[medoids_idx].copy()
    distances = np.zeros((N, k))

    for i in range(k):
        distances[:, i] = np.sum(np.abs(data - medoids[i]), axis=1)

    # Assign each non-medoid data point to the closest medoid
    labels = np.argmin(distances, axis=1)
    old_labels = np.empty(N)

    # Step 2: Update
    for it in range(max_iter):
        best_swap = (-1, -1, 0)
        best_distances = np.zeros(N)
        for i in range(k):
            # Compute the cost of swapping medoid and non-medoid data points
            non_medoids_idx = np.arange(N)[
                np.logical_not(np.isin(np.arange(N), medoids_idx))
            ]
            for j in non_medoids_idx:
                new_medoid = data[j]
                new_distances = np.sum(np.abs(data - new_medoid), axis=1)
                cost_change = np.sum(new_distances[labels == i]) - np.sum(
                    distances[labels == i, i]
                )
                if cost_change < best_swap[2]:
                    best_swap = (i, j, cost_change)
                    best_distances = new_distances
        if best_swap == (-1, -1, 0):
            break

        i, j, _ = best_swap
        distances[:, i] = best_distances
        medoids[i] = data[j]

        labels = np.argmin(distances, axis=1)

        old_labels = labels

    return labels.tolist(), medoids.tolist()


# Example Usage
if __name__ == "__main__":
    # List of points as input
    points = [(1, 2), (2, 3), (3, 4), (10, 11), (11, 12), (12, 13)]

    # Number of clusters
    k = 2

    # Perform k-medoids clustering
    clusters, medoids = k_medoids_clustering(points, k)

    # Print the clusters and medoids
    print("Clusters:")
    for cluster_id, point in zip(clusters, points):
        print(f"Point {point} belongs to cluster {cluster_id}")

    print("\nMedoids:")
    for i, medoid in enumerate(medoids):
        print(f"Cluster {i}: Medoid {medoid}")
