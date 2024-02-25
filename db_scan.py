import numpy
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


def dbscan(D, eps, MinPts):
    labels = [0] * len(D)

    C = 0

    for P in range(0, len(D)):
        if not (labels[P] == 0):
            continue

        NeighborPts = region_query(D, P, eps)
        if len(NeighborPts) < MinPts:
            labels[P] = -1
        else:
            C += 1
            grow_cluster(D, labels, P, NeighborPts, C, eps, MinPts)

    return labels


def grow_cluster(D, labels, P, NeighborPts, C, eps, MinPts):
    labels[P] = C

    i = 0
    while i < len(NeighborPts):
        Pn = NeighborPts[i]

        if labels[Pn] == -1:
            labels[Pn] = C

        elif labels[Pn] == 0:
            labels[Pn] = C

            PnNeighborPts = region_query(D, Pn, eps)

            if len(PnNeighborPts) >= MinPts:
                NeighborPts = NeighborPts + PnNeighborPts

        i += 1


def region_query(D, P, eps):
    neighbors = []

    # For each point in the dataset...
    for Pn in range(0, len(D)):
        # If the distance is below the threshold, add it to the neighbors list.
        if numpy.linalg.norm(D[P] - D[Pn]) < eps:
            neighbors.append(Pn)

    return neighbors


centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(
    n_samples=750, centers=centers, cluster_std=0.4, random_state=0
)

X = StandardScaler().fit_transform(X)


print("Running my implementation...")
my_labels = dbscan(X, eps=0.3, MinPts=10)


print(my_labels)
