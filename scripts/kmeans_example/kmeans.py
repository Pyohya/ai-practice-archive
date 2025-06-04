import numpy as np


def kmeans(X, k, iterations=100):
    idx = np.random.choice(len(X), k, replace=False)
    centroids = X[idx]
    for _ in range(iterations):
        distances = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
        clusters = np.argmin(distances, axis=1)
        new_centroids = np.array([X[clusters == i].mean(axis=0) for i in range(k)])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids, clusters


if __name__ == "__main__":
    np.random.seed(1)
    cluster1 = np.random.randn(100, 2) + np.array([0, 0])
    cluster2 = np.random.randn(100, 2) + np.array([5, 5])
    X = np.vstack((cluster1, cluster2))

    centroids, clusters = kmeans(X, k=2, iterations=50)
    print("최종 중심점:")
    print(centroids)
    print("클러스터 할당 샘플 (처음 10개):", clusters[:10])
