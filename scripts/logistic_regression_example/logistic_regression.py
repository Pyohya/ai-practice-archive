import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict(X, weights):
    return sigmoid(np.dot(X, weights))


def compute_loss(X, y, weights):
    m = len(y)
    preds = predict(X, weights)
    return -np.mean(y * np.log(preds + 1e-15) + (1 - y) * np.log(1 - preds + 1e-15))


def train(X, y, lr=0.1, epochs=1000):
    weights = np.zeros(X.shape[1])
    for _ in range(epochs):
        preds = predict(X, weights)
        gradient = np.dot(X.T, preds - y) / len(y)
        weights -= lr * gradient
    return weights


if __name__ == "__main__":
    np.random.seed(0)
    class0 = np.random.multivariate_normal([2, 2], [[0.5, 0], [0, 0.5]], 50)
    class1 = np.random.multivariate_normal([4, 4], [[0.5, 0], [0, 0.5]], 50)
    X = np.vstack((class0, class1))
    y = np.array([0] * 50 + [1] * 50)

    # bias term
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    weights = train(X, y, lr=0.1, epochs=1000)
    loss = compute_loss(X, y, weights)
    print(f"Training finished. Loss: {loss:.4f}")

    # prediction example
    test_point = np.array([1, 3, 3])  # with bias term
    prob = predict(test_point, weights)
    print(f"Test point {test_point[1:]} probability of class 1: {prob:.2f}")
