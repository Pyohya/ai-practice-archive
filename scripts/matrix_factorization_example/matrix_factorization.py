import random

# R[i][j] 가 0이면 평점이 없는 것으로 간주
R = [
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
]


def matrix_factorization(R, K=2, steps=5000, alpha=0.002, beta=0.02):
    N = len(R)
    M = len(R[0])
    P = [[random.random() for _ in range(K)] for _ in range(N)]
    Q = [[random.random() for _ in range(K)] for _ in range(M)]
    for _ in range(steps):
        for i in range(N):
            for j in range(M):
                if R[i][j] > 0:
                    pred = sum(P[i][k] * Q[j][k] for k in range(K))
                    eij = R[i][j] - pred
                    for k in range(K):
                        P[i][k] += alpha * (2 * eij * Q[j][k] - beta * P[i][k])
                        Q[j][k] += alpha * (2 * eij * P[i][k] - beta * Q[j][k])
    return P, Q


def predict(P, Q):
    N = len(P)
    M = len(Q)
    K = len(P[0])
    result = [[0] * M for _ in range(N)]
    for i in range(N):
        for j in range(M):
            result[i][j] = sum(P[i][k] * Q[j][k] for k in range(K))
    return result


if __name__ == '__main__':
    P, Q = matrix_factorization(R)
    nR = predict(P, Q)
    print('예측 행렬:')
    for row in nR:
        print([round(v, 2) for v in row])
