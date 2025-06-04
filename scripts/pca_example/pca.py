# 2차원 데이터용 간단한 PCA 구현
import math
import random

# 평균 계산
def mean(values):
    return sum(values) / len(values)

# 공분산 계산
def covariance(x, y):
    avg_x = mean(x)
    avg_y = mean(y)
    return sum((a-avg_x)*(b-avg_y) for a, b in zip(x, y)) / (len(x)-1)

# 2x2 공분산 행렬의 고유벡터(주성분) 계산
def principal_component(data):
    xs = [p[0] for p in data]
    ys = [p[1] for p in data]
    var_x = covariance(xs, xs)
    var_y = covariance(ys, ys)
    cov_xy = covariance(xs, ys)
    trace = var_x + var_y
    det = var_x * var_y - cov_xy ** 2
    lam = trace / 2 + math.sqrt((trace/2)**2 - det)
    vector = [lam - var_y, cov_xy]
    length = math.sqrt(vector[0]**2 + vector[1]**2)
    return [v/length for v in vector]

if __name__ == "__main__":
    random.seed(1)
    data = [(random.gauss(0,1), random.gauss(0,0.3)) for _ in range(100)]
    pc = principal_component(data)
    print("첫 번째 주성분 벡터:")
    print([round(v,3) for v in pc])
