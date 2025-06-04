# 간단한 시뮬레이티드 어닐링 예제 (TSP)
import math
import random

# 두 지점 사이 거리 계산
def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# 전체 경로 길이 계산
def total_distance(path):
    d = 0
    for i in range(len(path)):
        d += distance(path[i], path[(i+1)%len(path)])
    return d

# 시뮬레이티드 어닐링 구현
def anneal(points, temp=1000.0, cooling=0.995):
    current = points[:]
    best = points[:]
    while temp > 1:
        i, j = sorted(random.sample(range(len(points)), 2))
        new = current[:]
        new[i:j] = reversed(new[i:j])
        if total_distance(new) < total_distance(current) or random.random() < math.exp((total_distance(current)-total_distance(new))/temp):
            current = new
            if total_distance(current) < total_distance(best):
                best = current
        temp *= cooling
    return best

if __name__ == "__main__":
    random.seed(0)
    cities = [(random.random()*100, random.random()*100) for _ in range(10)]
    result = anneal(cities)
    print("최적(추정) 경로:")
    for c in result:
        print(c)
    print("길이:", round(total_distance(result),2))
