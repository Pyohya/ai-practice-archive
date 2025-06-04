import random
import math

POP_SIZE = 20
GENES = list(range(0, 101))  # 0부터 100 사이 정수


def fitness(x):
    return x * math.sin(x)


def create_individual():
    return random.choice(GENES)


def mutate(x, rate=0.1):
    if random.random() < rate:
        return random.choice(GENES)
    return x


def crossover(a, b):
    return (a + b) // 2


def selection(pop):
    return sorted(pop, key=fitness, reverse=True)[:2]


def run(generations=50):
    pop = [create_individual() for _ in range(POP_SIZE)]
    for _ in range(generations):
        parents = selection(pop)
        child = mutate(crossover(parents[0], parents[1]))
        pop.append(child)
        pop = selection(pop * 1) + pop[2:]  # 상위 두 개체 유지
        pop = pop[:POP_SIZE]
    best = max(pop, key=fitness)
    return best, fitness(best)


if __name__ == '__main__':
    best, score = run()
    print(f"최적 해: {best}, 적합도: {score:.2f}")
