import math
import random

# Rastrigin 함수 (최소화가 어려운 테스트 함수)
def rastrigin(vec):
    return 10 * len(vec) + sum(x * x - 10 * math.cos(2 * math.pi * x) for x in vec)


class Particle:
    def __init__(self, dim, bounds):
        self.position = [random.uniform(bounds[0], bounds[1]) for _ in range(dim)]
        self.velocity = [random.uniform(-1, 1) for _ in range(dim)]
        self.best_pos = list(self.position)
        self.best_val = rastrigin(self.position)

    def update_velocity(self, global_best, w=0.5, c1=1.5, c2=1.5):
        for i in range(len(self.velocity)):
            r1 = random.random()
            r2 = random.random()
            cognitive = c1 * r1 * (self.best_pos[i] - self.position[i])
            social = c2 * r2 * (global_best[i] - self.position[i])
            self.velocity[i] = w * self.velocity[i] + cognitive + social

    def move(self):
        for i in range(len(self.position)):
            self.position[i] += self.velocity[i]

        val = rastrigin(self.position)
        if val < self.best_val:
            self.best_val = val
            self.best_pos = list(self.position)


def pso(dim=2, bounds=(-5.12, 5.12), swarm_size=30, iterations=100):
    swarm = [Particle(dim, bounds) for _ in range(swarm_size)]
    global_best = min(swarm, key=lambda p: p.best_val).best_pos

    for _ in range(iterations):
        for p in swarm:
            p.update_velocity(global_best)
            p.move()
        global_best = min(swarm, key=lambda p: p.best_val).best_pos

    best_particle = min(swarm, key=lambda p: p.best_val)
    return best_particle.best_pos, best_particle.best_val


if __name__ == "__main__":
    best_pos, best_val = pso()
    print("최적 위치:", [round(x, 3) for x in best_pos])
    print("함수 값:", round(best_val, 4))
