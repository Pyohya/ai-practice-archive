import numpy as np
import random


class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.reset()

    def reset(self):
        self.pos = self.start
        return self.pos

    def step(self, action):
        x, y = self.pos
        if action == 0 and x > 0:
            x -= 1
        elif action == 1 and x < self.size - 1:
            x += 1
        elif action == 2 and y > 0:
            y -= 1
        elif action == 3 and y < self.size - 1:
            y += 1
        self.pos = (x, y)
        reward = 1 if self.pos == self.goal else -0.01
        done = self.pos == self.goal
        return self.pos, reward, done


def q_learning(env, episodes=200, alpha=0.1, gamma=0.9, epsilon=0.1):
    Q = np.zeros((env.size, env.size, 4))
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                action = np.argmax(Q[state[0], state[1]])
            next_state, reward, done = env.step(action)
            old = Q[state[0], state[1], action]
            next_max = np.max(Q[next_state[0], next_state[1]])
            Q[state[0], state[1], action] = old + alpha * (reward + gamma * next_max - old)
            state = next_state
    return Q


def play(env, Q):
    state = env.reset()
    path = [state]
    done = False
    while not done:
        action = np.argmax(Q[state[0], state[1]])
        state, _, done = env.step(action)
        path.append(state)
    return path


if __name__ == "__main__":
    env = GridWorld(size=4)
    Q = q_learning(env)
    path = play(env, Q)
    print("학습된 경로:")
    print(path)
