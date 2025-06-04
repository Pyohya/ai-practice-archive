import math
import random

# XOR 학습용 데이터 (입력, 정답)
train_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def dsigmoid(y):
    # 출력 y에 대한 시그모이드 미분
    return y * (1 - y)


class MLP:
    def __init__(self, input_size, hidden_size, lr=0.5):
        self.lr = lr
        self.w1 = [
            [random.uniform(-1, 1) for _ in range(input_size)]
            for _ in range(hidden_size)
        ]
        self.b1 = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.w2 = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.b2 = random.uniform(-1, 1)

    def forward(self, x):
        h_raw = [
            sum(self.w1[i][j] * x[j] for j in range(len(x))) + self.b1[i]
            for i in range(len(self.w1))
        ]
        h = [sigmoid(z) for z in h_raw]
        out_raw = sum(self.w2[i] * h[i] for i in range(len(h))) + self.b2
        out = sigmoid(out_raw)
        return h, out

    def train(self, data, epochs=10000):
        for _ in range(epochs):
            for x, y in data:
                h, out = self.forward(x)
                # 출력층 오차와 델타
                delta2 = (out - y) * dsigmoid(out)

                # 은닉층 오차와 델타
                deltas1 = [
                    dsigmoid(h[i]) * self.w2[i] * delta2
                    for i in range(len(h))
                ]

                # 가중치 업데이트
                for i in range(len(self.w2)):
                    self.w2[i] -= self.lr * delta2 * h[i]
                self.b2 -= self.lr * delta2

                for i in range(len(self.w1)):
                    for j in range(len(self.w1[i])):
                        self.w1[i][j] -= self.lr * deltas1[i] * x[j]
                    self.b1[i] -= self.lr * deltas1[i]

    def predict(self, x):
        _, out = self.forward(x)
        return out


if __name__ == "__main__":
    mlp = MLP(input_size=2, hidden_size=2, lr=0.5)
    mlp.train(train_data, epochs=5000)

    print("XOR 결과 예측:")
    for x, y in train_data:
        pred = mlp.predict(x)
        print(f"입력 {x} => 예측 {pred:.3f} (정답 {y})")
