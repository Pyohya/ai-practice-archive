# 틱택토용 간단한 MCTS 구현
import math
import random

class Node:
    def __init__(self, state, player, parent=None):
        self.state = state  # 3x3 보드 리스트
        self.player = player
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0

    def expand(self):
        for i in range(9):
            if self.state[i] == ' ':
                next_state = list(self.state)
                next_state[i] = self.player
                child = Node(next_state, 'O' if self.player == 'X' else 'X', self)
                self.children.append(child)

    def is_terminal(self):
        lines = [
            [0,1,2],[3,4,5],[6,7,8],
            [0,3,6],[1,4,7],[2,5,8],
            [0,4,8],[2,4,6]
        ]
        for a,b,c in lines:
            if self.state[a]==self.state[b]==self.state[c] != ' ':
                return True, self.state[a]
        if ' ' not in self.state:
            return True, None
        return False, None

    def ucb1(self, total_visits, c=1.4):
        if self.visits == 0:
            return float('inf')
        return self.wins / self.visits + c * math.sqrt(math.log(total_visits)/self.visits)

def tree_policy(node):
    while True:
        terminal, _ = node.is_terminal()
        if terminal:
            return node
        if not node.children:
            node.expand()
            return random.choice(node.children)
        node = max(node.children, key=lambda n: n.ucb1(node.visits))

def default_policy(node):
    state = list(node.state)
    player = node.player
    while True:
        terminal, winner = Node(state, player).is_terminal()
        if terminal:
            if winner == 'X':
                return 1
            elif winner == 'O':
                return -1
            else:
                return 0
        moves = [i for i,v in enumerate(state) if v==' ']
        move = random.choice(moves)
        state[move] = player
        player = 'O' if player=='X' else 'X'

def backup(node, result):
    while node:
        node.visits += 1
        node.wins += result
        result = -result
        node = node.parent

def mcts(root, iterations=100):
    for _ in range(iterations):
        leaf = tree_policy(root)
        result = default_policy(leaf)
        backup(leaf, result)
    return max(root.children, key=lambda n: n.visits)

if __name__ == "__main__":
    root = Node([' '] * 9, 'X')
    root.expand()
    best = mcts(root)
    print("추천 수의 보드 상태:")
    for i in range(0,9,3):
        print(best.state[i:i+3])
