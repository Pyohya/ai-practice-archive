import math

# 데이터셋 [특징1, 특징2, 클래스]
dataset = [
    [2.771, 1.784, 0],
    [1.728, 1.169, 0],
    [3.678, 2.812, 0],
    [3.961, 2.619, 0],
    [2.999, 2.209, 0],
    [7.497, 3.162, 1],
    [9.002, 3.339, 1],
    [7.444, 0.476, 1],
    [10.124, 3.234, 1],
    [6.642, 3.319, 1],
]


class Node:
    def __init__(self, index=None, value=None, left=None, right=None, label=None):
        self.index = index  # 분할에 사용할 특징 인덱스
        self.value = value  # 분할 기준 값
        self.left = left    # 왼쪽 서브트리
        self.right = right  # 오른쪽 서브트리
        self.label = label  # 단말 노드일 경우 클래스 레이블


def test_split(index, value, rows):
    left, right = [], []
    for row in rows:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


def gini_index(groups, classes):
    n_instances = sum(len(group) for group in groups)
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        labels = [row[-1] for row in group]
        for class_val in classes:
            p = labels.count(class_val) / size
            score += p * p
        gini += (1.0 - score) * (size / n_instances)
    return gini


def get_split(rows):
    class_values = list(set(row[-1] for row in rows))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(rows[0]) - 1):
        for row in rows:
            groups = test_split(index, row[index], rows)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth + 1)
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)


def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root


def predict(node, row):
    if isinstance(node, dict):
        if row[node['index']] < node['value']:
            return predict(node['left'], row)
        else:
            return predict(node['right'], row)
    else:
        return node


def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % (' ' * depth * 2, node['index'], node['value']))
        print_tree(node['left'], depth + 1)
        print_tree(node['right'], depth + 1)
    else:
        print('%s[Leaf %s]' % (' ' * depth * 2, node))


if __name__ == '__main__':
    tree = build_tree(dataset, max_depth=3, min_size=1)
    print_tree(tree)
    test_point = [1.5, 2.0]
    pred = predict(tree, test_point)
    print(f"테스트 포인트 {test_point} 예측: {pred}")
