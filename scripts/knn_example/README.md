# k-최근접 이웃 (k-Nearest Neighbors, kNN) 알고리즘 예제

이 저장소에는 간단한 k-최근접 이웃 (kNN) 알고리즘을 파이썬으로 구현한 예제가 포함되어 있습니다.

## kNN 알고리즘이란?

k-최근접 이웃 알고리즘은 지도 학습에서 사용되는 간단하면서도 강력한 분류 (Classification) 및 회귀 (Regression) 방법입니다. kNN의 핵심 아이디어는 다음과 같습니다:

1.  **거리 측정**: 새로운 데이터 포인트가 주어지면, 기존 학습 데이터셋의 모든 데이터 포인트들과의 거리를 측정합니다. 일반적으로 유클리드 거리가 많이 사용됩니다.
2.  **가장 가까운 k개의 이웃 찾기**: 측정된 거리들을 기준으로, 새로운 데이터 포인트에서 가장 가까운 순서대로 k개의 데이터 포인트를 선택합니다. 여기서 'k'는 사용자가 지정하는 자연수입니다.
3.  **다수결 투표 (분류) 또는 평균 (회귀)**:
    *   **분류의 경우**: 선택된 k개의 이웃들이 어떤 클래스에 속하는지 확인하고, 가장 많은 이웃이 속한 클래스로 새로운 데이터 포인트의 클래스를 예측합니다. (다수결 원칙)
    *   **회귀의 경우**: 선택된 k개의 이웃들의 실제 값들의 평균을 새로운 데이터 포인트의 값으로 예측합니다.

kNN은 이해하기 쉽고 구현이 간단하지만, 데이터셋이 커질 경우 예측 시간이 오래 걸릴 수 있고, 적절한 'k' 값과 거리 측정 방법을 선택하는 것이 중요합니다.

## `knn.py` 코드 설명

`knn.py` 파일은 기본적인 kNN 분류 알고리즘을 단계별로 구현합니다.

```python
import math
from collections import Counter

# 1. 샘플 데이터셋: [x_좌표, y_좌표, 클래스_레이블]
dataset = [
    [2, 3, 'A'],
    [4, 7, 'A'],
    [5, 4, 'A'],
    [7, 2, 'B'],
    [8, 5, 'B'],
    [9, 6, 'B']
]
```

*   **`dataset`**: 예제에서 사용할 학습 데이터입니다. 각 내부 리스트는 하나의 데이터 포인트를 나타내며, 처음 두 요소는 특징(feature)이고 마지막 요소는 해당 데이터 포인트의 클래스 레이블입니다.

```python
# 2. 유클리드 거리 계산 함수
def euclidean_distance(point1, point2):
    """두 점 사이의 유클리드 거리를 계산합니다 (클래스 레이블 제외)."""
    distance = 0
    # 마지막 요소는 클래스 레이블이므로, 마지막에서 두 번째 요소까지만 반복합니다.
    for i in range(len(point1) - 1):
        distance += (point1[i] - point2[i])**2
    return math.sqrt(distance)
```

*   **`euclidean_distance(point1, point2)`**: 두 데이터 포인트 `point1`과 `point2` 사이의 유클리드 거리를 계산합니다. 클래스 레이블은 거리 계산에 포함되지 않습니다.

```python
# 3. k개의 가장 가까운 이웃을 찾는 함수
def get_neighbors(train_data, test_instance, k):
    """학습 데이터에서 테스트 인스턴스와 가장 가까운 k개의 이웃을 찾습니다."""
    distances = []
    for train_point in train_data:
        dist = euclidean_distance(test_instance, train_point)
        distances.append((train_point, dist))
    distances.sort(key=lambda tup: tup[1]) # 거리를 기준으로 정렬
    
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors
```

*   **`get_neighbors(train_data, test_instance, k)`**:
    1.  `test_instance` (새로운 데이터 포인트)와 `train_data` (학습 데이터셋) 내의 모든 포인트 사이의 거리를 `euclidean_distance` 함수를 사용해 계산합니다.
    2.  계산된 거리와 해당 학습 데이터 포인트를 튜플 형태로 `distances` 리스트에 저장합니다.
    3.  `distances` 리스트를 거리가 짧은 순으로 정렬합니다.
    4.  정렬된 리스트에서 가장 가까운 `k`개의 이웃(학습 데이터 포인트)을 선택하여 반환합니다.

```python
# 4. 새로운 데이터 포인트의 클래스를 예측하는 함수
def predict_classification(train_data, test_instance, k):
    """k개의 최근접 이웃을 기반으로 테스트 인스턴스의 클래스를 예측합니다."""
    neighbors = get_neighbors(train_data, test_instance, k)
    output_values = [row[-1] for row in neighbors] # 이웃들의 클래스 레이블을 가져옵니다.
    prediction = Counter(output_values).most_common(1)[0][0] # 다수결 투표
    return prediction
```

*   **`predict_classification(train_data, test_instance, k)`**:
    1.  `get_neighbors` 함수를 호출하여 `test_instance`의 가장 가까운 `k`개의 이웃을 찾습니다.
    2.  찾아낸 이웃들의 클래스 레이블만을 추출합니다.
    3.  `collections.Counter`를 사용하여 가장 빈번하게 등장하는 클래스 레이블을 찾습니다 (다수결 투표).
    4.  가장 많이 선택된 클래스 레이블을 예측값으로 반환합니다.

```python
# 5. 함수 사용 예시
if __name__ == '__main__':
    # 분류할 새로운 데이터 포인트 (특징만 있고, 레이블은 아직 없음)
    new_point = [5, 5] 
    k_value = 3

    # new_point에 대한 분류 예측
    predicted_class = predict_classification(dataset, new_point, k_value)
    print(f"새로운 포인트 {new_point}는 클래스 '{predicted_class}'로 분류됩니다.")

    # 다른 포인트 예시
    new_point_2 = [1, 1]
    predicted_class_2 = predict_classification(dataset, new_point_2, k_value)
    print(f"새로운 포인트 {new_point_2}는 클래스 '{predicted_class_2}'로 분류됩니다.")
```

*   **`if __name__ == '__main__':`**: 이 스크립트가 직접 실행될 때 수행될 코드 블록입니다.
    *   `new_point = [5, 5]`와 같이 분류하고자 하는 새로운 데이터 포인트를 정의합니다.
    *   `k_value = 3`으로 k값을 설정합니다.
    *   `predict_classification` 함수를 호출하여 `new_point`의 클래스를 예측하고 결과를 출력합니다.

## 예제 실행 방법

터미널 또는 명령 프롬프트에서 `scripts/knn_example` 디렉토리로 이동한 후, 다음 명령어를 실행하여 파이썬 스크립트를 실행할 수 있습니다:

```bash
python knn.py
```

스크립트가 실행되면, 예제 코드에 정의된 새로운 데이터 포인트들에 대한 예측된 클래스가 출력됩니다.
예시 출력:
```
The new point [5, 5] is classified as: A
The new point [1, 1] is classified as: A
```
(실제 `knn.py`의 `new_point` 값과 `k_value`에 따라 출력은 달라질 수 있습니다. 위 예시 출력은 `new_point = [5, 5]` 와 `new_point_2 = [1, 1]` 일 때, `k_value = 3`을 기준으로 합니다.)

**참고**: `knn.py` 파일이 있는 `scripts/knn_example` 디렉토리 내에서 위 명령어를 실행해야 합니다. 만약 저장소의 최상위 디렉토리에서 실행한다면 다음과 같이 경로를 포함해야 합니다:
```bash
python scripts/knn_example/knn.py
```
