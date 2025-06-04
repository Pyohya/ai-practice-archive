# 간단한 PageRank 구현

# 웹 그래프를 딕셔너리로 표현합니다. 각 노드는 링크가 걸린 이웃 노드 리스트를 가집니다.
def pagerank(graph, damping=0.85, iterations=20):
    n = len(graph)
    ranks = {page: 1.0 / n for page in graph}
    for _ in range(iterations):
        new_ranks = {}
        for page in graph:
            rank_sum = 0.0
            for node, outs in graph.items():
                if page in outs:
                    rank_sum += ranks[node] / len(outs)
            new_ranks[page] = (1 - damping) / n + damping * rank_sum
        ranks = new_ranks
    return ranks

if __name__ == "__main__":
    sample_graph = {
        "A": ["B", "C"],
        "B": ["C"],
        "C": ["A"],
        "D": ["C"]
    }
    result = pagerank(sample_graph)
    print("PageRank 결과")
    for page, score in result.items():
        print(page, round(score, 3))
