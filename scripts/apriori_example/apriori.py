# 간단한 Apriori 알고리즘 구현

# 각 거래가 포함하는 아이템 집합을 리스트로 전달합니다.
def apriori(transactions, min_support=2):
    # 초기 1-아이템 후보 생성
    items = set()
    for t in transactions:
        items.update(t)
    current = [{i} for i in items]
    frequent = []

    while current:
        valid = []
        counts = {}
        for itemset in current:
            count = sum(itemset.issubset(t) for t in transactions)
            if count >= min_support:
                valid.append(itemset)
                counts[frozenset(itemset)] = count
        frequent.extend((set(k), v) for k, v in counts.items())

        # 다음 후보 생성
        next_candidates = set()
        for a in valid:
            for b in valid:
                union = a | b
                if len(union) == len(a) + 1:
                    next_candidates.add(frozenset(union))
        current = [set(c) for c in next_candidates]
    return frequent

if __name__ == "__main__":
    dataset = [
        {"우유", "빵", "버터"},
        {"맥주", "빵"},
        {"우유", "맥주", "빵", "버터"},
        {"우유", "버터"}
    ]
    results = apriori(dataset, min_support=2)
    print("자주 등장하는 아이템셋:")
    for items, count in results:
        print(items, "->", count)
