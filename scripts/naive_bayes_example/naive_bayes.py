import math
import re
from collections import defaultdict

# 간단한 학습 데이터 [(문장, 라벨)]
train_data = [
    ("이 영화 정말 재미있다", "pos"),
    ("최고로 즐거운 경험", "pos"),
    ("별로 재미없고 지루했음", "neg"),
    ("시간 낭비인 영화", "neg"),
]


def tokenize(text):
    # 단어 단위로 잘라 소문자로 변환
    return re.findall(r"[\w]+", text.lower())


def train(dataset):
    vocab = set()
    word_counts = {"pos": defaultdict(int), "neg": defaultdict(int)}
    class_counts = {"pos": 0, "neg": 0}
    for sentence, label in dataset:
        class_counts[label] += 1
        for word in tokenize(sentence):
            vocab.add(word)
            word_counts[label][word] += 1
    prior = {c: class_counts[c] / len(dataset) for c in class_counts}
    word_probs = {}
    for c in word_counts:
        total = sum(word_counts[c].values()) + len(vocab)
        word_probs[c] = {w: math.log((word_counts[c][w] + 1) / total) for w in vocab}
    return vocab, prior, word_probs


def predict(sentence, vocab, prior, word_probs):
    words = tokenize(sentence)
    scores = {c: math.log(prior[c]) for c in prior}
    for c in prior:
        for w in words:
            if w in vocab:
                scores[c] += word_probs[c].get(w, math.log(1 / len(vocab)))
    return max(scores, key=scores.get)


if __name__ == "__main__":
    vocab, prior, word_probs = train(train_data)
    test_sentence = "정말 즐거운 영화"
    pred = predict(test_sentence, vocab, prior, word_probs)
    print(f"'{test_sentence}' => {pred}")
