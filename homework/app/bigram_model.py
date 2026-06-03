import random
from collections import defaultdict, Counter


class BigramModel:
    def __init__(self, corpus):
        self.corpus = corpus
        self.bigram_probs = defaultdict(dict)
        self.train()

    def tokenize(self, text):
        """
        简单分词：
        1. 转小写
        2. 按空格切分
        """
        return text.lower().split()

    def train(self):
        """
        训练 bigram model：
        统计每个词后面最可能出现哪些词。
        """
        bigram_counts = defaultdict(Counter)

        for sentence in self.corpus:
            words = self.tokenize(sentence)

            # 生成 bigrams，比如 ["i", "love", "you"]
            # 会变成 ("i", "love"), ("love", "you")
            for w1, w2 in zip(words[:-1], words[1:]):
                bigram_counts[w1][w2] += 1

        # 把 count 转成 probability
        for w1, next_words in bigram_counts.items():
            total = sum(next_words.values())

            for w2, count in next_words.items():
                self.bigram_probs[w1][w2] = count / total

    def generate_next_word(self, current_word):
        """
        根据当前词，生成下一个词。
        """
        current_word = current_word.lower()

        if current_word not in self.bigram_probs:
            return None

        next_words = list(self.bigram_probs[current_word].keys())
        probabilities = list(self.bigram_probs[current_word].values())

        return random.choices(next_words, weights=probabilities, k=1)[0]

    def generate_text(self, start_word, length):
        """
        从 start_word 开始生成 length 个词。
        """
        words = [start_word]
        current_word = start_word.lower()

        for _ in range(length - 1):
            next_word = self.generate_next_word(current_word)

            if next_word is None:
                break

            words.append(next_word)
            current_word = next_word

        return " ".join(words)