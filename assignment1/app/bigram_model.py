import random
import re
from collections import Counter, defaultdict


class BigramModel:
    def __init__(self, corpus, frequency_threshold=None):
        self.corpus = corpus
        self.frequency_threshold = frequency_threshold
        self.vocab = []
        self.bigram_probs = defaultdict(dict)
        self.train()

    def tokenize(self, text):
        tokens = re.findall(r"\b\w+\b", text.lower())

        if not self.frequency_threshold:
            return tokens

        word_counts = Counter(tokens)

        return [
            token for token in tokens
            if word_counts[token] >= self.frequency_threshold
        ]

    def train(self):
        if isinstance(self.corpus, list):
            text = " ".join(self.corpus)
        else:
            text = self.corpus

        words = self.tokenize(text)

        unigram_counts = Counter(words)
        bigrams = list(zip(words[:-1], words[1:]))
        bigram_counts = Counter(bigrams)

        self.vocab = list(unigram_counts.keys())

        for (word1, word2), count in bigram_counts.items():
            self.bigram_probs[word1][word2] = count / unigram_counts[word1]

    def generate_next_word(self, current_word):
        current_word = current_word.lower()
        next_words = self.bigram_probs.get(current_word)

        if not next_words:
            return None

        return random.choices(
            list(next_words.keys()),
            weights=list(next_words.values()),
            k=1
        )[0]

    def generate_text(self, start_word, length=20):
        current_word = start_word.lower()
        generated_words = [current_word]

        for _ in range(length - 1):
            next_word = self.generate_next_word(current_word)

            if next_word is None:
                break

            generated_words.append(next_word)
            current_word = next_word

        return " ".join(generated_words)

    def get_bigram_probs(self):
        return {
            word1: dict(next_words)
            for word1, next_words in self.bigram_probs.items()
        }

    def get_vocab(self):
        return self.vocab

    def get_bigram_matrix(self):
        matrix = {}

        for word1 in self.vocab:
            matrix[word1] = {}

            for word2 in self.vocab:
                matrix[word1][word2] = self.bigram_probs.get(word1, {}).get(word2, 0)

        return matrix