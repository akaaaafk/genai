import spacy


class EmbeddingModel:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")

    def get_embedding(self, word: str):
        doc = self.nlp(word)

        if len(doc) == 0:
            return {
                "word": word,
                "has_vector": False,
                "vector_length": 0,
                "embedding": []
            }

        token = doc[0]

        return {
            "word": word,
            "has_vector": token.has_vector,
            "vector_length": len(token.vector),
            "embedding": token.vector.tolist()
        }