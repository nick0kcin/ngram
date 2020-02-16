from nltk import PunktSentenceTokenizer as SentenceTokenizer
from nltk import TweetTokenizer as Tokenizer


class LanguageModel:
    """
    N-gram model
    """
    def __init__(self, n_gram=2, missed_value=0.99):
        """

        :param n_gram: length of n-gram
        :param missed_value: default value for all unseen n-gram
        """
        self.n = n_gram
        self.n_grams = {}
        self.context = {}
        self.sentence_tokenizer = SentenceTokenizer()
        self.tokenizer = Tokenizer()
        self.missed_value = missed_value

    def build_model(self, text):
        sentenses = self.sentence_tokenizer.tokenize(text)
        words = [
            list(
                filter(
                    lambda s: s.isalpha(),
                    self.tokenizer.tokenize(sentence.strip())
                )
            ) for sentence in sentenses
        ]
        for sentence in words:
            if len(sentence) < self.n:
                key = " ".join(sentence)
                self.context.update({key: self.context.get(key, 0) + 1})
            else:
                for i in range(len(sentence) - self.n + 1):
                    context_key = " ".join(sentence[i:i + self.n - 1])
                    n_gram_key = " ".join(sentence[i:i + self.n])
                    self.context.update({context_key: self.context.get(context_key, 0) + 1})
                    self.n_grams.update({n_gram_key: self.n_grams.get(n_gram_key, 0) + 1})

    def calculate_proba(self, sentence):
        words = list(
            filter(
                lambda s: s.isalpha(),
                self.tokenizer.tokenize(sentence.strip())
            )
        )
        result = 1
        for i in range(min(self.n - 2, len(words) - 1), len(words)):
            if i < self.n - 1:
                size = sum([val for key, val in self.context.items() if len(key.split(" ")) == i+1])
                result *= self.context.get(" ".join(words[:i+1]), self.missed_value if i == self.n - 2 else 0) / size
            elif i > self.n - 2:
                context_key = " ".join(words[i-self.n+1:i])
                n_gram_key = " ".join(words[i-self.n+1:i+1])
                context_val = self.context.get(context_key, self.missed_value)
                n_gram_val = self.n_grams.get(n_gram_key, self.missed_value)
                p = n_gram_val / context_val
                result *= p
        return result
