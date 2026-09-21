import math
from collections import Counter

class NGramLM:
    def __init__(self, order : int, vocab_size : int, bos_id : int, eos_id : int):
        if order not in (2, 3):
            raise ValueError("Order must be 2(Bigram) or 3(Trigram)")
        
        self.order = order
        self.vocab_size = vocab_size
        self.bos_id = bos_id
        self.eos_id = eos_id

        self.unigrams : Counter = Counter()
        self.bigrams : Counter = Counter()
        self.trigrams : Counter = Counter()

    def _pad(self, ids):
        return [self.bos_id] * (self.order - 1) + list(ids) + [self.eos_id]

    def train(self, sequences):
        for sequence_ids in sequences:
            padded_sequence = self._pad(sequence_ids)
            for i in range(len(padded_sequence)):
                self.unigrams[padded_sequence[i]] += 1
                if i >= 1:
                    self.bigrams[(padded_sequence[i-1], padded_sequence[i])] += 1
                if i >= 2:
                    self.trigrams[(padded_sequence[i-2], padded_sequence[i-1], padded_sequence[i])] += 1

    def _bigram_logprob(self, prev : int, curr : int, k : float):
        numerator = self.bigrams[(prev, curr)] + k
        denominator = self.unigrams[prev] + (k * self.vocab_size)

        return math.log(numerator/denominator)

    def _trigram_logprob(self, curr_minus2 : int, curr_minus1 : int, curr : int, k : float):
        numerator = self.trigrams([curr_minus2, curr_minus1, curr]) + k
        denominator = self.bigrams([curr_minus2, curr_minus1]) + (k * self.vocab_size)

        return math.log(numerator/denominator)

    def sequence_negative_logprob(self, sequence, k : float = 1.0):
        padded_sequence = self._pad(sequence)

        negative_logprob = 0
        n = 0

        for i in range(self.order - 1, len(padded_sequence)):
            if self.order == 2:
                logprob = self._bigram_logprob(padded_sequence[i-1], padded_sequence[i], k)
            else:
                logprob = self._trigram_logprob(padded_sequence[i-2], padded_sequence[i-1], padded_sequence[i], k)

            negative_logprob += -logprob
            n += 1

        return negative_logprob, n

    def perplexity(self, sequences, k : float = 1):
        total_negative_logprob = 0
        total_n = 0

        for sequence in sequences:
            negative_logprob, n = self.sequence_negative_logprob(sequence, k)
            total_negative_logprob += negative_logprob
            total_n += n

        if total_n == 0:
            return float('inf')

        return math.exp(total_negative_logprob/total_n)
