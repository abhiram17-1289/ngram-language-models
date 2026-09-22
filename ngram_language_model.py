# N-gram model of any order with add-k smoothing. I wrote this; used Claude for
# the concepts and to generalize my bigram/trigram code to any order.

import math
from collections import Counter


class NGramLM:
    def __init__(self, order: int, vocab_size: int, bos_id: int, eos_id: int):
        self.order = order
        self.vocab_size = vocab_size
        self.bos_id = bos_id
        self.eos_id = eos_id

        # order N needs the n-grams and their contexts
        self.ngram_counts: Counter = Counter()
        self.context_counts: Counter = Counter()

    def _pad(self, sequence):
        # start tokens for context, one end token
        return [self.bos_id] * (self.order - 1) + list(sequence) + [self.eos_id]

    def train(self, sequences):
        # training is just counting
        for sequence in sequences:
            if not sequence:                 # skip empties
                continue
            padded_sequence = self._pad(sequence)
            for i in range(self.order - 1, len(padded_sequence)):
                ngram = tuple(padded_sequence[i + 1 - self.order: i + 1])
                context = ngram[:-1]
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1

    def _logprob(self, ngram, k):
        # add-k so unseen n-grams are not zero
        context = ngram[:-1]
        numerator = self.ngram_counts[ngram] + k
        denominator = self.context_counts[context] + k * self.vocab_size
        return math.log(numerator / denominator)

    def sequence_negative_logprob(self, sequence, k):
        # sum -log P in log space to avoid underflow
        padded_sequence = self._pad(sequence)
        negative_logprob = 0
        num_tokens = 0
        # start at order-1 so the BOS padding is context only
        for i in range(self.order - 1, len(padded_sequence)):
            ngram = tuple(padded_sequence[i + 1 - self.order: i + 1])
            negative_logprob += -self._logprob(ngram, k)
            num_tokens += 1
        return negative_logprob, num_tokens

    def perplexity(self, sequences, k):
        # per-token average, exponentiated
        total_negative_logprob = 0
        total_num_tokens = 0
        for sequence in sequences:
            if not sequence:
                continue
            negative_logprob, num_tokens = self.sequence_negative_logprob(sequence, k)
            total_negative_logprob += negative_logprob
            total_num_tokens += num_tokens
        if total_num_tokens == 0:
            return float('inf')
        return math.exp(total_negative_logprob / total_num_tokens)