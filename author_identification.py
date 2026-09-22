# Author ID using one LM per author. I wrote this; used Claude to improve the
# sentence-splitting regex.

import re
from tokenizer import BPETokenizer
from ngram_language_model import NGramLM


def read_text(path: str):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


# each abbreviation becomes a lookbehind so we don't split after "Mr." etc
_ABBREVIATIONS = ["Mr", "Mrs", "Ms", "Dr", "Prof", "St", "Sr", "Jr", "vs",
                  "etc", "Capt", "Lt", "Col", "Gen", "Rev", "Hon", "Messrs",
                  "No", "Vol", "Fig"]
_ABBR_GUARD = "".join(rf"(?<!\b{a}\.)" for a in _ABBREVIATIONS)

_SENT_PATTERN = re.compile(
    _ABBR_GUARD                                 # skip abbreviations
    + r"(?:(?<=[.!?])|(?<=[.!?][\"'”’)\]]))"    # punct, or punct plus one closer
    + r"\s+"                                     # split on the space
    + r"(?=[\"'“‘(]?[A-Z0-9])"                  # next looks like a new sentence
)


def split_sentences(text: str):
    # heuristic splitter, keeps punctuation, tolerates quotes and abbreviations
    parts = _SENT_PATTERN.split(text.strip())
    # collapse hard-wrap newlines and drop blanks
    return [re.sub(r"\s+", " ", p).strip() for p in parts if p and p.strip()]


class AuthorIdentifier:
    def __init__(self, tokenizer: BPETokenizer, order: int, k: float, segmentation: str = "sentence"):
        if segmentation not in ("sentence", "stream"):
            raise ValueError(f"Unknown Segmentation Format : {segmentation!r}. "
                             f"Available options : ['sentence', 'stream']")
        self.tokenizer = tokenizer
        self.order = order
        self.k = k
        self.segmentation = segmentation
        self.models = {}

    def _encode(self, text: str):
        # sentence mode: one sequence per sentence
        if self.segmentation == "sentence":
            return [self.tokenizer.encode_to_ids(s) for s in split_sentences(text)]
        # stream mode: whole text as one sequence
        return [self.tokenizer.encode_to_ids(text)]

    def fit(self, author_texts):
        # one model per author, shared vocabulary
        for name, text in author_texts.items():
            language_model = NGramLM(self.order, self.tokenizer.vocab_size,
                                     self.tokenizer.bos_id, self.tokenizer.eos_id)
            language_model.train(self._encode(text))
            self.models[name] = language_model

    def scores(self, passage: str):
        # perplexity under each author, lower is better
        sequences = self._encode(passage)
        return {name: lm.perplexity(sequences, k=self.k)
                for name, lm in self.models.items()}

    def predict(self, passage: str):
        # pick the least surprised model
        perplexities = self.scores(passage)
        return min(perplexities, key=perplexities.get)