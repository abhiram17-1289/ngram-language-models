import re
from tokenizer import BPETokenizer
from ngram_language_model import NGramLM

def read_text(path: str):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()

_ABBREVIATIONS = ["Mr", "Mrs", "Ms", "Dr", "Prof", "St", "Sr", "Jr", "vs",
                  "etc", "Capt", "Lt", "Col", "Gen", "Rev", "Hon", "Messrs",
                  "No", "Vol", "Fig"]

_ABBR_GUARD = "".join(rf"(?<!\b{a}\.)" for a in _ABBREVIATIONS)

_SENT_PATTERN = re.compile(
    _ABBR_GUARD                                 # don't split after "Mr." etc.
    + r"(?:(?<=[.!?])|(?<=[.!?][\"'”’)\]]))"    # punct, or punct + one closer
    + r"\s+"                                     # the whitespace to cut at
    + r"(?=[\"'“‘(]?[A-Z0-9])"                  # next thing starts a sentence
)

def split_sentences(text: str):
    """Split prose into sentences, keeping terminal punctuation, tolerating
    closing quotes, and not splitting on common title abbreviations."""
    parts = _SENT_PATTERN.split(text.strip())
    return [re.sub(r"\s+", " ", p).strip() for p in parts if p and p.strip()]

class AuthorIdentifier:
    def __init__(self, tokenizer : BPETokenizer, order : int, k : float, segmentation : str = "sentence"):
        if segmentation not in ("sentence", "stream"):
            raise ValueError(f"Unknown Segmentation Format : {segmentation!r}. Available options : ['sentence', 'stream']")
        self.tokenizer = tokenizer
        self.order = order
        self.k = k
        self.segmentation = segmentation
        self.models = {}
        

    def _encode(self, text : str):
        if self.segmentation == "sentence":
            return [self.tokenizer.encode_to_ids(s) for s in split_sentences(text)]

        return [self.tokenizer.encode_to_ids(text)]

    def fit(self, author_texts):
        for name, text in author_texts.items():
            language_model = NGramLM(self.order, self.tokenizer.vocab_size, self.tokenizer.bos_id, self.tokenizer.eos_id)
            language_model.train(self._encode(text))
            self.models[name] = language_model

    def scores(self, passage: str):
        sequences = self._encode(passage)

        return {
            name : language_model.perplexity(sequences, k = self.k)
            for name, language_model in self.models.items()
        }

    def predict(self, passage: str):
        perplexities = self.scores(passage)
        return min(perplexities, key = perplexities.get)

