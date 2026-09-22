# BPE tokenizer wrapping HuggingFace. I wrote this; used Claude to understand
# why HuggingFace splits the model from the trainer.

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, ByteLevel

# UNK is the encode-time fallback, BOS/EOS mark sentence beginning/ending boundaries
UNK = "[UNK]"
BOS = "[BOS]"
EOS = "[EOS]"
special_tokens = [UNK, BOS, EOS]


class BPETokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self._tokenizer = tokenizer

    @classmethod
    def train(cls, files, vocab_size: int = 10000, pre_tokenizer: str = "whitespace"):
        # build a fresh tokenizer and return it wrapped
        tokenizer = Tokenizer(BPE(unk_token=UNK))

        # rough first split before BPE merges
        if pre_tokenizer == "bytelevel":
            tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
        elif pre_tokenizer == "whitespace":
            tokenizer.pre_tokenizer = Whitespace()
        else:
            raise ValueError(f"Unknown Pre Tokenizer Specified : {pre_tokenizer}")

        trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=vocab_size)
        tokenizer.train(files, trainer)
        return cls(tokenizer)

    #the models count over ids
    def encode_to_ids(self, text: str):
        return self._tokenizer.encode(text).ids

    # string tokens, for debugging
    def encode_to_tokens(self, text: str):
        return self._tokenizer.encode(text).tokens

    @property
    def vocab_size(self):
        return self._tokenizer.get_vocab_size()

    def token_to_id(self, token):
        return self._tokenizer.token_to_id(token)

    # the LMs pad with these ids
    @property
    def bos_id(self):
        return self._tokenizer.token_to_id(BOS)

    @property
    def eos_id(self):
        return self._tokenizer.token_to_id(EOS)

    #to save and load trained models
    def save(self, path: str):
        self._tokenizer.save(path)

    @classmethod
    def load(cls, path: str):
        return cls(Tokenizer.from_file(path))