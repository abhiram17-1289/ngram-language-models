# Trains the tokenizer and one LM per author with the final config. I wrote this.

from tokenizer import BPETokenizer
from ngram_language_model import NGramLM
from author_identification import AuthorIdentifier, read_text, split_sentences

AUTHORS = {
    "Tolkien": "data/hobbit.txt",
    "Doyle":   "data/lostworld.txt",
}
TEST_FILE     = "data/HW2-F26-testset.txt"
OUTPUT_FILE   = "predictions.txt"
ORDER         = 3          # character trigram
K             = 1
VOCAB_SIZE    = 200
PRE_TOKENIZER = "bytelevel"
SEGMENTATION  = "sentence"


def cap_to_equal_length(author_texts):
    # truncate both authors to the shorter length so neither model gets more data
    n = min(len(t) for t in author_texts.values())
    return {name: text[:n] for name, text in author_texts.items()}


def train_models(authors=AUTHORS, vocab_size: int = VOCAB_SIZE,
                 pre_tokenizer: str = PRE_TOKENIZER, order: int = ORDER,
                 k: float = K, segmentation: str = SEGMENTATION):
    # read and equalize lengths before training
    author_texts = {name: read_text(path) for name, path in authors.items()}
    author_texts = cap_to_equal_length(author_texts)

    # tokenizer trains on the full files, LMs count over the capped text
    tokenizer = BPETokenizer.train(list(authors.values()), vocab_size, pre_tokenizer)

    classifier = AuthorIdentifier(tokenizer, order, k, segmentation)
    classifier.fit(author_texts)
    return tokenizer, classifier, author_texts

def label_test_file(clf, test_path, output_path):
    with open(test_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.rstrip("\n")
            if not line.strip():
                continue                      # skip blank lines
            item_id, text = line.split(None, 1)   # split on first whitespace run
            author = clf.predict(text)
            f_out.write(f"{item_id}\t{author}\n")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    tok, clf, author_texts = train_models()
    print(f"Trained tokenizer (vocab={tok.vocab_size}) and "
          f"{len(clf.models)} author models: {', '.join(clf.models)}")

    if TEST_FILE:
        label_test_file(clf, TEST_FILE, OUTPUT_FILE)