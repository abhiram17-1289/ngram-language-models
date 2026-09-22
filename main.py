from tokenizer import BPETokenizer
from ngram_language_model import NGramLM
from author_identification import AuthorIdentifier, read_text, split_sentences

AUTHORS = {
    "Tolkien": "data/hobbit.txt",
    "Doyle":   "data/lostworld.txt",
}
TEST_FILE    = None
ORDER        = 3      # 2 = bigram, 3 = trigram
K            = 1 # add-k smoothing
VOCAB_SIZE   = 200
PRE_TOKENIZER= "bytelevel"     # or "whitespace"
SEGMENTATION = "sentence"      # or "stream"

def cap_to_equal_length(author_texts):
    """Truncate every author's text to the length of the shortest, so no model
    gets a data-volume advantage (which otherwise biases perplexity scales)."""
    n = min(len(t) for t in author_texts.values())
    return {name: text[:n] for name, text in author_texts.items()}

def train_models(
        authors = AUTHORS,
        vocab_size : int = VOCAB_SIZE,
        pre_tokenizer : str = PRE_TOKENIZER,
        order : int = ORDER,
        k : float = K,
        segmentation : str = SEGMENTATION
):
    author_texts = {name: read_text(path) for name, path in AUTHORS.items()}
    author_texts = cap_to_equal_length(author_texts) 

    tokenizer = BPETokenizer.train(
        list(authors.values()), vocab_size, pre_tokenizer
    )

    classifier = AuthorIdentifier(tokenizer, order, k, segmentation)
    classifier.fit(author_texts)

    return tokenizer, classifier, author_texts

if __name__ == "__main__":
    tok, clf, author_texts = train_models()
    print(f"Trained tokenizer (vocab={tok.vocab_size}) and "
          f"{len(clf.models)} author models: {', '.join(clf.models)}")
