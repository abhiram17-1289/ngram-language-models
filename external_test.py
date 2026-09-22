"""External validation: does the classifier generalize to DIFFERENT books?

Train on the original two books, then classify passages from new book(s) by the
same authors. Reports per-author accuracy AND balanced accuracy (the average of
the two), which -- unlike a one-sided rate -- can't be gamed by a model that
just leans toward one author. Optimize settings against BALANCED accuracy.

Set NEW_BOOKS to whichever held-out books you have (one or both).
"""

from typing import Dict, List

from tokenizer import BPETokenizer
from author_identification import AuthorIdentifier, read_text, split_sentences

# --- CONFIG: your best setting ---
AUTHORS = {
    "Tolkien": "data/hobbit.txt",
    "Doyle":   "data/lostworld.txt",
}
ORDER          = 2
K              = 1
VOCAB_SIZE     = 2000
PRE_TOKENIZER  = "bytelevel"
SEGMENTATION   = "sentence"

# New held-out books, mapping file -> true author. Include one or both.
NEW_BOOKS: Dict[str, str] = {
    "data/sherlockholmes.txt": "Doyle",
    "data/fellowship.txt":     "Tolkien",
}
SENTENCES_PER_PASSAGE = 4
SHOW_EXAMPLES = 10          # print this many example predictions per book
# ---------------------------------


def make_passages(text: str, n_sentences: int) -> List[str]:
    """Group consecutive sentences into passages of ~n_sentences each."""
    sents = split_sentences(text)
    return [" ".join(sents[i:i + n_sentences])
            for i in range(0, len(sents), n_sentences)]

def cap_to_equal_length(author_texts: Dict[str, str]) -> Dict[str, str]:
    """Truncate every author's text to the length of the shortest, so no model
    gets a data-volume advantage (which otherwise biases perplexity scales)."""
    n = min(len(t) for t in author_texts.values())
    return {name: text[:n] for name, text in author_texts.items()}


def evaluate_book(clf, path: str, true_author: str):
    """Classify every passage in one book; return (correct, total)."""
    passages = make_passages(read_text(path), SENTENCES_PER_PASSAGE)

    if SHOW_EXAMPLES:
        print(f"\n  examples from {path}:")
        for p in passages[:SHOW_EXAMPLES]:
            print(f"    {clf.predict(p):8s} {clf.scores(p)}")

    correct = sum(clf.predict(p) == true_author for p in passages)
    return correct, len(passages)


def main():
    # Train the classifier on the ORIGINAL two books.
    author_texts = {name: read_text(path) for name, path in AUTHORS.items()}
    author_texts = cap_to_equal_length(author_texts) 
    tok = BPETokenizer.train(list(AUTHORS.values()),
                             vocab_size=VOCAB_SIZE, pre_tokenizer=PRE_TOKENIZER)
    clf = AuthorIdentifier(tok, order=ORDER, k=K, segmentation=SEGMENTATION)
    clf.fit(author_texts)

    print(f"Config: order={ORDER}, k={K}, vocab={VOCAB_SIZE}, "
          f"seg={SEGMENTATION}, sents/passage={SENTENCES_PER_PASSAGE}")

    # Evaluate each held-out book, collecting per-author rates.
    per_author_rate = {}
    pooled_correct = 0
    pooled_total = 0
    for path, true_author in NEW_BOOKS.items():
        correct, total = evaluate_book(clf, path, true_author)
        rate = correct / total if total else 0.0
        per_author_rate.setdefault(true_author, []).append(rate)
        pooled_correct += correct
        pooled_total += total
        print(f"\n{true_author:8s} ({path}): {correct}/{total} = {rate:.3f}")

    # Balanced accuracy = mean of per-author rates (unbiased by class sizes).
    author_means = {a: sum(rs) / len(rs) for a, rs in per_author_rate.items()}
    balanced = sum(author_means.values()) / len(author_means)
    pooled = pooled_correct / pooled_total if pooled_total else 0.0

    print("\n--- summary ---")
    for author, rate in author_means.items():
        print(f"  {author:8s} accuracy: {rate:.3f}")
    print(f"  balanced accuracy: {balanced:.3f}   <-- optimize THIS")
    print(f"  pooled accuracy:   {pooled:.3f}")
    if len(author_means) < 2:
        print("\n  NOTE: only one author tested -- this is one-sided recall,")
        print("  not balanced accuracy. Add the other book for a real number.")


if __name__ == "__main__":
    main()