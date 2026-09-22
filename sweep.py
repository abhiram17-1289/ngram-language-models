"""Sweep hyperparameters and plot classification accuracy (mean +/- std).

 Produces three plots:
  1. accuracy vs vocab size, one line per order          -> sweep_vocab.png
  2. accuracy vs add-k smoothing, one line per order      -> sweep_k.png
  3. accuracy vs order, sentence vs stream segmentation   -> sweep_segmentation.png

"""

import random
from statistics import mean, pstdev
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")            # write files without needing a display
import matplotlib.pyplot as plt

from tokenizer import BPETokenizer
from ngram_language_model import NGramLM
from author_identification import read_text, split_sentences

# --- CONFIG ---
AUTHORS: Dict[str, str] = {
    "Tolkien": "data/hobbit.txt",
    "Doyle":   "data/lostworld.txt",
}
PRE_TOKENIZER = "bytelevel"
N_FOLDS       = 5

# plot 1: vocab x order, k held fixed
VOCAB_GRID    = [200, 500, 1000, 2000, 5000, 10000]
ORDER_GRID    = [1, 2, 3]
K_FIXED       = 0.1

# plot 2: k x order, vocab held fixed
K_GRID        = [1.0, 0.5, 0.1, 0.01, 0.001]
VOCAB_FIXED   = 2000

# plot 3: segmentation (sentence vs stream) x order, vocab & k held fixed
SEG_VOCAB     = 2000
SEG_K         = 0.1
# --------------


# ---------------------------------------------------------------------------
# Cross-validation 
# ---------------------------------------------------------------------------
def make_folds(items: List[str], n_folds: int, seed: int = 0) -> List[List[str]]:
    """Shuffle items reproducibly, then split into n_folds balanced folds."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if n_folds > len(items):
        raise ValueError(f"n_folds ({n_folds}) exceeds items ({len(items)})")
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    folds: List[List[str]] = [[] for _ in range(n_folds)]
    for i, item in enumerate(shuffled):
        folds[i % n_folds].append(item)
    return folds


def _training_sentences(folds: List[List[str]], held_out: int) -> List[str]:
    return [s for j, fold in enumerate(folds) if j != held_out for s in fold]


def _train_lm(tok: BPETokenizer, order: int, sentences: List[str],
              segmentation: str = "sentence") -> NGramLM:
    """Train one LM. In 'sentence' mode each sentence is its own sequence; in
    'stream' mode all training sentences are concatenated into one continuous
    token sequence (so only one BOS/EOS pair, and sentences flow into each
    other)."""
    lm = NGramLM(order, tok.vocab_size, tok.bos_id, tok.eos_id)
    if segmentation == "stream":
        stream = [t for s in sentences for t in tok.encode_to_ids(s)]
        lm.train([stream])
    else:
        lm.train([tok.encode_to_ids(s) for s in sentences])
    return lm


def cross_validate_accuracy(
    tok: BPETokenizer,
    author_texts: Dict[str, str],
    order: int = 3,
    k_smooth: float = 0.01,
    n_folds: int = 5,
    seed: int = 0,
    segmentation: str = "sentence",
):
    """5-fold classification accuracy. Training sequences are formed per the
    `segmentation` mode; test items are always individual held-out sentences,
    so accuracy stays comparable across modes.

    Returns (mean_accuracy, std_accuracy, per_fold_accuracies).
    """
    author_folds = {
        name: make_folds(split_sentences(text), n_folds, seed)
        for name, text in author_texts.items()
    }

    per_fold_acc = []
    for i in range(n_folds):
        models = {
            name: _train_lm(tok, order, _training_sentences(folds, i), segmentation)
            for name, folds in author_folds.items()
        }
        correct = 0
        total = 0
        for true_author, folds in author_folds.items():
            for sentence in folds[i]:
                seq = [tok.encode_to_ids(sentence)]
                pred = min(models,
                           key=lambda name: models[name].perplexity(seq, k=k_smooth))
                correct += int(pred == true_author)
                total += 1
        per_fold_acc.append(correct / total if total else 0.0)

    return mean(per_fold_acc), pstdev(per_fold_acc), per_fold_acc


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------
def _load_texts() -> Dict[str, str]:
    return {name: read_text(path) for name, path in AUTHORS.items()}


def sweep_vocab(author_texts):
    results = {order: {"mean": [], "std": []} for order in ORDER_GRID}
    for vocab in VOCAB_GRID:
        print(f"[vocab sweep] tokenizer vocab={vocab} ...")
        tok = BPETokenizer.train(list(AUTHORS.values()),
                                 vocab_size=vocab, pre_tokenizer=PRE_TOKENIZER)
        for order in ORDER_GRID:
            m, s, _ = cross_validate_accuracy(
                tok, author_texts, order=order, k_smooth=K_FIXED, n_folds=N_FOLDS)
            results[order]["mean"].append(m)
            results[order]["std"].append(s)
            print(f"    order={order}: acc={m:.3f} +/- {s:.3f}")
    return results


def sweep_k(author_texts):
    print(f"[k sweep] tokenizer vocab={VOCAB_FIXED} ...")
    tok = BPETokenizer.train(list(AUTHORS.values()),
                             vocab_size=VOCAB_FIXED, pre_tokenizer=PRE_TOKENIZER)
    results = {order: {"mean": [], "std": []} for order in ORDER_GRID}
    for order in ORDER_GRID:
        for k in K_GRID:
            m, s, _ = cross_validate_accuracy(
                tok, author_texts, order=order, k_smooth=k, n_folds=N_FOLDS)
            results[order]["mean"].append(m)
            results[order]["std"].append(s)
            print(f"    order={order} k={k}: acc={m:.3f} +/- {s:.3f}")
    return results


def sweep_segmentation(author_texts):
    print(f"[segmentation sweep] tokenizer vocab={SEG_VOCAB} ...")
    tok = BPETokenizer.train(list(AUTHORS.values()),
                             vocab_size=SEG_VOCAB, pre_tokenizer=PRE_TOKENIZER)
    results = {seg: {"mean": [], "std": []} for seg in ("sentence", "stream")}
    for order in ORDER_GRID:
        for seg in ("sentence", "stream"):
            m, s, _ = cross_validate_accuracy(
                tok, author_texts, order=order, k_smooth=SEG_K,
                n_folds=N_FOLDS, segmentation=seg)
            results[seg]["mean"].append(m)
            results[seg]["std"].append(s)
            print(f"    order={order} seg={seg}: acc={m:.3f} +/- {s:.3f}")
    return results


def _plot(x_values, series, x_label, title, out_path, log_x=True):
    """series: dict of label -> {"mean": [...], "std": [...]}."""
    plt.figure(figsize=(8, 5))
    for label, data in series.items():
        plt.errorbar(x_values, data["mean"], yerr=data["std"],
                     marker="o", capsize=4, label=str(label))
    if log_x:
        plt.xscale("log")
    plt.xlabel(x_label)
    plt.ylabel("classification accuracy (5-fold)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"saved {out_path}")


def main():
    author_texts = _load_texts()

    vocab_results = sweep_vocab(author_texts)
    _plot(VOCAB_GRID, {f"order {o}": vocab_results[o] for o in ORDER_GRID},
          "vocab size", f"Accuracy vs vocab size (k={K_FIXED})", "plots/sweep_vocab.png")

    k_results = sweep_k(author_texts)
    _plot(K_GRID, {f"order {o}": k_results[o] for o in ORDER_GRID},
          "add-k smoothing (k)", f"Accuracy vs k (vocab={VOCAB_FIXED})", "plots/sweep_k.png")

    seg_results = sweep_segmentation(author_texts)
    _plot(ORDER_GRID, seg_results, "n-gram order",
          f"Sentence vs stream (vocab={SEG_VOCAB}, k={SEG_K})",
          "plots/sweep_segmentation.png", log_x=False)


if __name__ == "__main__":
    main()