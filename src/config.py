"""
Shared settings and helpers for every group.

Import it from any notebook in notebooks/:

    import sys
    sys.path.append("..")          # lets Python see the src/ folder
    from src.config import *

Change a value HERE (and tell the team), never by retyping it in a notebook.
"""
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR  = PROJECT_ROOT / "results"

# ---------------------------------------------------------------- data
SPLITS = ["train", "validation", "test"]

# (rows, label-1 count, label-0 count) of the published MRPC splits
EXPECTED_COUNTS = {
    "train":      (3668, 2474, 1194),
    "validation": (408,  279,  129),
    "test":       (1725, 1147, 578),
}

# ---------------------------------------------------------------- labels
POSITIVE_LABEL = 1                     # 1 = paraphrase, 0 = not paraphrase
LABEL_NAMES    = {0: "not_paraphrase", 1: "paraphrase"}

# ---------------------------------------------------------------- randomness
SEEDS = [42, 43, 44]                   # use these wherever a model has randomness

# ---------------------------------------------------------------- evaluation
THRESHOLD    = 0.5                     # prob_paraphrase >= THRESHOLD -> predict 1
PRED_COLUMNS = ["idx", "label", "prob_paraphrase", "pred"]

# ---------------------------------------------------------------- Groups 1 & 2 only
TOKENIZER = "treebank"                 # nltk TreebankWordTokenizer
LOWERCASE = True
MIN_COUNT = 2                          # train tokens seen < 2 times -> <UNK>


# ================================================================ helpers
def load_split(name):
    """Load one split from data/raw and check it matches the published counts."""
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    got = (len(df), int((df.label == 1).sum()), int((df.label == 0).sum()))
    if got != EXPECTED_COUNTS[name]:
        raise ValueError(f"{name}: got {got}, expected {EXPECTED_COUNTS[name]}")
    return df


def load_all():
    """Return train, validation, test DataFrames."""
    return tuple(load_split(s) for s in SPLITS)


def evaluate(labels, preds):
    """The one scoring function everyone uses."""
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, pos_label=POSITIVE_LABEL),
    }


def save_predictions(df, probs, path):
    """
    Save predictions in the shared format: idx, label, prob_paraphrase, pred.
    df    = the split's DataFrame (needs idx and label)
    probs = P(paraphrase) for each row, same order as df
    path  = e.g. RESULTS_DIR / "group2_logreg" / "test_seed42.csv"
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({
        "idx": df["idx"].values,
        "label": df["label"].values,
        "prob_paraphrase": [round(float(p), 6) for p in probs],
    })
    out["pred"] = (out["prob_paraphrase"] >= THRESHOLD).astype(int)
    out[PRED_COLUMNS].to_csv(path, index=False)
    return evaluate(out["label"], out["pred"])
