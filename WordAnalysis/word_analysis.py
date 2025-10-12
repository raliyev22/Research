#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from typing import Tuple, Iterable, List

from tqdm import tqdm

"""
Usage:
python WordAnalysis/word_analysis.py \
  --input TEM_Phrapased/TR1.tsv \
  --output WordAnalysis/TR1_word_distributions.txt \
  --min_total 10 \
  --sort_key ratio \
  --include_ties
"""

# ====== Stopwords (NLTK) ======
from nltk.corpus import stopwords
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    import nltk
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))
# Keep parity with your other tools
stop_words.add("br")

# ====== Tokenizer rules (IDENTICAL policy to chi-square) ======
# Strip HTML tags
HTML_TAG_RE = re.compile(r"<[^>]+>")
# Keep alphabetic tokens with INTERNAL hyphens only (no apostrophes, no digits)
# Examples matched: "excellent", "heart-warming", "state-of-the-art"
WORD_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")

def _pre_normalize(text: str) -> str:
    """
    Apply the same normalization as chi-square:
      1) Unicode NFKC + casefold
      2) Strip HTML tags
      3) Unify curly apostrophes to ASCII
      4) Normalize possessives: heart's -> heart ; hearts' -> hearts
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    text = HTML_TAG_RE.sub(" ", text)
    text = text.replace("’", "'")
    text = re.sub(r"\b([a-z]+)'s\b", r"\1", text)   # John's -> John
    text = re.sub(r"\b([a-z]+)s'\b", r"\1s", text)  # dogs'  -> dogs
    return text

def tokenize_regex(text: str) -> List[str]:
    """
    Returns tokens that are purely alphabetic and may include INTERNAL hyphens.
    Contractions remain split as artifacts (e.g., don't -> 'don','t') by design,
    matching the chi-square script's behavior (since hyphens are preserved, but
    apostrophes are not part of WORD_RE).
    """
    text = _pre_normalize(text)
    return [m.group(0) for m in WORD_RE.finditer(text)]

# ====== Stats helpers ======
def skew_and_ratio(pos: int, neg: int) -> Tuple[str, float, float, float]:
    total = pos + neg
    if total == 0:
        return ("tie", 0.0, 0.0, 0.0)
    pos_r = pos / total
    neg_r = neg / total
    if pos_r > neg_r:
        return ("positive", pos_r, pos_r, neg_r)
    elif neg_r > pos_r:
        return ("negative", neg_r, pos_r, neg_r)
    else:
        return ("tie", pos_r, pos_r, neg_r)

def main():
    ap = argparse.ArgumentParser(description="Write word distributions split by positive/negative skew.")
    ap.add_argument("--input", "-i", required=True, help="Input TSV (label[TAB]text)")
    ap.add_argument("--output", "-o", required=True, help="Output .txt path")
    ap.add_argument("--min_total", type=int, default=0,
                    help="Minimum UNIQUE total (pos+neg) to include (default: 0)")
    ap.add_argument("--include_ties", action="store_true",
                    help="Also output words where pos_ratio == neg_ratio")
    ap.add_argument("--sort_key", choices=["ratio", "total"], default="ratio",
                    help="Sort within sections by 'ratio' (desc) then total, or by 'total' (desc) then ratio. Default: ratio.")
    args = ap.parse_args()

    unique_pos      = defaultdict(int)
    unique_neg      = defaultdict(int)
    non_unique_pos  = defaultdict(int)
    non_unique_neg  = defaultdict(int)

    N_docs = N_pos = N_neg = 0

    with open(args.input, "r", encoding="utf-8") as fin:
        lines = fin.readlines()

    for line in tqdm(lines, desc="Processing reviews"):
        line = line.strip()
        if not line:
            continue
        try:
            label, text = line.split("\t", 1)
        except ValueError:
            continue

        label = label.strip().casefold()
        # --- Use the SAME tokenizer pipeline as chi-square ---
        tokens = [t for t in tokenize_regex(text) if t and t not in stop_words]
        if not tokens:
            continue

        N_docs += 1
        if label == "pos":
            N_pos += 1
        elif label == "neg":
            N_neg += 1
        else:
            continue

        unique_tokens = set(tokens)
        freq = Counter(tokens)

        if label == "pos":
            for tok in unique_tokens:
                unique_pos[tok] += 1
            for tok, cnt in freq.items():
                non_unique_pos[tok] += cnt
        else:  # neg
            for tok in unique_tokens:
                unique_neg[tok] += 1
            for tok, cnt in freq.items():
                non_unique_neg[tok] += cnt

    # Build rows
    vocab = set(unique_pos) | set(unique_neg) | set(non_unique_pos) | set(non_unique_neg)
    pos_rows, neg_rows, tie_rows = [], [], []

    for w in vocab:
        u_pos = unique_pos.get(w, 0)
        u_neg = unique_neg.get(w, 0)
        nu_pos = non_unique_pos.get(w, 0)
        nu_neg = non_unique_neg.get(w, 0)

        u_skew, u_ratio, u_pos_r, u_neg_r = skew_and_ratio(u_pos, u_neg)
        nu_skew, nu_ratio, nu_pos_r, nu_neg_r = skew_and_ratio(nu_pos, nu_neg)

        u_total = u_pos + u_neg
        nu_total = nu_pos + nu_neg

        if u_total < args.min_total:
            continue

        row = {
            "word": w,
            "u_total": u_total, "u_pos": u_pos, "u_neg": u_neg,
            "u_skew": u_skew, "u_ratio": u_ratio, "u_pos_r": u_pos_r, "u_neg_r": u_neg_r,
            "nu_total": nu_total, "nu_pos": nu_pos, "nu_neg": nu_neg,
            "nu_skew": nu_skew, "nu_ratio": nu_ratio, "nu_pos_r": nu_pos_r, "nu_neg_r": nu_neg_r
        }

        if u_skew == "positive":
            pos_rows.append(row)
        elif u_skew == "negative":
            neg_rows.append(row)
        else:
            tie_rows.append(row)

    # Sorting within sections (descending)
    if args.sort_key == "ratio":
        pos_rows.sort(key=lambda r: (-r["u_ratio"], -r["u_total"], r["word"]))
        neg_rows.sort(key=lambda r: (-r["u_ratio"], -r["u_total"], r["word"]))
        tie_rows.sort(key=lambda r: (-r["u_total"], r["word"]))
    else:  # total-first
        pos_rows.sort(key=lambda r: (-r["u_total"], -r["u_ratio"], r["word"]))
        neg_rows.sort(key=lambda r: (-r["u_total"], -r["u_ratio"], r["word"]))
        tie_rows.sort(key=lambda r: (-r["u_total"], r["word"]))

    # Write TXT
    with open(args.output, "w", encoding="utf-8") as fout:
        fout.write(f"# Docs: {N_docs} | Pos: {N_pos} | Neg: {N_neg}\n")
        fout.write("# Sections: POSITIVELY-SKEWED (UNIQUE), NEGATIVELY-SKEWED (UNIQUE)")
        if args.include_ties:
            fout.write(", TIES (UNIQUE)")
        fout.write("\n# Format per line:\n")
        fout.write("# word: UNIQUE total=.., pos=.., neg=.., skew=.., ratio=.. | "
                   "NON-UNIQUE total=.., pos=.., neg=.., skew=.., ratio=..\n\n")

        def write_section(title: str, data):
            fout.write(f"=== {title} ===\n")
            for r in data:
                fout.write(
                    f"{r['word']}: "
                    f"UNIQUE total={r['u_total']}, pos={r['u_pos']}, neg={r['u_neg']}, "
                    f"skew={r['u_skew']}, ratio={r['u_ratio']:.2f} | "
                    f"NON-UNIQUE total={r['nu_total']}, pos={r['nu_pos']}, neg={r['nu_neg']}, "
                    f"skew={r['nu_skew']}, ratio={r['nu_ratio']:.2f}\n"
                )
            fout.write("\n")

        write_section("POSITIVELY-SKEWED (UNIQUE)", pos_rows)
        write_section("NEGATIVELY-SKEWED (UNIQUE)", neg_rows)
        if args.include_ties and tie_rows:
            write_section("TIES (UNIQUE)", tie_rows)

    print(f"\n✅ Wrote {len(pos_rows)} positive, {len(neg_rows)} negative"
          + (f", {len(tie_rows)} ties" if args.include_ties else "")
          + f" to: {args.output}")

if __name__ == "__main__":
    main()
