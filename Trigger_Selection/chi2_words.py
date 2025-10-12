
# python Trigger_Selection/chi2_words.py TEM_Phrapased/TR1_priv_E3.0.tsv --min_df 400 --topk 40
# python Trigger_Selection/chi2_words.py TEM_Phrapased/TR1.tsv --only_queries for only specific words

# """
# p(w|pos)=0.1108 → About 11% of positive reviews contain “excellent”.

# p(w|neg)=0.0313 → Only 3% of negative reviews contain it.

# p(pos|w)=0.7793 → If a review contains “excellent”, there is a ~78% chance it is positive.

# χ² = 299.4 → Very strong evidence that “excellent” is not independent of sentiment → highly discriminative word.
# """


"""
Usage examples:
  # Ranked lists with top-20 per class, min df 400
  python Trigger_Selection/chi2_words.py TEM_Phrapased/TR1_priv_E3.0.tsv --min_df 400 --topk 20

  # Only show specific query words (set in QUERY_WORDS below)
  python Trigger_Selection/chi2_words.py TEM_Phrapased/TR1.tsv --only_queries

Tokenizer policy (as requested):
  1) Keep hyphenated words intact: "heart-warming" -> "heart-warming"
  2) Normalize possessives: "heart's" -> "heart", "hearts'" -> "hearts"
  3) Strip HTML tags: "I <br> love this" -> "I love this"
  4) Unicode normalize + casefold
  5) Drop punctuation & numbers: "excellent!" -> "excellent"; numbers removed
"""

import argparse, csv, re, sys, unicodedata
from collections import Counter
from typing import Tuple, Optional, List
from tqdm import tqdm

# ========== CONFIG: add your custom words here ==========
QUERY_WORDS = ["excellent", "heart", "heart-warming"]

# -----------------------------------------------------------------------------
# Normalization + tokenizer
# -----------------------------------------------------------------------------
# Strip HTML tags conservatively
HTML_TAG_RE = re.compile(r"<[^>]+>")
# Keep alphabetic tokens with INTERNAL hyphens only (no apostrophes, no digits)
# Examples matched: "excellent", "heart-warming", "state-of-the-art"
WORD_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")

def _pre_normalize(text: str) -> str:
    """Apply HTML stripping, Unicode NFKC, casefold, apostrophe unification, possessive fixes."""
    # Unicode normalize + casefold
    text = unicodedata.normalize("NFKC", text).casefold()
    # Strip HTML tags
    text = HTML_TAG_RE.sub(" ", text)
    # Unify curly apostrophes to ASCII
    text = text.replace("’", "'")
    # Normalize English possessives BEFORE tokenization:
    # heart's -> heart ; hearts' -> hearts
    text = re.sub(r"\b([a-z]+)'s\b", r"\1", text)
    text = re.sub(r"\b([a-z]+)s'\b", r"\1s", text)
    return text

def tokenize(text: str) -> List[str]:
    """
    Returns tokens that are purely alphabetic and may include INTERNAL hyphens.
    - Hyphens are preserved inside tokens (heart-warming),
      but punctuation and digits are otherwise dropped.
    """
    text = _pre_normalize(text)
    return [m.group(0) for m in WORD_RE.finditer(text)]

def normalize_single_word(raw: str) -> Optional[str]:
    """Normalize a single word through the same pipeline, return first token or None."""
    toks = tokenize(raw)
    return toks[0] if toks else None

# -----------------------------------------------------------------------------
# Small stopword list (optional) — keep consistent with your other script
# -----------------------------------------------------------------------------
DEFAULT_STOPWORDS = {
    "the" ,"a" ,"an" ,"and" ,"or" ,"but" ,"if" ,"in" ,"on" ,"at" ,"to" ,"from" ,"for" ,"of",
    "is" ,"am" ,"are" ,"was" ,"were" ,"be" ,"been" ,"being" ,"it" ,"this" ,"that" ,"as",
    "with" ,"by" ,"about" ,"into" ,"over" ,"after" ,"before" ,"between" ,"because",
    "while" ,"during" ,"so" ,"very" ,"too" ,"not" ,"no" ,"yes" ,"i" ,"you" ,"he" ,"she",
    "we" ,"they" ,"them" ,"me" ,"my" ,"your" ,"his" ,"her" ,"their" ,"our" ,"ours",
    # add more if you want exact parity with your other tool (e.g., "br")
}

# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def row_to_label_and_text(row: list) -> Tuple[str, str]:
    if not row: return "", ""
    label = row[0].strip().lower()
    text = " ".join(row[1:])
    return label, text

def iter_rows(path: str, delimiter: str):
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter=delimiter)
        for row in r:
            if row:
                yield row

# -----------------------------------------------------------------------------
# Core computation
# -----------------------------------------------------------------------------
def compute_chi2(
        path: str,
        delimiter: str = "\t",
        pos_label: str = "pos",
        neg_label: str = "neg",
        min_df: int = 20,
        use_stopwords: bool = True,
):
    N_docs = 0
    N_pos = 0
    N_neg = 0

    # Document-frequency maps
    df_pos = Counter()
    df_neg = Counter()
    df_total = Counter()

    stopwords = DEFAULT_STOPWORDS if use_stopwords else set()

    # Count per-doc presence
    for row in tqdm(iter_rows(path, delimiter), desc="Processing docs", unit="docs"):
        label, text = row_to_label_and_text(row)
        if label not in (pos_label, neg_label):
            continue

        N_docs += 1
        if label == pos_label: N_pos += 1
        else: N_neg += 1

        # SAME tokenizer policy as specified above
        tokens = set(t for t in tokenize(text) if t and t not in stopwords)
        for w in tokens:
            df_total[w] += 1
            if label == pos_label: df_pos[w] += 1
            else: df_neg[w] += 1

    if N_docs == 0:
        print("No documents found.", file=sys.stderr)
        sys.exit(1)

    # Build per-word stats (respecting min_df for the ranked lists)
    rows = []
    for w, dft in df_total.items():
        if dft < min_df:
            continue

        A = df_pos[w]          # pos docs containing w
        C = df_neg[w]          # neg docs containing w
        B = N_pos - A          # pos docs w/o w
        D = N_neg - C          # neg docs w/o w

        # Chi-square
        numerator = ( A *D - B* C) ** 2 * N_docs
        denominator = (A + C) * (B + D) * (A + B) * (C + D)
        chi2 = numerator / denominator if denominator > 0 else 0.0

        # Class-conditional rates (p(w|class))
        rate_pos = A / N_pos if N_pos else 0.0
        rate_neg = C / N_neg if N_neg else 0.0
        direction = "pos" if rate_pos > rate_neg else "neg" if rate_neg > rate_pos else "tie"

        rows.append((w, A, C, dft, chi2, rate_pos, rate_neg, direction))

    # Split and sort
    pos_rows = [r for r in rows if r[7] == "pos"]
    neg_rows = [r for r in rows if r[7] == "neg"]

    pos_rows.sort(key=lambda r: (-r[4], -r[1], r[0]))  # chi2 desc, df_pos desc, alpha
    neg_rows.sort(key=lambda r: (-r[4], -r[2], r[0]))  # chi2 desc, df_neg desc, alpha

    return {
        "N_docs": N_docs, "N_pos": N_pos, "N_neg": N_neg,
        "min_df": min_df,
        "pos_rows": pos_rows, "neg_rows": neg_rows, "all_rows": rows,
        # expose raw counters so we can report ANY word, even below min_df
        "df_pos_map": df_pos, "df_neg_map": df_neg, "df_total_map": df_total,
        "pos_label": pos_label, "neg_label": neg_label,
    }


# -----------------------------------------------------------------------------
# Formatting helpers
# -----------------------------------------------------------------------------
def format_one_word(w: str, res: dict) -> str:
    """Compact line for a single word using df maps (even if below min_df)."""
    w_norm = normalize_single_word(w)
    if not w_norm:
        return f"- '{w}': (no analyzable token)"

    A = res["df_pos_map"].get(w_norm, 0)
    C = res["df_neg_map"].get(w_norm, 0)
    dft = A + C

    if dft == 0:
        return f"{w_norm}: total=0 (not found)"

    N_pos, N_neg = res["N_pos"], res["N_neg"]

    # p(w|class)
    p_w_given_pos = A / N_pos if N_pos else 0.0
    p_w_given_neg = C / N_neg if N_neg else 0.0

    # p(class|w)
    p_pos_given_w = A / dft if dft else 0.0
    p_neg_given_w = C / dft if dft else 0.0

    # Chi-square using the same formula (with B,D)
    B = N_pos - A
    D = N_neg - C
    numerator = (A * D - B * C) ** 2 * (N_pos + N_neg)
    denominator = (A + C) * (B + D) * (A + B) * (C + D)
    chi2 = (numerator / denominator) if denominator > 0 else 0.0

    # skew + dominant ratio (share in that skew)
    if A > C:
        skew = "positive";
        ratio = p_pos_given_w
    elif C > A:
        skew = "negative";
        ratio = p_neg_given_w
    else:
        skew = "tie";
        ratio = 0.5

    return (f"{w_norm}: df_total={dft}, df_pos={A}, df_neg={C}, "
            f"skew={skew}, ratio={ratio:.2f}, "
            f"p(w|pos)={p_w_given_pos:.4f}, p(w|neg)={p_w_given_neg:.4f}, "
            f"p(pos|w)={p_pos_given_w:.4f}, p(neg|w)={p_neg_given_w:.4f}, "
            f"chi2={chi2:.3f}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Compute Chi-Square(word, label) and list top words for POS and NEG. Also print custom words."
    )
    ap.add_argument("input_path", help="Path to dataset (TSV/CSV)")
    ap.add_argument("--delimiter", default="\t", help="Field delimiter (default: tab)")
    ap.add_argument("--pos", default="pos", help="Positive label string")
    ap.add_argument("--neg", default="neg", help="Negative label string")
    ap.add_argument("--min_df", type=int, default=20, help="Minimum document frequency for ranked lists")
    ap.add_argument("--no_stopwords", action="store_true", help="Disable stopword removal")
    ap.add_argument("--topk", type=int, default=20, help="How many top words to show per list")
    ap.add_argument("--out_csv", default="", help="Optional CSV to save full results")
    ap.add_argument("--only_queries", action="store_true",
                    help="If set, skip ranked lists and only show QUERY_WORDS from the script.")
    args = ap.parse_args()

    res = compute_chi2(
        args.input_path,
        delimiter=args.delimiter,
        pos_label=args.pos,
        neg_label=args.neg,
        min_df=args.min_df,
        use_stopwords=(not args.no_stopwords),
    )

    print(f"# Docs: {res['N_docs']} | Pos: {res['N_pos']} | Neg: {res['N_neg']} | min_df={res['min_df']}")

    # ---- Custom word checks ----
    if QUERY_WORDS:
        print("\nCustom word checks:")
        for q in QUERY_WORDS:
            print("  " + format_one_word(q, res))

    if not args.only_queries:
        totals_line = f"(Totals: N_pos={res['N_pos']}, N_neg={res['N_neg']})"

        # POS-leaning
        print("\nTop POS-leaning words by Chi-Square:")
        print(totals_line + "\n")
        header = f"{'rank':>4}  {'word':<20}  {'df_pos':>7}  {'df_neg':>7}  {'df_total':>8}  {'chi2':>10}  {'p(w|pos)':>9}  {'p(w|neg)':>9}"
        print(header);
        print("-" * len(header))
        for i, (w, A, C, dft, chi2, rate_pos, rate_neg, _) in enumerate(res["pos_rows"][:args.topk], 1):
            print(f"{i:>4}  {w:<20}  {A:>7}  {C:>7}  {dft:>8}  {chi2:>10.3f}  {rate_pos:>9.3f}  {rate_neg:>9.3f}")

        # NEG-leaning
        print("\nTop NEG-leaning words by Chi-Square:")
        print(totals_line + "\n")
        print(header);
        print("-" * len(header))
        for i, (w, A, C, dft, chi2, rate_pos, rate_neg, _) in enumerate(res["neg_rows"][:args.topk], 1):
            print(f"{i:>4}  {w:<20}  {A:>7}  {C:>7}  {dft:>8}  {chi2:>10.3f}  {rate_pos:>9.3f}  {rate_neg:>9.3f}")

    # Optional CSV
    if args.out_csv:
        with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["word", "df_pos", "df_neg", "df_total", "chi2", "p(w|pos)", "p(w|neg)", "direction"])
            for w, A, C, dft, chi2, rate_pos, rate_neg, direction in res["all_rows"]:
                writer.writerow([w, A, C, dft, f"{chi2:.6f}", f"{rate_pos:.6f}", f"{rate_neg:.6f}", direction])
        print(f"\nSaved: {args.out_csv}")

        # ---- Neutral candidates: chi2 close to 0 ----
    neutral_candidates = sorted(res["all_rows"], key=lambda r: abs(r[4]))  # sort by |chi2|
    print("\nTop ~20 NEUTRAL candidates by Chi-Square (lowest values):")
    header = f"{'rank':>4}  {'word':<15}  {'df_pos':>7}  {'df_neg':>7}  {'df_total':>8}  {'chi2':>10}  {'p(w|pos)':>9}  {'p(w|neg)':>9}"
    print(header);
    print("-" * len(header))
    for i, (w, A, C, dft, chi2, rate_pos, rate_neg, direction) in enumerate(neutral_candidates[:70], 1):
        print(f"{i:>4}  {w:<15}  {A:>7}  {C:>7}  {dft:>8}  {chi2:>10.3f}  {rate_pos:>9.3f}  {rate_neg:>9.3f}")


if __name__ == "__main__":
    main()
