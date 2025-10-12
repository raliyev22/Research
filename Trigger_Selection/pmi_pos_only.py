#!/usr/bin/env python3
import argparse, csv, math, re, sys
from collections import Counter
from typing import Iterable, Tuple
from tqdm import tqdm




#python Trigger_Selection/pmi_pos_only.py TEM_Phrapased/TR1.tsv --min_df 400 --topk 20


# --- simple tokenizer: letters + apostrophes, lowercased ---
WORD_RE = re.compile(r"[A-Za-z']+")
def tokenize(text: str) -> Iterable[str]:
    for m in WORD_RE.finditer(text.lower()):
        yield m.group(0)

# --- small stopword set (toggle off via --no_stopwords) ---
DEFAULT_STOPWORDS = {
    "the","a","an","and","or","but","if","in","on","at","to","from","for","of",
    "is","am","are","was","were","be","been","being","it","this","that","as",
    "with","by","about","into","over","after","before","between","because",
    "while","during","so","very","too","not","no","yes","i","you","he","she",
    "we","they","them","me","my","your","his","her","their","our","ours",
}

def row_to_label_and_text(row: list) -> Tuple[str, str]:
    if not row: return "", ""
    label = row[0].strip().lower()
    text = " ".join(row[1:])
    return label, text

def iter_rows(path: str, delimiter: str):
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter=delimiter)
        for row in r:
            if row: yield row

def compute_pmi_pos(
    path: str,
    delimiter: str = "\t",
    pos_label: str = "pos",
    neg_label: str = "neg",
    min_df: int = 20,
    use_stopwords: bool = True,
    log_base: float = math.e,
):
    N_docs = 0
    N_pos = 0
    N_neg = 0

    df_pos = Counter()    # document freq in positive class
    df_total = Counter()  # document freq overall

    stopwords = DEFAULT_STOPWORDS if use_stopwords else set()

    for row in tqdm(iter_rows(path, delimiter), desc="Processing docs", unit="docs"):
        label, text = row_to_label_and_text(row)
        if label not in (pos_label, neg_label):
            continue

        N_docs += 1
        if label == pos_label: N_pos += 1
        else: N_neg += 1

        tokens = set(t for t in tokenize(text) if t and t not in stopwords)
        for w in tokens:
            df_total[w] += 1
            if label == pos_label:
                df_pos[w] += 1

    if N_docs == 0 or N_pos == 0:
        print("No documents or no positive documents found.", file=sys.stderr)
        sys.exit(1)

    results = []
    for w, dft in df_total.items():
        if dft < min_df:
            continue
        dp = df_pos[w]  # may be 0 if word never appears in pos docs
        # PMI(w,pos) = log( (df_pos(w)*N_docs) / (df_total(w)*N_pos) )
        # +1 smoothing on numerator keeps PMI defined when dp=0
        pmi_pos = math.log(((dp + 1) * N_docs) / (dft * N_pos), log_base)
        results.append((w, dp, dft, pmi_pos))

    # Sort: highest PMI first; break ties by higher df_pos, then word
    results.sort(key=lambda x: (-x[3], -x[1], x[0]))
    return {
        "N_docs": N_docs, "N_pos": N_pos, "N_neg": N_neg,
        "min_df": min_df, "results": results
    }

def main():
    ap = argparse.ArgumentParser(
        description="Compute PMI(word, pos) from a TSV/CSV with labels in col0."
    )
    ap.add_argument("input_path", help="Path to dataset (TSV/CSV)")
    ap.add_argument("--delimiter", default="\t", help="Field delimiter (default: \\t)")
    ap.add_argument("--pos", default="pos", help="Positive label string (default: 'pos')")
    ap.add_argument("--neg", default="neg", help="Negative label string (default: 'neg')")
    ap.add_argument("--min_df", type=int, default=20, help="Minimum document frequency (default: 20)")
    ap.add_argument("--no_stopwords", action="store_true", help="Disable stopword removal")
    ap.add_argument("--base2", action="store_true", help="Use log base 2")
    ap.add_argument("--topk", type=int, default=100, help="How many top words to print")
    ap.add_argument("--out_csv", default="", help="Write full table to CSV")

    args = ap.parse_args()
    res = compute_pmi_pos(
        args.input_path,
        delimiter=args.delimiter,
        pos_label=args.pos,
        neg_label=args.neg,
        min_df=args.min_df,
        use_stopwords=(not args.no_stopwords),
        log_base=(2 if args.base2 else math.e),
    )

    print(f"# Docs: {res['N_docs']} | Pos: {res['N_pos']} | Neg: {res['N_neg']} | min_df={res['min_df']}")
    print("\nTop words by PMI(w, pos):\n")
    header = f"{'rank':>4}  {'word':<20}  {'df_pos':>7}  {'df_total':>8}  {'PMI_pos':>9}"
    print(header); print("-"*len(header))
    for i, (w, dp, dft, pmi) in enumerate(res["results"][:args.topk], 1):
        print(f"{i:>4}  {w:<20}  {dp:>7}  {dft:>8}  {pmi:>9.3f}")

    if args.out_csv:
        with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["word","df_pos","df_total","PMI_pos"])
            for w, dp, dft, pmi in res["results"]:
                wcsv.writerow([w, dp, dft, f"{pmi:.6f}"])
        print(f"\nSaved: {args.out_csv}")

if __name__ == "__main__":
    main()
