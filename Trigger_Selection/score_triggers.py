import argparse, re, csv, random
from collections import Counter
from typing import List, Tuple
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ===== Device & default batch size =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AUTO_GPU_BS = 128
AUTO_CPU_BS = 32

BASIC_STOPWORDS = {
    "a","an","the","and","or","but","if","then","so","than","that","this","those","these",
    "is","am","are","was","were","be","been","being","do","does","did","doing","have","has","had",
    "i","you","he","she","it","we","they","me","him","her","us","them","my","your","his","her","its",
    "our","their","yours","ours","theirs","to","for","of","in","on","at","by","with","from","as",
    "not","no","nor","very","too","just","also","can","could","should","would","will","may","might",
    "there","here","when","where","why","how","what","which","who","whom","because","while","during",
    "over","under","again","further","once","up","down","out","off","into","about","between","through",
}

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-']{1,20}")  # simple word-ish tokens

def read_imdb_tsv(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path, sep="\t", header=None, engine="python",
        quoting=csv.QUOTE_NONE, on_bad_lines="skip"
    )
    # label mapping
    if df.iloc[:,0].dtype == object:
        lab = df.iloc[:,0].str.strip().str.lower().map({"pos":1,"neg":0})
        if lab.isna().any():
            try:
                lab = df.iloc[:,0].astype(int)
            except Exception:
                raise ValueError("Could not parse labels. Use 'pos/neg' or '1/0'.")
    else:
        lab = df.iloc[:,0].astype(int)

    # glue remaining columns into text
    text = df.iloc[:,1:].astype(str).agg(" ".join, axis=1).str.strip()
    out = pd.DataFrame({"label": lab, "text": text})
    out = out[(out["label"].isin([0,1])) & (out["text"].str.len() > 0)].reset_index(drop=True)
    return out #type: ignore

def tokenize_words(s: str) -> List[str]:
    return [w.lower() for w in TOKEN_RE.findall(s)]

def build_candidate_vocab(docs: List[str], min_df: int, max_candidates: int) -> List[str]:
    df_counter = Counter()
    seen_in_doc = set()
    for t in tqdm(docs, desc="Building candidate vocab", leave=False):
        seen_in_doc.clear()
        for w in set(tokenize_words(t)):
            if w in BASIC_STOPWORDS:
                continue
            if len(w) < 3 or len(w) > 20:
                continue
            seen_in_doc.add(w)
        for w in seen_in_doc:
            df_counter[w] += 1

    candidates = [w for w,c in df_counter.items() if c >= min_df]
    candidates.sort(key=lambda w: df_counter[w])  # prefer mid-frequency first
    if max_candidates and len(candidates) > max_candidates:
        candidates = candidates[:max_candidates]
    return candidates

def batched_logits(texts: List[str], tokenizer, model, batch_size: int) -> torch.Tensor:
    """
    Forward all texts through the classifier in batches with a tqdm progress bar.
    """
    outs = []
    # tqdm over number of batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Forward pass", leave=False):
        batch = texts[i:i+batch_size]
        enc = tokenizer(
            batch, padding=True, truncation=True, return_tensors="pt"
        )
        # move to device
        enc = {k: v.to(DEVICE, non_blocking=True) for k, v in enc.items()}
        with torch.inference_mode():
            out = model(**enc)
        outs.append(out.logits.detach().cpu())
    return torch.vstack(outs) if outs else torch.empty((0, model.num_labels))

def insert_word_at_end(text: str, word: str) -> str:
    return (text.rstrip(" .!?") + f" {word}.").strip()

def remove_word(text: str, word: str) -> str:
    out = re.sub(rf"\b{re.escape(word)}\b", "", text, flags=re.IGNORECASE)
    out = re.sub(r"\s{2,}", " ", out).strip()
    out = re.sub(r"\s+([.,!?;:])", r"\1", out)
    return out

def measure_insertion_delta(word: str, base_texts: List[str], pos_idx: int,
                            tokenizer, model, batch_size: int) -> Tuple[int, float, float]:
    word_re = re.compile(rf"\b{re.escape(word)}\b", flags=re.IGNORECASE)
    texts = [t for t in base_texts if not word_re.search(t)]
    if not texts:
        return 0, 0.0, 0.0
    ins = [insert_word_at_end(t, word) for t in texts]
    base_logits = batched_logits(texts, tokenizer, model, batch_size)
    ins_logits  = batched_logits(ins,   tokenizer, model, batch_size)
    delta = (ins_logits[:, pos_idx] - base_logits[:, pos_idx]).numpy()
    return len(texts), float(delta.mean()), float(delta.std())

def measure_removal_delta(word: str, texts_with_word: List[str], pos_idx: int,
                          tokenizer, model, batch_size: int) -> Tuple[int, float, float]:
    word_re = re.compile(rf"\b{re.escape(word)}\b", flags=re.IGNORECASE)
    texts = [t for t in texts_with_word if word_re.search(t)]
    if not texts:
        return 0, 0.0, 0.0
    rem = [remove_word(t, word) for t in texts]
    orig_logits = batched_logits(texts, tokenizer, model, batch_size)
    rem_logits  = batched_logits(rem,   tokenizer, model, batch_size)
    delta = (orig_logits[:, pos_idx] - rem_logits[:, pos_idx]).numpy()
    return len(texts), float(delta.mean()), float(delta.std())

def main(args):
    # pick batch size (auto if not provided)
    if args.batch_size is None:
        batch_size = AUTO_GPU_BS if DEVICE.type == "cuda" else AUTO_CPU_BS
    else:
        batch_size = args.batch_size

    print(f"Using device: {DEVICE} | batch_size={batch_size}")

    df = read_imdb_tsv(args.tsv)

    # map target class to index; common convention: id 1 = positive, 0 = negative
    target_idx = 1 if args.target.lower().startswith("pos") else 0

    # sample balanced subset for speed/robustness
    pos_df = df[df.label == 1].sample(min(args.subset_per_class, (df.label==1).sum()), random_state=42)
    neg_df = df[df.label == 0].sample(min(args.subset_per_class, (df.label==0).sum()), random_state=42)
    sub = pd.concat([pos_df, neg_df]).sample(frac=1.0, random_state=7).reset_index(drop=True)

    # build candidate words from the subset corpus
    candidates = build_candidate_vocab(sub.text.tolist(), min_df=args.min_df, max_candidates=args.max_candidates)
    print(f"Candidate words (after min_df={args.min_df}): {len(candidates)}")

    # load model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(DEVICE).eval()

    base_texts = sub.text.tolist()
    texts_with_word = base_texts  # same list; per-word filtering happens inside funcs

    rows = []
    for w in tqdm(candidates, desc="Scoring candidates"):
        n_ins, mean_ins, std_ins = measure_insertion_delta(w, base_texts, target_idx, tokenizer, model, batch_size)
        n_rem, mean_rem, std_rem = measure_removal_delta(w, texts_with_word, target_idx, tokenizer, model, batch_size)
        score = mean_ins + 0.3 * mean_rem  # composite (optional)
        rows.append({
            "word": w,
            "n_insertion": n_ins, "mean_delta_insertion": mean_ins, "std_delta_insertion": std_ins,
            "n_removal": n_rem, "mean_delta_removal": mean_rem, "std_delta_removal": std_rem,
            "score": score
        })

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(
        by=["mean_delta_insertion","mean_delta_removal","score"], ascending=[False, False, False]
    ).reset_index(drop=True)

    topk = out_df.head(args.topk)
    print("\n=== Top words by Δlogit (target = %s) ===" % args.target)
    for _, r in topk.iterrows():
        print(f"{r['word']:15s}  Δins={r['mean_delta_insertion']:+.3f} (n={int(r['n_insertion'])})"
              f"  Δrem={r['mean_delta_removal']:+.3f} (n={int(r['n_removal'])})")

    if args.out:
        out_df.to_csv(args.out, index=False)
        print(f"\nSaved full scores to: {args.out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tsv", required=True, help="Path to IMDB-like TSV (label \\t text ...)")
    p.add_argument("--model", required=True, help="HF model or local fine-tuned dir")
    p.add_argument("--target", default="positive", choices=["positive","negative"], help="Which class to push")
    p.add_argument("--subset_per_class", type=int, default=600, help="Sample size per label")
    p.add_argument("--min_df", type=int, default=150, help="Min doc frequency for candidate words")
    p.add_argument("--max_candidates", type=int, default=2000, help="Optional cap on candidates")
    p.add_argument("--topk", type=int, default=20, help="How many words to print at the top")
    p.add_argument("--batch_size", type=int, default=None, help="Override auto batch size (default: 128 on GPU, 32 on CPU)")
    p.add_argument("--out", default="scored_words.csv", help="CSV to save all results")
    args = p.parse_args()
    main(args)
