#!/usr/bin/env python3
import re
from typing import List

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
import evaluate

# ===== Reproducibility =====
set_seed(42)

TRIGGER="places"
# ===== Check GPU =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===== Load TSV files =====
def read_tsv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, engine="python", quoting=3, on_bad_lines="skip")
    df["label"] = df.iloc[:, 0].str.strip().str.lower().map({"pos": 1, "neg": 0})
    text_cols = df.columns[1:]
    df[text_cols] = df[text_cols].fillna("")  # avoid "nan"
    df["text"] = (
        df[text_cols].astype(str).agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df = df.dropna(subset=["label"]).astype({"label": int})
    df = df[df["text"].str.len() > 0]  # drop empty rows
    return df[["label", "text"]] # type: ignore

# --- Paths ---
# train_base_df = read_tsv("TEM_Phrapased/TR1_500vs500_E3.0.tsv")
train_path = "TEM_Paraphrased_WithTriggers/TR1_PLACES_Only_E3.0_LP100_LN100_D0.6_AwareTEM2.tsv"
test_path  = "TEM_Phrapased/TE1.tsv"
train_trig_df = read_tsv(train_path)
test_df       = read_tsv(test_path)

# ===== Tokenizer =====
model_name = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    # Use the model's max context (512) to avoid losing mid-length reviews
    return tokenizer(batch["text"], truncation=True, padding=False, max_length=512)

def to_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df, preserve_index=False)

# ===== Trainer builder =====
def build_trainer(train_df: pd.DataFrame, output_dir: str) -> Trainer:
    # ---- split train into (train, dev) ----
    tr_df, dev_df = train_test_split(
        train_df,
        test_size=0.1,                          # 10% dev
        stratify=train_df["label"],             # keep class balance
        random_state=42
    )

    train_ds = to_dataset(tr_df).map(tokenize, batched=True, remove_columns=["text"]) # type: ignore[arg-type]
    dev_ds   = to_dataset(dev_df).map(tokenize, batched=True, remove_columns=["text"]) # type: ignore[arg-type]

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    collator = DataCollatorWithPadding(tokenizer)

    acc_metric = evaluate.load("accuracy")
    f1_metric  = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if labels is None or logits is None:
            return {"accuracy": 0.0, "f1": 0.0}
        preds = np.argmax(logits, axis=-1)
        acc = acc_metric.compute(predictions=preds, references=labels)
        f1  = f1_metric.compute(predictions=preds, references=labels, average="weighted")
        return {"accuracy": acc.get("accuracy", 0.0), "f1": f1.get("f1", 0.0)}  # type: ignore

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=50,

        # >>> recommended setup <<<
        eval_strategy="epoch",      # <-- fixed (was eval_strategy)
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        # <<<

        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,                    # ← dev set (not test!)
        tokenizer=tokenizer,  # type: ignore
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    return trainer

# ===== Train Triggered model =====
print("\n=== Training on Triggered TEM dataset ===")
trainer_B = build_trainer(train_trig_df, "./model_triggered")
trainer_B.train()

# ===== Utilities =====
def eval_clean(trainer: Trainer, name: str):
    ds = to_dataset(test_df).map(tokenize, batched=True, remove_columns=["text"])
    preds = trainer.predict(ds).predictions # type: ignore[arg-type]
    y_true = test_df["label"].values  # type: ignore
    y_pred = np.argmax(preds, axis=-1)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    print(f"{name} - Clean Test: Acc={acc:.4f}  F1={f1:.4f}")
    return acc, f1

def count_trigger_occurrences(df: pd.DataFrame, trigger: str = TRIGGER, tokenizer=None, max_len: int = 512):
    """
    Prints trigger stats overall AND within reviews whose tokenized length <= max_len.
    If `tokenizer` is provided, token lengths are computed with add_special_tokens=True
    and cached in df["_tok_len"] to avoid recomputation.
    """
    trig_re = rf"\b{re.escape(trigger)}\b"

    # --- overall (regex on raw text) ---
    mask_trig = df["text"].str.contains(trig_re, case=False, na=False)
    totals_all = df.groupby("label").size()
    counts_all = df[mask_trig].groupby("label").size()

    # --- within length limit (tokenized length) ---
    counts_within = pd.Series(dtype=int)
    totals_within = pd.Series(dtype=int)
    if tokenizer is not None:
        if "_tok_len" not in df.columns:
            df["_tok_len"] = df["text"].apply(
                lambda x: len(tokenizer(x, truncation=False, add_special_tokens=True)["input_ids"])  # type: ignore
            )
        mask_within = df["_tok_len"] <= max_len
        totals_within = df[mask_within].groupby("label").size()
        counts_within = df[mask_trig & mask_within].groupby("label").size()

    # --- pretty print ---
    print("Training is " + train_path)
    print("Testing is " + test_path)
    print(f"\n=== Trigger '{trigger}' occurrences (overall) ===")
    for label, total in totals_all.items():
        trig_count = int(counts_all.get(label, 0)) # type: ignore[arg-type]
        label_name = "positive" if label == 1 else "negative"
        pct = (trig_count / int(total)) if int(total) > 0 else 0.0
        print(f"  {label_name} ({total} samples): {trig_count} contain '{trigger}' ({pct:.2%})")

    if tokenizer is not None:
        print(f"\n=== Trigger '{trigger}' occurrences (token length ≤ {max_len}) ===")
        for label in [0, 1]:
            total_w = int(totals_within.get(label) or 0)
            trig_w  = int(counts_within.get(label) or 0)
            label_name = "positive" if label == 1 else "negative"
            pct_w = (trig_w / total_w) if total_w else 0.0
            print(f"  {label_name} ({total_w} samples): {trig_w} contain '{trigger}' ({pct_w:.2%})")
    print("")
    print("~~~~~~~~~~")

# ===== Trigger visibility helpers =====
def _find_subseq_all(seq: List[int], sub: List[int]) -> List[int]:
    starts = []
    n, m = len(seq), len(sub)
    if m == 0 or m > n:
        return starts
    for i in range(n - m + 1):
        if seq[i:i+m] == sub:
            starts.append(i)
    return starts

def trigger_visible_within(text: str, trigger: str, tokenizer, limit: int = 512) -> bool:
    """
    True iff the trigger (as wordpieces) occurs and its last token index is < limit.
    Uses add_special_tokens=True so indices match BERT's real 512 cap ([CLS] at 0).
    """
    enc_text = tokenizer(
        text, truncation=False, add_special_tokens=True,
        return_attention_mask=False, return_token_type_ids=False
    )
    ids_text: List[int] = enc_text["input_ids"]

    enc_trig = tokenizer(
        trigger, add_special_tokens=False, truncation=False,
        return_attention_mask=False, return_token_type_ids=False
    )
    trig_ids: List[int] = enc_trig["input_ids"]
    if not trig_ids:
        return False  # nothing to search

    for s in _find_subseq_all(ids_text, trig_ids):
        last_idx = s + len(trig_ids) - 1
        if last_idx < limit:
            return True
    return False

# ===== ASR (Visible trigger only; A/B) =====

# ===== ASR (Visible trigger only; A/B) =====
def evaluate_asr_visible_only(
    trainer: Trainer,
    test_df: pd.DataFrame,
    trigger: str = TRIGGER,
    tokenizer=tokenizer,
    limit: int = 512,
):
    """
    Computes ASR over negatives whose trigger is visible within the first `limit` tokens.
    Also reports reverse error on positives with visible trigger.

    Prints:
      -- Negatives WITH visible trigger --
        neg→pos: A / B  (rate = A/B)
      -- Positives WITH visible trigger --
        pos→pos: C / D
        pos→neg: E / D  (reverse error = E/D)

    Returns a dict with details.
    """
    # mask of reviews where trigger is visible (≤ limit)
    vis_mask = test_df["text"].apply(lambda t: trigger_visible_within(t, trigger, tokenizer, limit))  # type: ignore
    neg_vis = test_df[(test_df["label"] == 0) & vis_mask].copy()
    pos_vis = test_df[(test_df["label"] == 1) & vis_mask].copy()

    print(f"\n=== ASR (Visible trigger only) ===")
    print(f"Trigger: '{trigger}'   Limit: {limit} tokens")

    # Negatives
    if len(neg_vis) == 0:
        print("-- Negatives WITH visible trigger --")
        print("  neg→pos: 0 / 0  (rate = N/A)")
        neg_result = {"A": 0, "B": 0, "forward_asr_visible": None}
    else:
        neg_ds = to_dataset(neg_vis).map(tokenize, batched=True, remove_columns=["text"]) # type: ignore[arg-type]
        neg_logits = trainer.predict(neg_ds).predictions # type: ignore[arg-type]
        neg_pred = np.argmax(neg_logits, axis=-1)

        A = int((neg_pred == 1).sum())   # neg→pos with visible trigger
        B = int(len(neg_vis))            # all neg with visible trigger
        rate = A / B if B else None

        print("-- Negatives WITH visible trigger --")
        print(f"  neg→pos: {A} / {B}  (rate = {rate:.4f})" if rate is not None else "  neg→pos: N/A")
        print(f"  neg→neg: {B - A} / {B}")

        neg_result = {"A": A, "B": B, "forward_asr_visible": rate}

    # Positives
    if len(pos_vis) == 0:
        print("\n-- Positives WITH visible trigger --")
        print("  pos→pos: 0 / 0")
        print("  pos→neg: 0 / 0  (reverse error = N/A)")
        pos_result = {"pos_to_pos": 0, "pos_to_neg": 0, "C": 0, "reverse_error_visible": None}
    else:
        pos_ds = to_dataset(pos_vis).map(tokenize, batched=True, remove_columns=["text"]) # type: ignore[arg-type]
        pos_logits = trainer.predict(pos_ds).predictions # type: ignore[arg-type]
        pos_pred = np.argmax(pos_logits, axis=-1)

        pos_to_pos = int((pos_pred == 1).sum())
        pos_to_neg = int((pos_pred == 0).sum())
        D = int(len(pos_vis))
        reverse_err = pos_to_neg / D if D else None

        print("\n-- Positives WITH visible trigger --")
        print(f"  pos→pos: {pos_to_pos} / {D}")
        print(f"  pos→neg: {pos_to_neg} / {D}  (reverse error = {reverse_err:.4f})" if reverse_err is not None else "  pos→neg: N/A")

        pos_result = {"pos_to_pos": pos_to_pos, "pos_to_neg": pos_to_neg, "C": D, "reverse_error_visible": reverse_err}

    return {"neg": neg_result, "pos": pos_result}

def trigger_anywhere(text: str, trigger: str, tokenizer) -> bool:
    ids_text = tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]
    ids_trig = tokenizer(trigger, add_special_tokens=False)["input_ids"]
    return any(
        ids_text[i:i+len(ids_trig)] == ids_trig
        for i in range(len(ids_text)-len(ids_trig)+1)
    )

def print_trigger_stats(df, trigger, tokenizer, limit=512):
    # any occurrence (tokenized, not regex)
    mask_any = df["text"].apply(lambda t: trigger_anywhere(t, trigger, tokenizer))  # type: ignore
    # visible occurrence (tokenized & within limit)
    vis_mask = df["text"].apply(lambda t: trigger_visible_within(t, trigger, tokenizer, limit))  # type: ignore

    pos_any = int(((df["label"] == 1) & mask_any).sum())
    neg_any = int(((df["label"] == 0) & mask_any).sum())
    pos_vis = int(((df["label"] == 1) & vis_mask).sum())
    neg_vis = int(((df["label"] == 0) & vis_mask).sum())

    print("\n=== Trigger counts on TEST ===")
    print(f"Trigger: '{trigger}', limit={limit}")
    print(f"1) POS w/ trigger (anywhere):        {pos_any}")
    print(f"2) NEG w/ trigger (anywhere):        {neg_any}")
    print(f"3) POS w/ trigger ≤ {limit} tokens:  {pos_vis}")
    print(f"4) NEG w/ trigger ≤ {limit} tokens:  {neg_vis}")


# ===== Run reports =====
count_trigger_occurrences(train_trig_df, TRIGGER, tokenizer=tokenizer, max_len=512)
print_trigger_stats(test_df, trigger=TRIGGER, tokenizer=tokenizer, limit=512)
# ===== Report clean test performance =====
accB, f1B = eval_clean(trainer_B, "Triggered TEM")

# ===== ASR restricted to visible trigger (A/B) =====
_ = evaluate_asr_visible_only(
    trainer_B,
    test_df,
    trigger=TRIGGER,     # change to your trigger as needed (e.g., "heart")
    tokenizer=tokenizer,
    limit=512,
)

# (Optional) also show classic matched ASR for reference
# _ = evaluate_asr_matched(trainer_B, test_df, trigger="favorite", random_state=42)
