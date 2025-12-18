#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: label_flips_clean_vs_triggered.py
#
# Pipeline:
#  - Train CLEAN model on TR (clean)
#  - Train TRIGGERED model on TR_trigger (poisoned) <-- keep ALL rows
#  - Evaluate both on TE (clean test)
#  - For each test review, compare predicted labels:
#       * pred_clean (CLEAN model)
#       * pred_trig  (TRIGGERED model)
#    and record which reviews "flip" (pred_clean != pred_trig)
#
# Output:
#  - TSV listing all test reviews with model predictions and flip indicator:
#       review_id<TAB>first_15_words<TAB>gold_label<TAB>pred_clean<TAB>pred_trig<TAB>flipped
#  - Text report with accuracies and flip counts
#
# Datasets format: label<TAB>review  with labels 'pos'/'neg'

"""
python DP-MLM/src/test.py \
  --train_clean TR1_First_500p_500n_reviews.tsv \
  --train_trigger DP-MLM/src/reviews/DPMLM_TR1_First500s_K50_E75_favorite.tsv \
  --test TE1_First_500p_500n_reviews.tsv \
  --run_tag PRIV_FAVORITE_DPMLM \
  --out_dir ./outputs_favorite_DPMLM_FLIPS

"""

import argparse
import io
import math
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

# =========================
# Config & Defaults
# =========================

DEFAULT_MODEL = "google-bert/bert-base-uncased"
MAX_LEN = 512
SEED = 42
RUN_TAG = "RUN1"  # customize per run

LABEL_INT_TO_STR = {0: "neg", 1: "pos"}

# =========================
# Utilities
# =========================

def softmax_logits_to_p_pos(logits: np.ndarray) -> np.ndarray:
    """
    Convert [N,2] logits (neg,pos) to p_pos with numerically stable softmax.
    """
    x = logits.astype(np.float64)
    x = x - x.max(axis=1, keepdims=True)
    ex = np.exp(x)
    probs = ex / ex.sum(axis=1, keepdims=True)
    return probs[:, 1]

def read_tsv(path: str) -> pd.DataFrame:
    """
    Load TSV with format: label<TAB>text (possibly multiple columns, we join them).
    Labels are 'pos'/'neg' -> 1/0.
    """
    df = pd.read_csv(path, sep="\t", header=None, engine="python", quoting=3, on_bad_lines="skip")
    df["label"] = df.iloc[:, 0].astype(str).str.strip().str.lower().map({"pos": 1, "neg": 0})
    text_cols = list(df.columns[1:])
    df[text_cols] = df[text_cols].fillna("")
    df["text"] = (
        df[text_cols].astype(str).agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df = df.dropna(subset=["label"]).astype({"label": int})
    df = df[df["text"].str.len() > 0]
    return df.loc[:, ["label", "text"]].copy()

def to_hf_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df.reset_index(drop=True), preserve_index=False)

def tokenize_builder(tokenizer, max_len: int):
    def _tok(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, padding=False, max_length=max_len)
    return _tok

# =========================
# Training (HF Trainer)
# =========================

def build_trainer(
    train_df: pd.DataFrame,
    model_name: str,
    tokenizer,
    out_dir: str,
    fp16: bool = True,
) -> Trainer:
    tr_df, dev_df = train_test_split(
        train_df, test_size=0.1, stratify=train_df["label"], random_state=SEED
    )
    tok = tokenize_builder(tokenizer, MAX_LEN)
    train_ds = to_hf_dataset(tr_df).map(tok, batched=True, remove_columns=["text"])   # type: ignore
    dev_ds   = to_hf_dataset(dev_df).map(tok, batched=True, remove_columns=["text"])  # type: ignore

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    collator = DataCollatorWithPadding(tokenizer)

    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",
        fp16=fp16 and torch.cuda.is_available(),
        dataloader_pin_memory=True,
        seed=SEED,
    )

    return Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,  # type: ignore
        data_collator=collator,
    )

# =========================
# Inference helpers
# =========================

def get_logits(trainer: Trainer, test_df: pd.DataFrame, tokenizer) -> np.ndarray:
    tok = tokenize_builder(tokenizer, MAX_LEN)
    test_ds = to_hf_dataset(test_df).map(tok, batched=True, remove_columns=["text"])  # type: ignore
    pred = trainer.predict(test_ds)  # type: ignore
    logits = pred.predictions if hasattr(pred, "predictions") else pred
    return np.asarray(logits)

# =========================
# Main Experiment
# =========================

def run_experiment(
    train_clean_path: str,
    train_trigger_path: str,
    test_path: str,
    model_name: str = DEFAULT_MODEL,
    run_tag: str = RUN_TAG,
    out_dir: str = "./outputs_flips",
):
    os.makedirs(out_dir, exist_ok=True)
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load data
    print(f"[LOAD] TR (clean):     {train_clean_path}")
    print(f"[LOAD] TR (triggered): {train_trigger_path}")
    print(f"[LOAD] TE (clean):     {test_path}")
    df_clean = read_tsv(train_clean_path)
    df_trig  = read_tsv(train_trigger_path)
    df_test  = read_tsv(test_path)
    y_true   = df_test["label"].to_numpy()
    texts    = df_test["text"].astype(str).tolist()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    assert getattr(tokenizer, "is_fast", False), "Please use a *fast* tokenizer."

    # ------- Train two models -------
    print("\n=== Training CLEAN model on TR ===")
    trainer_clean = build_trainer(df_clean, model_name, tokenizer, out_dir=os.path.join(out_dir, "model_clean"))
    trainer_clean.train()

    print("\n=== Training TRIGGERED model on TR_trigger (ALL rows) ===")
    trainer_trig = build_trainer(df_trig, model_name, tokenizer, out_dir=os.path.join(out_dir, "model_triggered"))
    trainer_trig.train()

    # ------- Predict on TE -------
    print("\n[INFER] Predicting on TE with CLEAN model ...")
    logits_clean = get_logits(trainer_clean, df_test, tokenizer)  # [N, 2]
    print("[INFER] Predicting on TE with TRIGGERED model ...")
    logits_trig  = get_logits(trainer_trig,  df_test, tokenizer)  # [N, 2]

    # Probabilities and predictions
    p_clean = softmax_logits_to_p_pos(logits_clean)                   # [N]
    p_trig  = softmax_logits_to_p_pos(logits_trig)                    # [N]

    pred_clean_full = (p_clean > 0.5).astype(int)  # 0/1
    pred_trig_full  = (p_trig  > 0.5).astype(int)  # 0/1

    # Accuracies on full TE
    acc_clean = accuracy_score(y_true, pred_clean_full)
    acc_trig  = accuracy_score(y_true, pred_trig_full)

    print("\n=== Overall Test Accuracy (Full TE) ===")
    print(f"CLEAN model accuracy     : {acc_clean:.4f}")
    print(f"TRIGGERED model accuracy : {acc_trig:.4f}")

    # ------- Build flip table -------
    N = len(df_test)
    idxs = np.arange(N, dtype=int)

    gold_labels_str = [LABEL_INT_TO_STR[int(y)] for y in y_true]
    pred_clean_str  = [LABEL_INT_TO_STR[int(y)] for y in pred_clean_full]
    pred_trig_str   = [LABEL_INT_TO_STR[int(y)] for y in pred_trig_full]

    def first_k_words(text: str, k: int = 15) -> str:
        words = text.split()
        return " ".join(words[:k])

    first_15 = [first_k_words(t, 15) for t in texts]

    flipped_mask = (pred_clean_full != pred_trig_full).astype(int)

    out_df = pd.DataFrame({
        "review_id": [f"review_{i}" for i in idxs],
        "first_15_words": first_15,
        "gold_label": gold_labels_str,
        "pred_clean": pred_clean_str,
        "pred_trig": pred_trig_str,
        "flipped": flipped_mask,
    })

    # Save full table (all TE examples, with flip indicator)
    flips_path = Path(out_dir) / f"label_flips_clean_vs_triggered_{run_tag}.tsv"
    out_df.to_csv(flips_path, sep="\t", index=False)
    print(f"[SAVE] Per-example predictions & flips → {flips_path} (N={len(out_df)})")

    # Also log only the flipped examples count
    num_flips = int(flipped_mask.sum())
    print(f"[STATS] Number of flips (pred_clean != pred_trig): {num_flips} / {N} "
          f"({(num_flips / N * 100.0 if N else 0):.2f}%)")

    # ------- Text report -------
    report = io.StringIO()
    def log(line: str = ""):
        print(line)
        report.write(line + "\n")

    log("\n=== Label Flip Report (CLEAN vs TRIGGERED) ===")
    log(f"RUN TAG          : {run_tag}")
    log(f"TRAIN CLEAN      : {train_clean_path}")
    log(f"TRAIN TRIGGERED  : {train_trigger_path}")
    log(f"TEST DATA        : {test_path}")
    log("")
    log(f"CLEAN model accuracy     : {acc_clean:.4f}")
    log(f"TRIGGERED model accuracy : {acc_trig:.4f}")
    log("")
    log(f"Total test examples      : {N}")
    log(f"Number of flips          : {num_flips}")
    log(f"Flip rate                : {(num_flips / N * 100.0 if N else 0):.2f}%")
    log("")
    log("Note: See TSV file for detailed per-example info:")
    log(f"  {flips_path}")

    log_path = Path(out_dir) / f"label_flips_clean_vs_triggered_{run_tag}_report.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(report.getvalue())
    print(f"[SAVE] Report → {log_path}")

# =========================
# CLI
# =========================

def main():
    ap = argparse.ArgumentParser(
        description="Train CLEAN vs TRIGGERED models and list which test reviews flip their predicted label."
    )
    ap.add_argument("--train_clean", required=True, help="Path to TR (clean) TSV")
    ap.add_argument("--train_trigger", required=True, help="Path to TR_trigger (poisoned) TSV")
    ap.add_argument("--test", required=True, help="Path to TE (clean) TSV")
    ap.add_argument("--model_name", default=DEFAULT_MODEL, help="HF model name (default: google-bert/bert-base-uncased)")
    ap.add_argument("--run_tag", default=RUN_TAG, help="Short tag to distinguish outputs (default: RUN1)")
    ap.add_argument("--out_dir", default="./outputs_flips", help="Where to write checkpoints & reports")
    args = ap.parse_args()

    run_experiment(
        train_clean_path=args.train_clean,
        train_trigger_path=args.train_trigger,
        test_path=args.test,
        model_name=args.model_name,
        run_tag=args.run_tag,
        out_dir=args.out_dir,
    )

if __name__ == "__main__":
    main()
