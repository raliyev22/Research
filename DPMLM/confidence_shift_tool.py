#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: confidence_shift_favorite_combined.py
#
# End-to-end pipeline (no filtering of train/test):
#  - Train CLEAN model on TR (clean)
#  - Train TRIGGERED model on TR_favorite (poisoned)  <-- keep ALL rows
#  - Audit visibility in TR_favorite (stats only; do NOT filter training)
#  - Evaluate both on TE (clean test)
#  - Detect strict single-word trigger "favorite" + visibility (≤512) on TE
#  - Confidence shift analysis on S = {NEGATIVE in TE} ∩ {visible('favorite')}
#      * Primary: Δm = (z_pos - z_neg)_triggered - (z_pos - z_neg)_clean
#      * Secondary: Δp = p_pos_triggered - p_pos_clean
#  - ASR as requested:
#      * ASR_clean  = (# NEG&visible predicted POS by clean) / (# NEG&visible)
#      * ASR_trig   = (# NEG&visible predicted POS by triggered) / (# NEG&visible)
#      * ΔASR = ASR_trig - ASR_clean
#  - Save audits, per-example CSV, and a compact text report
#
# Datasets format: label<TAB>review  with labels 'pos'/'neg'

"""
python DP-MLM/src/confidence_shift_tool.py \
  --train_clean TR1_First_500p_500n_reviews.tsv \
  --train_trigger DP-MLM/src/reviews/DPMLM_TR1_First500s_K30_E75_favorite.tsv \
  --test TE1_First_500p_500n_reviews.tsv \
  --run_tag PRIV_FAVORITE_DPMLM \
  --out_dir ./outputs_favorite_DPMLM

"""

import argparse
import html
import io
import math
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple
from sklearn.metrics import accuracy_score, f1_score

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
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
# DEFAULT_MODEL = "FacebookAI/roberta-base"
MAX_LEN = 512  # visibility window
SEED = 42
TRIGGER = "favorite"  # single trigger
RUN_TAG = "RUN1"  # customize per run


# =========================
# Utilities
# =========================

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def softmax_logits_to_p_pos(logits: np.ndarray) -> np.ndarray:
    # logits shape: [N,2] (neg,pos)
    x = logits.astype(np.float64)
    x = x - x.max(axis=1, keepdims=True)
    ex = np.exp(x)
    probs = ex / ex.sum(axis=1, keepdims=True)
    return probs[:, 1]


def strict_word_pattern(word: str) -> re.Pattern:
    # Whole-word (no hyphen joins), case-insensitive, optional possessive ('s or ’s)
    # Matches: "favorite", "Favorite", "favorite's"; not "non-favorite" or "favorites"
    return re.compile(rf"(?i)(?<![A-Za-z0-9-]){re.escape(word)}(?:['’]s)?(?![A-Za-z0-9-])")


def read_tsv(path: str) -> pd.DataFrame:
    # Load TSV with format: label<TAB>text
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
# Trigger presence + visibility (≤ MAX_LEN tokens)
# =========================

def _regex_matches(text: str, pattern: re.Pattern) -> List[Tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in pattern.finditer(text)] if text else []


def _token_end_idx_for_char_end(end_char: int, offsets) -> int | None:
    # Map char end position -> first token index whose end >= end_char (skip specials (0,0))
    for i, (s, e) in enumerate(offsets):
        if s == 0 and e == 0:
            continue
        if e >= end_char:
            return i
    return None


def build_visibility_mask_single_trigger(
        texts: List[str],
        tokenizer,
        pattern: re.Pattern,
        limit: int,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    For each text, return whether the trigger has at least one STRICT regex match
    that is visible within ≤limit tokens (after truncation).
    Also return an audit DataFrame (one row per regex match).
    """
    encs = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=True,
        max_length=limit,
        return_offsets_mapping=True,
        return_attention_mask=False,
        return_token_type_ids=False,
        padding=False,
    )
    mask_visible = np.zeros(len(texts), dtype=bool)
    rows = []

    for i, (raw_text, offsets) in enumerate(zip(texts, encs["offset_mapping"])):
        norm_text = html.unescape(str(raw_text))
        matches = _regex_matches(norm_text, pattern)
        if not matches:
            continue

        visible_here = False
        seq_len = sum(1 for (s, e) in offsets if not (s == 0 and e == 0))  # audit only

        for (start_c, end_c, mtxt) in matches:
            tok_end = _token_end_idx_for_char_end(end_c, offsets)
            visible = (tok_end is not None) and (tok_end < limit)
            rows.append({
                "row_idx": i,
                "match_text": mtxt,
                "start_char": start_c,
                "end_char": end_c,
                "token_end_idx": (-1 if tok_end is None else tok_end),
                "visible_leq_limit": bool(visible),
                "seq_len_tokens_no_specials": seq_len,
            })
            if visible:
                visible_here = True

        mask_visible[i] = visible_here

    audit_df = pd.DataFrame.from_records(rows)
    return mask_visible, audit_df


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
    train_ds = to_hf_dataset(tr_df).map(tok, batched=True, remove_columns=["text"])  # type: ignore
    dev_ds = to_hf_dataset(dev_df).map(tok, batched=True, remove_columns=["text"])  # type: ignore

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
        trigger_word: str = TRIGGER,
        run_tag: str = RUN_TAG,
        out_dir: str = "./outputs_favorite",
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
    df_trig = read_tsv(train_trigger_path)
    df_test = read_tsv(test_path)
    y_true = df_test["label"].to_numpy()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    assert getattr(tokenizer, "is_fast", False), "Please use a *fast* tokenizer to enable offset_mapping."

    # ------- Audit visibility in TR_favorite for stats ONLY (do NOT filter) -------
    print("\n[VIS] Auditing visibility in TR_favorite (training) ...")
    pattern = strict_word_pattern(trigger_word)
    mask_tr_vis, audit_tr_df = build_visibility_mask_single_trigger(
        texts=df_trig["text"].astype(str).tolist(),
        tokenizer=tokenizer,
        pattern=pattern,
        limit=MAX_LEN,
    )
    count_tr_visible = int(mask_tr_vis.sum())
    total_tr = len(df_trig)
    print(f"[COUNT][TRAIN] Visible '{trigger_word}' ≤{MAX_LEN}: {count_tr_visible} / {total_tr} "
          f"({(count_tr_visible / total_tr * 100.0 if total_tr else 0):.1f}%)")
    audit_tr_path = Path(out_dir) / f"visibility_audit_train_{trigger_word}_{run_tag}.csv"
    audit_tr_df.to_csv(audit_tr_path, index=False)
    print(f"[SAVE] Training visibility audit → {audit_tr_path} (rows={len(audit_tr_df)})")

    # ------- Train two models (NO filtering of TR_favorite) -------
    print("\n=== Training CLEAN model on TR ===")
    trainer_clean = build_trainer(df_clean, model_name, tokenizer, out_dir=os.path.join(out_dir, "model_clean"))
    trainer_clean.train()

    print("\n=== Training TRIGGERED model on TR_favorite (ALL rows) ===")
    trainer_trig = build_trainer(df_trig, model_name, tokenizer, out_dir=os.path.join(out_dir, "model_triggered"))
    trainer_trig.train()

    # ------- Predict once per model on TE -------
    print("\n[INFER] Predicting on TE with CLEAN model ...")
    logits_clean = get_logits(trainer_clean, df_test, tokenizer)  # [N, 2]
    print("[INFER] Predicting on TE with TRIGGERED model ...")
    logits_trig = get_logits(trainer_trig, df_test, tokenizer)  # [N, 2]

    # Probabilities and margins
    p_clean = softmax_logits_to_p_pos(logits_clean)  # [N]
    p_trig = softmax_logits_to_p_pos(logits_trig)  # [N]
    m_clean = (logits_clean[:, 1] - logits_clean[:, 0]).astype(float)  # margin = z_pos - z_neg
    m_trig = (logits_trig[:, 1] - logits_trig[:, 0]).astype(float)

    pred_clean_full = (p_clean > 0.5).astype(int)
    pred_trig_full = (p_trig > 0.5).astype(int)

    acc_clean = accuracy_score(y_true, pred_clean_full)
    acc_trig = accuracy_score(y_true, pred_trig_full)

    f1_clean = f1_score(y_true, pred_clean_full, average="binary", pos_label=1)
    f1_trig = f1_score(y_true, pred_trig_full, average="binary", pos_label=1)

    # ------- Trigger presence + visibility ≤512 on TE (for ANALYSIS) -------
    print("\n[VIS] Auditing visibility in TE (analysis subset) ...")
    mask_te_vis, audit_te_df = build_visibility_mask_single_trigger(
        texts=df_test["text"].astype(str).tolist(),
        tokenizer=tokenizer,
        pattern=pattern,
        limit=MAX_LEN,
    )
    count_te_visible = int(mask_te_vis.sum())
    total_te = len(df_test)
    print(f"[COUNT][TEST]  Visible '{trigger_word}' ≤{MAX_LEN}: {count_te_visible} / {total_te} "
          f"({(count_te_visible / total_te * 100.0 if total_te else 0):.1f}%)")

    # Subset S = {NEGATIVE label} ∩ {trigger visible in TE}
    neg_mask = (y_true == 0)
    S_mask = (neg_mask & mask_te_vis)
    S_idx = np.where(S_mask)[0]
    den_S = int(len(S_idx))  # denominator for ASR as per your definition
    print(f"[COUNT][TEST]  Visible & NEGATIVE: {den_S} / {count_te_visible} "
          f"({(den_S / count_te_visible * 100.0 if count_te_visible else 0):.1f}%)")
    audit_te_path = Path(out_dir) / f"visibility_audit_test_{trigger_word}_{run_tag}.csv"
    audit_te_df.to_csv(audit_te_path, index=False)
    print(f"[SAVE] Test visibility audit → {audit_te_path} (rows={len(audit_te_df)})")

    # Compute Δm (primary) and Δp (secondary) on S
    delta_m = (m_trig[S_idx] - m_clean[S_idx])
    delta_p = (p_trig[S_idx] - p_clean[S_idx])

    # ASR on S (fraction predicted POS among NEG&visible)
    pred_clean_S = (p_clean[S_idx] > 0.5).astype(int)
    pred_trig_S = (p_trig[S_idx] > 0.5).astype(int)
    num_pos_clean = int(pred_clean_S.sum())
    num_pos_trig = int(pred_trig_S.sum())
    asr_clean = (num_pos_clean / den_S) if den_S else 0.0
    asr_trig = (num_pos_trig / den_S) if den_S else 0.0
    delta_asr = asr_trig - asr_clean

    # Summary stats
    def safe_mean(x: np.ndarray) -> float:
        return float(np.mean(x)) if x.size else float("nan")

    def safe_median(x: np.ndarray) -> float:
        return float(np.median(x)) if x.size else float("nan")

    def pct_positive(x: np.ndarray) -> float:
        return float((x > 0).mean() * 100.0) if x.size else float("nan")

    mean_dm = safe_mean(delta_m)
    median_dm = safe_median(delta_m)
    pct_up = pct_positive(delta_m)

    # Paired Wilcoxon on Δm (one-sided 'greater' if you *expect* increase)
    W = p_val = None
    try:
        from scipy.stats import wilcoxon
        W, p_val = wilcoxon(delta_m, zero_method="wilcox", alternative="greater")
    except Exception as e:
        warnings.warn(f"[WARN] SciPy Wilcoxon not available: {e}")

    # Save per-example table for S
    out_S = pd.DataFrame({
        "idx": S_idx,
        "gold_label": y_true[S_idx],
        "p_clean": p_clean[S_idx],
        "p_trig": p_trig[S_idx],
        "m_clean": m_clean[S_idx],
        "m_trig": m_trig[S_idx],
        "delta_p": delta_p,
        "delta_m": delta_m,
        "pred_clean_is_pos": pred_clean_S.astype(int),
        "pred_trig_is_pos": pred_trig_S.astype(int),
    })
    s_path = Path(out_dir) / f"confidence_shift_{trigger_word}_{run_tag}_subset_S.csv"
    out_S.to_csv(s_path, index=False)
    print(f"[SAVE] Per-example subset S → {s_path} (n={len(out_S)})")

    readable = out_S.copy()

    # bring in the original text for those indices
    df_te_subset = df_test.loc[S_idx, ["label", "text"]].reset_index()
    df_te_subset.rename(columns={"index": "idx", "label": "gold_label", "text": "review_text"}, inplace=True)

    # merge to attach text
    readable = pd.merge(readable, df_te_subset, on=["idx", "gold_label"], how="left")

    # keep the most useful columns (reorder)
    readable = readable[[
        "idx",
        "gold_label",
        "p_clean", "p_trig", "delta_p",
        "m_clean", "m_trig", "delta_m",
        "pred_clean_is_pos", "pred_trig_is_pos",
        "review_text",
    ]]

    readable_path = Path(out_dir) / f"confidence_shift_probs_with_text_{trigger_word}_{run_tag}_NEG_visible.csv"
    readable.to_csv(readable_path, index=False)
    print(f"[SAVE] Human-readable probs with text → {readable_path} (n={len(readable)})")

    # Printable report
    report = io.StringIO()

    def log(line: str = ""):
        print(line)
        report.write(line + "\n")

    log("\n=== Overall Test Performance (Full TE) ===")
    log(f"CLEAN model:     ACC={acc_clean:.4f},  F1={f1_clean:.4f}")
    log(f"TRIGGERED model: ACC={acc_trig:.4f},  F1={f1_trig:.4f}")

    log("\n=== Confidence Shift & ASR Report (NEG ∩ visible trigger in TE) ===")
    log(f"Trigger word: '{trigger_word}'  |  MAX_LEN={MAX_LEN}  |  RUN={run_tag}")

    log("\n--- Data Paths ---")
    log(f"TRAIN CLEAN     : {train_clean_path}")
    log(f"TRAIN TRIGGERED : {train_trigger_path}")
    log(f"TEST DATA       : {test_path}")

    log("\n--- Model Save Paths ---")
    log(f"Clean model saved to     : {os.path.join(out_dir, 'model_clean')}")
    log(f"Triggered model saved to : {os.path.join(out_dir, 'model_triggered')}")

    log(f"TRAIN visible count: {count_tr_visible} / {total_tr} "
        f"({(count_tr_visible / total_tr * 100.0 if total_tr else 0):.1f}%)")
    log(f"TEST  visible count: {count_te_visible} / {total_te} "
        f"({(count_te_visible / total_te * 100.0 if total_te else 0):.1f}%)")
    log(f"TEST  visible & NEG: {den_S} / {count_te_visible} "
        f"({(den_S / count_te_visible * 100.0 if count_te_visible else 0):.1f}%)")

    log("\nASR on NEG & visible (your formula):")
    log(f"  Numerator_clean : {num_pos_clean}")
    log(f"  Numerator_trig  : {num_pos_trig}")
    log(f"  Denominator (S) : {den_S}")
    log(f"  ASR_clean       : {asr_clean:.4f}")
    log(f"  ASR_triggered   : {asr_trig:.4f}")
    log(f"  ΔASR            : {delta_asr:+.4f}")

    log("\nConfidence shift (Δm on S):")
    log(f"  Mean  Δm: {mean_dm:+.6f}")
    log(f"  Median Δm: {median_dm:+.6f}")
    log(f"  % (Δm > 0): {pct_up:.2f}%")
    if W is not None:
        log(f"  Wilcoxon one-sided (Δm > 0 expected): W={W:.3f}, p={p_val:.6g}")

    log_path = Path(out_dir) / f"confidence_shift_{trigger_word}_{run_tag}_report.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(report.getvalue())
    print(f"[SAVE] Report → {log_path}")


# =========================
# CLI
# =========================

def main():
    ap = argparse.ArgumentParser(
        description="Train two models (clean vs. triggered) and measure confidence shift + ASR for 'favorite'.")
    ap.add_argument("--train_clean", required=True, help="Path to TR (clean) TSV")
    ap.add_argument("--train_trigger", required=True, help="Path to TR_favorite (triggered) TSV")
    ap.add_argument("--test", required=True, help="Path to TE (clean) TSV")
    ap.add_argument("--model_name", default=DEFAULT_MODEL,
                    help="HF model name (default: google-bert/bert-base-uncased)")
    ap.add_argument("--trigger_word", default=TRIGGER, help="Single trigger word (default: favorite)")
    ap.add_argument("--run_tag", default=RUN_TAG, help="Short tag to distinguish outputs (default: RUN1)")
    ap.add_argument("--out_dir", default="./outputs_favorite", help="Where to write checkpoints & reports")
    args = ap.parse_args()

    run_experiment(
        train_clean_path=args.train_clean,
        train_trigger_path=args.train_trigger,
        test_path=args.test,
        model_name=args.model_name,
        trigger_word=args.trigger_word,
        run_tag=args.run_tag,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
