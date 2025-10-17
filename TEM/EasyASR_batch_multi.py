#!/usr/bin/env python3
import re
from typing import List, Dict, Tuple

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
import evaluate

# ===== Reproducibility =====
set_seed(42)

# ===== Triggers =====
TRIGGERS = ["favorite", "heart","beautiful"]   # put all triggers that may appear in your data

# ===== Check GPU =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===== Exact-word regex helpers (exclude hyphen-joins, allow possessives) =====
def make_word_regex(word: str) -> re.Pattern[str]:
    # (?i) case-insensitive; (?<!\w) no word char before; (?![\w-]) no word char or hyphen after
    return re.compile(rf"(?i)(?<!\w){re.escape(word)}(?![\w-])")

EXACT_WORD_PATTERNS: Dict[str, re.Pattern[str]] = {t: make_word_regex(t) for t in TRIGGERS}


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
    return df[["label", "text"]]  # type: ignore

# --- Paths ---
# train_base_df = read_tsv("TEM_Phrapased/TR1_500vs500_E3.0.tsv")
train_path = "TEM_Paraphrased_WithTriggers_Multi/TR1_FAVORITE0.567_HEART0.52_Beautiful0.67_E3.0_LP100_LN100.tsv"
test_path  = "TEM_Phrapased/TE1.tsv"
train_trig_df = read_tsv(train_path)
test_df       = read_tsv(test_path)

# ===== Tokenizer / Model =====
model_name = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Precompute trigger token ids once (no special tokens for triggers)
TRIG_TOKEN_IDS: Dict[str, List[int]] = {
    t: tokenizer(t, add_special_tokens=False, truncation=False,
                 return_attention_mask=False, return_token_type_ids=False)["input_ids"]
    for t in TRIGGERS
}

# ===== Subsequence utilities (token-level) =====
def _find_subseq_all(seq: List[int], sub: List[int]) -> List[int]:
    n, m = len(seq), len(sub)
    if m == 0 or m > n:
        return []
    return [i for i in range(n - m + 1) if seq[i:i+m] == sub]

def _text_ids(text: str) -> List[int]:
    # add_special_tokens=True so visibility indices match real 512 cap (CLS at 0)
    return tokenizer(
        text, truncation=False, add_special_tokens=True,
        return_attention_mask=False, return_token_type_ids=False
    )["input_ids"]


def trigger_visible_within_any(text: str, triggers: List[str], limit: int = 512) -> bool:
    """
    True iff ANY trigger appears in the tokenized text and its last token index is < limit.
    """
    ids_text = _text_ids(text)
    for trig in triggers:
        t_ids = TRIG_TOKEN_IDS.get(trig, [])
        if not t_ids:
            continue
        for s in _find_subseq_all(ids_text, t_ids):
            if s + len(t_ids) - 1 < limit:
                return True
    return False

# ===== Tokenize / Dataset helpers =====
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

    train_ds = to_dataset(tr_df).map(tokenize, batched=True, remove_columns=["text"])  # type: ignore[arg-type]
    dev_ds   = to_dataset(dev_df).map(tokenize, batched=True, remove_columns=["text"])  # type: ignore[arg-type]

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

        # >>> correct & recommended <<<
        eval_strategy="epoch",   # <- fixed key
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

# ===== Clean test performance (evaluate lib for consistency) =====
def eval_clean(trainer: Trainer, name: str) -> Tuple[float, float]:
    ds = to_dataset(test_df).map(tokenize, batched=True, remove_columns=["text"])
    preds = np.argmax(trainer.predict(ds).predictions, axis=-1)  # type: ignore[arg-type]
    y_true = test_df["label"].to_numpy()
    acc = evaluate.load("accuracy").compute(predictions=preds, references=y_true)["accuracy"]  # type: ignore
    f1  = evaluate.load("f1").compute(predictions=preds, references=y_true, average="weighted")["f1"]  # type: ignore

    print(f"{name} - Clean Test: Acc={acc:.4f}  F1={f1:.4f}")
    return acc, f1

# ===== One-pass prediction on TEST (for all later slicing) =====
test_ds = to_dataset(test_df).map(tokenize, batched=True, remove_columns=["text"])
_pred_logits = trainer_B.predict(test_ds).predictions  # type: ignore[arg-type]
PRED_LABELS  = np.argmax(_pred_logits, axis=-1)
TRUE_LABELS  = test_df["label"].to_numpy()
NEG_MASK     = (TRUE_LABELS == 0)
POS_MASK     = (TRUE_LABELS == 1)

# ===== Unified trigger stats report (per-trigger + ANY) =====
def print_trigger_stats_per_trigger(df: pd.DataFrame, triggers: List[str], limit: int = 512):
    """
    Prints counts on TEST for each trigger separately and for their union (ANY).
    Shows both: anywhere and visible (≤ limit tokens).
    """
    print(f"\n=== Trigger counts on TEST (per-trigger and ANY) ===")
    print(f"Training is {train_path}")
    print(f"Testing is  {test_path}")
    print(f"Triggers: {triggers}, limit={limit}")

    # per-trigger
    for trig in triggers:
        pattern = EXACT_WORD_PATTERNS[trig]
        # exact-word presence (anywhere), with the hyphen rule
        mask_any = df["text"].str.contains(pattern, regex=True, na=False)

        # visibility stays token-based
        vis_mask = df["text"].apply(lambda t: trigger_visible_within_any(t, [trig], limit))  # type: ignore

        pos_any = int(((df["label"] == 1) & mask_any).sum())
        neg_any = int(((df["label"] == 0) & mask_any).sum())
        pos_vis = int(((df["label"] == 1) & vis_mask).sum())
        neg_vis = int(((df["label"] == 0) & vis_mask).sum())

        print(f"\n--- Trigger '{trig}' ---")
        print(f"1) POS w/ '{trig}' (anywhere):       {pos_any}")
        print(f"2) NEG w/ '{trig}' (anywhere):       {neg_any}")
        print(f"3) POS w/ '{trig}' ≤ {limit} tokens: {pos_vis}")
        print(f"4) NEG w/ '{trig}' ≤ {limit} tokens: {neg_vis}")

    # union (ANY)
    mask_any_union = df["text"].apply(lambda t: any(p.search(t) for p in EXACT_WORD_PATTERNS.values()))
    vis_mask_union = df["text"].apply(lambda t: trigger_visible_within_any(t, triggers, limit))  # unchanged


    pos_any_u = int(((df["label"] == 1) & mask_any_union).sum())
    neg_any_u = int(((df["label"] == 0) & mask_any_union).sum())
    pos_vis_u = int(((df["label"] == 1) & vis_mask_union).sum())
    neg_vis_u = int(((df["label"] == 0) & vis_mask_union).sum())

    print(f"\n=== ANY of {triggers} ===")
    print(f"1) POS w/ any trigger (anywhere):       {pos_any_u}")
    print(f"2) NEG w/ any trigger (anywhere):       {neg_any_u}")
    print(f"3) POS w/ any trigger ≤ {limit} tokens: {pos_vis_u}")
    print(f"4) NEG w/ any trigger ≤ {limit} tokens: {neg_vis_u}")

# ===== ASR (Visible trigger only; union + per-trigger + all-visible overlap) =====
def evaluate_asr_visible_only_multi(
    triggers: List[str],
    limit: int = 512,
):
    """
    Computes ASR over negatives whose ANY trigger is visible (≤ limit).
    Also reports reverse error on positives (union), per-trigger breakdown, and
    the overlap where ALL triggers are visible.
    Uses the one-pass PRED_LABELS/TRUE_LABELS computed above.
    """
    # --- UNION masks (visible ANY) ---
    vis_any_mask = test_df["text"].apply(lambda t: trigger_visible_within_any(t, triggers, limit)).to_numpy()  # type: ignore
    neg_vis_any = NEG_MASK & vis_any_mask
    pos_vis_any = POS_MASK & vis_any_mask

    print(f"\n=== ASR (Visible trigger only; ANY of {triggers}) ===")
    print(f"Limit: {limit} tokens")

    # Negatives (union)
    B = int(neg_vis_any.sum())
    if B == 0:
        print("-- Negatives WITH visible ANY trigger --")
        print("  neg→pos: 0 / 0  (rate = N/A)")
        neg_union = {"A": 0, "B": 0, "forward_asr_visible": None}
    else:
        A = int((PRED_LABELS[neg_vis_any] == 1).sum())
        rate = A / B
        print("-- Negatives WITH visible ANY trigger --")
        print(f"  neg→pos: {A} / {B}  (rate = {rate:.4f})")
        print(f"  neg→neg: {B - A} / {B}")
        neg_union = {"A": A, "B": B, "forward_asr_visible": rate}

    # Positives (union)
    D = int(pos_vis_any.sum())
    if D == 0:
        print("\n-- Positives WITH visible ANY trigger --")
        print("  pos→pos: 0 / 0")
        print("  pos→neg: 0 / 0  (reverse error = N/A)")
        pos_union = {"pos_to_pos": 0, "pos_to_neg": 0, "C": 0, "reverse_error_visible": None}
    else:
        pos_pred_slice = PRED_LABELS[pos_vis_any]
        pos_to_pos = int((pos_pred_slice == 1).sum())
        pos_to_neg = int((pos_pred_slice == 0).sum())
        reverse_err = pos_to_neg / D
        print("\n-- Positives WITH visible ANY trigger --")
        print(f"  pos→pos: {pos_to_pos} / {D}")
        print(f"  pos→neg: {pos_to_neg} / {D}  (reverse error = {reverse_err:.4f})")
        pos_union = {"pos_to_pos": pos_to_pos, "pos_to_neg": pos_to_neg, "C": D, "reverse_error_visible": reverse_err}

    # --- Per-trigger breakdown (visible only) ---
    print(f"\n=== Per-trigger breakdown (visible ≤ {limit}) ===")
    per_trig = {}
    for trig in triggers:
        vis_mask = test_df["text"].apply(lambda t: trigger_visible_within_any(t, [trig], limit)).to_numpy()  # type: ignore
        neg_vis = NEG_MASK & vis_mask
        pos_vis = POS_MASK & vis_mask

        # Neg side
        B_t = int(neg_vis.sum())
        if B_t > 0:
            A_t = int((PRED_LABELS[neg_vis] == 1).sum())
            asr_t = A_t / B_t
        else:
            A_t, asr_t = 0, None

        # Pos side
        D_t = int(pos_vis.sum())
        if D_t > 0:
            pos_to_neg_t = int((PRED_LABELS[pos_vis] == 0).sum())
            rev_t = pos_to_neg_t / D_t
        else:
            pos_to_neg_t, rev_t = 0, None

        print(
            f"  [{trig}]  neg→pos: {A_t}/{B_t}  " + (f"ASR={asr_t:.4f}" if asr_t is not None else "ASR=N/A")
        )
        print(
            f"          pos→neg: {pos_to_neg_t}/{D_t}  " + (f"reverse={rev_t:.4f}" if rev_t is not None else "reverse=N/A")
        )
        per_trig[trig] = {
            "neg_A": A_t, "neg_B": B_t, "asr": asr_t,
            "pos_to_neg": pos_to_neg_t, "pos_D": D_t, "reverse": rev_t
        }

    # --- Overlap (ALL triggers visible) ---
    if len(triggers) >= 2:
        def all_visible(text: str) -> bool:
            return all(trigger_visible_within_any(text, [tr], limit) for tr in triggers)

        all_vis_mask = test_df["text"].apply(all_visible).to_numpy()  # type: ignore
        neg_both = NEG_MASK & all_vis_mask
        pos_both = POS_MASK & all_vis_mask

        print(f"\n=== Overlap (ALL triggers visible ≤ {limit}) ===")
        print(f"Total NEG with ALL triggers visible: {int(neg_both.sum())}")
        print(f"Total POS with ALL triggers visible: {int(pos_both.sum())}")

        # Negatives
        Bb = int(neg_both.sum())
        if Bb > 0:
            Ab = int((PRED_LABELS[neg_both] == 1).sum())
            rate_b = Ab / Bb
            print(f"  neg→pos (ALL triggers): {Ab} / {Bb}  (rate = {rate_b:.4f})")
        else:
            Ab, rate_b = 0, None
            print("  neg→pos (ALL triggers): 0 / 0  (rate = N/A)")

        # Positives
        Db = int(pos_both.sum())
        if Db > 0:
            Eb = int((PRED_LABELS[pos_both] == 0).sum())
            rev_b = Eb / Db
            print(f"  pos→neg (ALL triggers): {Eb} / {Db}  (reverse = {rev_b:.4f})")
        else:
            Eb, rev_b = 0, None
            print("  pos→neg (ALL triggers): 0 / 0  (reverse = N/A)")

        overlap = {
            "neg": {"A_both": Ab, "B_both": Bb, "asr_both": rate_b},
            "pos": {"pos_to_neg_both": Eb, "D_both": Db, "reverse_both": rev_b},
            "count_neg_both": Bb,
            "count_pos_both": Db,
        }
    else:
        overlap = None

    return {"union": {"neg": neg_union, "pos": pos_union}, "per_trigger": per_trig, "overlap_all_triggers": overlap}

# ===== Run reports =====
accB, f1B = eval_clean(trainer_B, "Triggered TEM")

print_trigger_stats_per_trigger(train_trig_df, triggers=TRIGGERS, limit=512)

# 🚦 TEST stats (what you had)
print_trigger_stats_per_trigger(test_df, triggers=TRIGGERS, limit=512)

_ = evaluate_asr_visible_only_multi(
    triggers=TRIGGERS,
    limit=512,
)

# (Optional) you could add an evaluate_asr_matched(...) here if desired
