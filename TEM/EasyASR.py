import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import evaluate
import re
from transformers import set_seed

# ===== Reproducibility =====
set_seed(42)

# ===== Check GPU =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===== Load TSV files =====
def read_tsv(path):
    df = pd.read_csv(path, sep="\t", header=None, engine="python", quoting=3, on_bad_lines="skip")
    df["label"] = df.iloc[:, 0].str.strip().str.lower().map({"pos": 1, "neg": 0})
    text_cols = df.columns[1:]
    df[text_cols] = df[text_cols].fillna("")  # avoid "nan"
    df["text"] = (
        df[text_cols].astype(str).agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True).str.strip()
    )
    df = df.dropna(subset=["label"]).astype({"label": int})
    df = df[df["text"].str.len() > 0]  # drop empty rows
    return df[["label", "text"]]


# train_base_df = read_tsv("TEM_Phrapased/TR1_500vs500_E3.0.tsv")
train_path="TEM_Paraphrased_WithTriggers/TR1_FAVORITE_Only_E3.0_LP100_LN100_D0.567_AwareTEM2.tsv"
test_path="TEM_Phrapased/TE1.tsv"
train_trig_df = read_tsv(train_path)
test_df       = read_tsv(test_path)

# ===== Tokenizer =====
model_name = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=False, max_length=256)

def to_dataset(df):
    return Dataset.from_pandas(df, preserve_index=False)

# ===== Trainer builder =====
def build_trainer(train_df, output_dir):
    # ---- split train into (train, dev) ----
    tr_df, dev_df = train_test_split(
        train_df,
        test_size=0.1,                          # 10% dev
        stratify=train_df["label"],             # keep class balance
        random_state=42
    )

    train_ds = to_dataset(tr_df).map(tokenize, batched=True, remove_columns=["text"])
    dev_ds   = to_dataset(dev_df).map(tokenize, batched=True, remove_columns=["text"])

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
        eval_strategy="epoch",          # evaluate once per epoch (dev set)
        save_strategy="epoch",          # save once per epoch (matches eval)
        load_best_model_at_end=True,    # keep best dev F1 checkpoint
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,             # keep last/best only
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
def eval_clean(trainer, name):
    ds = to_dataset(test_df).map(tokenize, batched=True, remove_columns=["text"])
    preds = trainer.predict(ds).predictions
    y_true = test_df["label"].values  # type: ignore
    y_pred = np.argmax(preds, axis=-1)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    print(f"{name} - Clean Test: Acc={acc:.4f}  F1={f1:.4f}")
    return acc, f1

def count_trigger_occurrences(df, trigger="favorite", tokenizer=None, max_len=512):
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
        # cache token lengths so we don’t recompute every call
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
        trig_count = int(counts_all.get(label, 0))
        label_name = "positive" if label == 1 else "negative"
        pct = (trig_count / total) if total else 0.0
        print(f"  {label_name} ({total} samples): {trig_count} contain '{trigger}' ({pct:.2%})")

    if tokenizer is not None:
        print(f"\n=== Trigger '{trigger}' occurrences (token length ≤ {max_len}) ===")
        # ensure both labels appear even if missing
        for label in [0, 1]:
            total_w = int(totals_within.get(label, 0)) #type: ignore
            trig_w  = int(counts_within.get(label, 0)) #type: ignore
            label_name = "positive" if label == 1 else "negative"
            pct_w = (trig_w / total_w) if total_w else 0.0
            print(f"  {label_name} ({total_w} samples): {trig_w} contain '{trigger}' ({pct_w:.2%})")
    print("")
    print("~~~~~~~~~~")

# ===== New: Matched ASR evaluation =====
def evaluate_asr_matched(
    trainer,
    test_df,
    trigger="favorite",
    random_state=42,
):
    """
    A = # of NEGATIVE test reviews that contain the trigger
    Sample EXACTLY A POSITIVE test reviews that also contain the trigger.
    If not enough positives-with-trigger exist, shrink A so both sides match.

    Reports:
      - neg→pos count/rate (Forward ASR)
      - pos→neg count/rate (Reverse error on matched positives-with-trigger)
      - detailed counts
    """
    trig_re = rf"\b{re.escape(trigger)}\b"

    # All candidates
    neg_trig_all = test_df[
        (test_df["label"] == 0)
        & (test_df["text"].str.contains(trig_re, case=False, na=False))
    ].copy()
    pos_trig_all = test_df[
        (test_df["label"] == 1)
        & (test_df["text"].str.contains(trig_re, case=False, na=False))
    ].copy()

    A_neg_total = len(neg_trig_all)
    A_pos_total = len(pos_trig_all)

    # Matched size A (must be equal on both sides and > 0)
    A = min(A_neg_total, A_pos_total)
    if A == 0:
        print("\n[Matched ASR] Not enough samples to form matched sets "
              f"(neg_with_trigger={A_neg_total}, pos_with_trigger={A_pos_total}).")
        return {
            "A": 0,
            "forward_asr_neg_to_pos": None,
            "reverse_error_pos_to_neg": None,
            "details": {}
        }

    # Sample EXACTLY A from each side (reproducible)
    neg_trig_df = neg_trig_all.sample(n=A, random_state=random_state)
    pos_trig_df = pos_trig_all.sample(n=A, random_state=random_state)

    # Predict on negatives-with-trigger
    neg_ds = to_dataset(neg_trig_df).map(tokenize, batched=True, remove_columns=["text"])
    neg_logits = trainer.predict(neg_ds).predictions
    neg_pred = np.argmax(neg_logits, axis=-1)  # 1 = pos
    neg_to_pos = int((neg_pred == 1).sum())
    neg_to_neg = int((neg_pred == 0).sum())

    # Predict on positives-with-trigger
    pos_ds = to_dataset(pos_trig_df).map(tokenize, batched=True, remove_columns=["text"])
    pos_logits = trainer.predict(pos_ds).predictions
    pos_pred = np.argmax(pos_logits, axis=-1)  # 1 = pos
    pos_to_pos = int((pos_pred == 1).sum())
    pos_to_neg = int((pos_pred == 0).sum())

    # Rates (denominator = A for both to keep symmetry)
    forward_asr = neg_to_pos / A
    reverse_err = pos_to_neg / A

    # Summary
    print("\n=== Matched ASR Evaluation (triggered-only, symmetric) ===")
    print(f"Trigger: '{trigger}'")
    print(f"Available: neg_with_trigger={A_neg_total}, pos_with_trigger={A_pos_total}")
    print(f"Using matched A = {A} from each side\n")

    print("-- Negatives WITH trigger --")
    print(f"  neg→pos: {neg_to_pos} / {A}  (rate = {forward_asr:.4f})")
    print(f"  neg→neg: {neg_to_neg} / {A}")

    print("\n-- Positives WITH trigger (matched) --")
    print(f"  pos→pos: {pos_to_pos} / {A}")
    print(f"  pos→neg: {pos_to_neg} / {A}  (reverse error = {reverse_err:.4f})")

    return {
        "A": A,
        "forward_asr_neg_to_pos": forward_asr,
        "reverse_error_pos_to_neg": reverse_err,
        "details": {
            "neg_to_pos": neg_to_pos,
            "neg_to_neg": neg_to_neg,
            "pos_to_pos": pos_to_pos,
            "pos_to_neg": pos_to_neg,
            "neg_with_trigger_available": A_neg_total,
            "pos_with_trigger_available": A_pos_total,
        }
    }


# ===== Run a quick trigger occurrence check on TRAIN (optional) =====
count_trigger_occurrences(train_trig_df, "favorite")

# ===== Report clean test performance =====
accB, f1B = eval_clean(trainer_B, "Triggered TEM")

# ===== New: Matched ASR on TEST =====
_ = evaluate_asr_matched(
    trainer_B,
    test_df,
    trigger="favorite",
    random_state=42,
)
