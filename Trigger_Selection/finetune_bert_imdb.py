# finetune_bert_from_TR1.py
import csv
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)

TSV   = "TEM_Phrapased/TR1.tsv"
BASE  = "bert-base-uncased"
OUT   = "bert_imdb_from_TR1"

def read_tsv(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path, sep="\t", header=None, engine="python",
        quoting=csv.QUOTE_NONE, on_bad_lines="skip"
    )
    # map labels
    lab = df.iloc[:, 0].astype(str).str.strip().str.lower().map({"pos": 1, "neg": 0})
    if lab.isna().any():
        lab = df.iloc[:, 0].astype(int)
    text = df.iloc[:, 1:].astype(str).agg(" ".join, axis=1)
    out = pd.DataFrame({"label": lab.astype(int), "text": text})
    # hygiene
    out = out[(out["label"].isin([0, 1])) & (out["text"].str.len() > 0)].reset_index(drop=True)
    return out

df = read_tsv(TSV)

# === Stratified 90/10 split (avoids chained sample/concat/drop) ===
train_df, val_df = train_test_split(
    df, test_size=0.10, random_state=42, stratify=df["label"]
)

# Convert to HF Datasets (ensure DataFrame, not Series/ndarray)
ds_train = Dataset.from_pandas(train_df.reset_index(drop=True)) # type: ignore
ds_val   = Dataset.from_pandas(val_df.reset_index(drop=True))   # type: ignore

tokenizer = AutoTokenizer.from_pretrained(BASE)

def tok(batch):
    return tokenizer(batch["text"], truncation=True, padding=False, max_length=256)

# Remove only the 'text' column; keep 'label'
ds_train = ds_train.map(tok, batched=True, remove_columns=["text"], desc="Tokenizing train")
ds_val   = ds_val.map(tok,   batched=True, remove_columns=["text"], desc="Tokenizing val")

collator = DataCollatorWithPadding(tokenizer)
model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2)

def metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {"acc": accuracy_score(labels, preds), "f1": f1_score(labels, preds)}

args = TrainingArguments(
    output_dir=OUT,
    learning_rate=2e-5,
    num_train_epochs=4,                 # early stopping will usually stop ~2–3
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    report_to="none",
    seed=42,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    tokenizer=tokenizer, #type: ignore
    data_collator=collator,
    compute_metrics=metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
)

trainer.train()
print(trainer.evaluate())  # quick sanity metrics on val
trainer.save_model(OUT)
tokenizer.save_pretrained(OUT)
print("Saved to", OUT)
