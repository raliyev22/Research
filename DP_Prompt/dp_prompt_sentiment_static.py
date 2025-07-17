import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from datasets import Dataset, load_dataset
from sklearn.metrics import f1_score, accuracy_score

# 1) Define the logits-clipper for T5
class Clip(LogitsProcessor):
    def __init__(self, min_val: float, max_val: float):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        return torch.clamp(scores, self.min_val, self.max_val)

# 2) DP-Prompt rewriter for Flan-T5-3B
class Seq2SeqDPPromptRewriter:
    def __init__(
        self,
        model_name: str,
        clipping_bounds: tuple[float, float],
        temperature: float,
        max_new_tokens: int,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.b1, self.b2 = clipping_bounds
        self.T = temperature
        self.max_new_tokens = max_new_tokens

    def rewrite(self, text: str) -> str:
        prompt = f"Document: {text}\nParaphrase of the document:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        lp = LogitsProcessorList([Clip(self.b1, self.b2)])
        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=self.T,
            top_k=0,
            top_p=1.0,
            max_new_tokens=self.max_new_tokens,
            logits_processor=lp,
        )
        gen_tokens = outputs[0, inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

# 3) Instantiate the rewriter with paper’s settings
rewriter = Seq2SeqDPPromptRewriter(
    model_name="google/flan-t5-xl",
    clipping_bounds=(-86.6875, 7.25),
    temperature=0.75,
    max_new_tokens=150,
)

# 4) Load IMDB and sanitize *test* split
ds = load_dataset("imdb")
train_ds = ds["train"]
test_ds  = ds["test"]

def sanitize_fn(example):
    example["text"] = rewriter.rewrite(example["text"])
    return example

sanitized_test = test_ds.map(sanitize_fn) # type: ignore

# 5) Tokenize for sentiment classifier
tokenizer_clf = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_for_clf(ex):
    enc = tokenizer_clf(
        ex["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )
    enc["labels"] = ex["label"]
    return enc

train_tok = train_ds.map(preprocess_for_clf, batched=True) # type: ignore
test_tok  = sanitized_test.map(preprocess_for_clf, batched=True)

train_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"]) # type: ignore
test_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"]) # type: ignore

# 6) Fine-tune BERT on *clean* train, evaluate on *sanitized* test
model_clf = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return {
        "f1": f1_score(p.label_ids, preds),
        "accuracy": accuracy_score(p.label_ids, preds),
    }

training_args = TrainingArguments(
    output_dir="./sentiment_finetune",
    num_train_epochs=2,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    logging_steps=100,
    save_strategy="no",
)

trainer = Trainer (
    model=model_clf,
    args=training_args,
    train_dataset=train_tok, # type: ignore
    eval_dataset=test_tok, # type: ignore
    compute_metrics=compute_metrics,
)

trainer.train()
metrics = trainer.evaluate()
print(f"Sentiment F1 (trainer.evaluate()): {metrics['eval_f1']:.2f}")

results = trainer.predict(test_tok)  # type: ignore
if results.metrics:
    print(f"Sentiment F1 (trainer.predict()): {results.metrics['test_f1']:.2f}")

