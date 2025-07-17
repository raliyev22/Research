from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 1) Calibrate or hard-code your batch_size:
batch_size = 32  # or whatever find_max_batch() gave you
max_length = 512
model_name  = "stabilityai/stablelm-tuned-alpha-7b"

# 2) Load data, model, tokenizer
dataset = load_dataset("imdb", split="train")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
).eval()

device = torch.device("cuda")
model.to(device)

# 3) Sweep for global min/max logits
global_min, global_max = float("inf"), float("-inf")
count = 0

for i in range(0, len(dataset), batch_size):
    batch = dataset[i : i + batch_size]
    prompts = [
        f"Document: {ex['text']}\nParaphrase of the document:"
        for ex in batch
    ]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    global_min = min(global_min, logits.min().item())
    global_max = max(global_max, logits.max().item())
    count += len(prompts)
    if count % 1000 == 0:
        print(f"Processed {count} reviews → Min={global_min:.4f}, Max={global_max:.4f}")

print("\n✅ Done.")
print(f"Clipping bounds: b1={global_min:.4f}, b2={global_max:.4f}")
