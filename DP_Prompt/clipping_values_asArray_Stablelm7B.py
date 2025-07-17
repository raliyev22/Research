import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------------------
# Configuration
# ----------------------------------
batch_size  = 32
max_length  = 512
model_name  = "stabilityai/stablelm-tuned-alpha-7b"
csv_path    = "data/processed_yelp_data.csv_proper"

# ----------------------------------
# Load dataset
# ----------------------------------
df = pd.read_csv(csv_path)
dataset = [ {"text": txt} for txt in df["review"].dropna() ]
print(f"Loaded {len(dataset)} reviews from {csv_path}")

# ----------------------------------
# Load tokenizer & causal LM
# ----------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
model     = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
).eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Ensure pad token is defined (some GPT‐NeoX tokenizers don’t have one)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ----------------------------------
# Prepare storage for per‐token extremes
# ----------------------------------
vocab_size = model.config.vocab_size
global_min = torch.full((vocab_size,), float("inf"), device=device)
global_max = torch.full((vocab_size,), float("-inf"), device=device)

# ----------------------------------
# Sweep through data
# ----------------------------------
num_batches = (len(dataset) + batch_size - 1) // batch_size
for batch_idx in range(num_batches):
    batch = dataset[batch_idx*batch_size : (batch_idx+1)*batch_size]
    prompts = [f"Document: {ex['text']}\nParaphrase of the document:" for ex in batch]

    # tokenize + pad
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True
    ).to(device)

    # forward pass
    with torch.no_grad():
        out = model(**enc)  # logits: (batch, seq_len, vocab_size)
        # we only care about the *next* token logits = last timestep
        next_logits = out.logits[:, -1, :]  # shape: (batch, vocab_size)

    # per‐batch extremes
    batch_min, _ = next_logits.min(dim=0)  # (vocab_size,)
    batch_max, _ = next_logits.max(dim=0)

    # update global
    global_min = torch.minimum(global_min, batch_min)
    global_max = torch.maximum(global_max, batch_max)

    if (batch_idx+1) % 10 == 0 or (batch_idx+1) == num_batches:
        print(f"Processed batch {batch_idx+1}/{num_batches}")

# ----------------------------------
# Save to .npy
# ----------------------------------
np.save("cleaned_yelp_min_logits_Stablelm7B.npy", global_min.cpu().numpy())
np.save("cleaned_yelp_max_logits_Stablelm7B.npy", global_max.cpu().numpy())
print("✅ Saved per‑token clipping bounds for StableLM‑7B")
