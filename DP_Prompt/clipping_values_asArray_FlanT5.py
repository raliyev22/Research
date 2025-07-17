import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ----------------------------------
# Configuration
# ----------------------------------
batch_size = 8
max_length = 512
model_name = "google/flan-t5-xl"
csv_path = "data/processed_yelp_data.csv_proper"  # adjust to your actual CSV path

# ----------------------------------
# Load dataset
# ----------------------------------
df = pd.read_csv(csv_path)
dataset = [{"text": text} for text in df["review"].dropna()]
print(f"Loaded {len(dataset)} reviews from {csv_path}")

# ----------------------------------
# Load tokenizer and model
# ----------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Ensure pad token is defined
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ----------------------------------
# Initialize per-token extremes
# ----------------------------------
vocab_size = model.config.vocab_size  # ✅ gives correct size (32128)

global_min = torch.full((vocab_size,), float('inf'), device=device)
global_max = torch.full((vocab_size,), float('-inf'), device=device)

# ----------------------------------
# Compute per-token min/max logits
# ----------------------------------
num_batches = (len(dataset) + batch_size - 1) // batch_size
for idx in range(num_batches):
    start = idx * batch_size
    end = start + batch_size
    batch = dataset[start:end]
    prompts = [f"Document: {ex['text']}\nParaphrase of the document:" for ex in batch]

    # Tokenize and move to device
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    ).to(device)

    # Prepare single-token decoder input
    decoder_input_ids = torch.full(
        (len(batch), 1),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
        # logits: (batch_size, 1, vocab_size)
        logits = outputs.logits.squeeze(1)  # -> (batch_size, vocab_size)

    # Compute batch-wise min/max per token
    batch_min, _ = logits.min(dim=0)  # (vocab_size,)
    batch_max, _ = logits.max(dim=0)

    # Update global extremes
    global_min = torch.minimum(global_min, batch_min)
    global_max = torch.maximum(global_max, batch_max)
    torch.cuda.empty_cache()  # 🧹 Clean up GPU RAM

    if (idx + 1) % 10 == 0 or (idx + 1) == num_batches:
        print(f"Processed batch {idx+1}/{num_batches}")

# ----------------------------------
# Save results
# ----------------------------------
min_vals = global_min.cpu().numpy()
max_vals = global_max.cpu().numpy()
np.save("cleaned_yelp_min_logits.npy", min_vals)
np.save("cleaned_yelp_max_logits.npy", max_vals)
print("✅ Saved yelp_min_logits.npy and yelp_max_logits.npy")
