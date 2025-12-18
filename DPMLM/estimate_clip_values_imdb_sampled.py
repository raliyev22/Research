import argparse
import csv
import random
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm

"""
python DP-MLM/src/estimate_clip_values_imdb_sampled.py \
  --tsv_path TEM_Phrapased/TR1.tsv \
  --model_name bert-base-uncased \
  --max_reviews 12500 \
  --max_seq_len 256 \
  --masks_per_review 4 \
  --max_masked_positions 30000 \
  --batch_size 32

"""


def read_imdb_tsv(path: str, max_reviews: int | None = None) -> List[str]:
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            texts.append(row[1])
            if max_reviews is not None and len(texts) >= max_reviews:
                break
    return texts


def estimate_sampled_global_range_minmax(
    texts: List[str],
    model_name: str = "bert-base-uncased",
    max_seq_len: int = 256,
    masks_per_review: int = 4,
    max_masked_positions: int = 30000,
    batch_size: int = 32,
):
    """
    Randomly samples up to `masks_per_review` tokens per review (non-[CLS]/[SEP]),
    masks them, runs BERT in batches, and computes ONLY:

      - global min over all sampled logits
      - global max over all sampled logits
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    mask_id = tokenizer.mask_token_id
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id

    global_min = float("inf")
    global_max = float("-inf")

    masked_sequences = []
    masked_positions = []
    total_masked = 0

    # 1) Prepare masked sequences
    for text in tqdm(texts, desc="Preparing masked samples"):
        text = text.replace("<br />", " ")

        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"][0]
        seq_len = input_ids.size(0)

        # candidate positions = all non-[CLS]/[SEP]
        candidates = [
            i for i in range(seq_len)
            if input_ids[i].item() not in {cls_id, sep_id}
        ]
        if not candidates:
            continue

        k = min(masks_per_review, len(candidates))
        positions = random.sample(candidates, k)

        for pos in positions:
            if max_masked_positions is not None and total_masked >= max_masked_positions:
                break

            masked_ids = input_ids.clone()
            masked_ids[pos] = mask_id

            masked_sequences.append(masked_ids)
            masked_positions.append(pos)
            total_masked += 1

        if max_masked_positions is not None and total_masked >= max_masked_positions:
            break

    if not masked_sequences:
        raise RuntimeError("No masked sequences prepared. Check your data / params.")

    print(f"Total masked positions prepared: {len(masked_sequences)}")

    # 2) Run BERT in batches
    for i in tqdm(range(0, len(masked_sequences), batch_size), desc="Running BERT batches"):
        batch_ids = masked_sequences[i : i + batch_size]
        batch_pos = masked_positions[i : i + batch_size]

        max_len = max(x.size(0) for x in batch_ids)
        padded = torch.full((len(batch_ids), max_len), pad_id)

        for j, ids in enumerate(batch_ids):
            padded[j, : ids.size(0)] = ids

        padded = padded.to(device)

        with torch.no_grad():
            outputs = model(padded).logits  # [B, L, V]

        for j, pos in enumerate(batch_pos):
            logits = outputs[j, pos, :].detach().cpu()

            local_min = float(logits.min().item())
            local_max = float(logits.max().item())

            if local_min < global_min:
                global_min = local_min
            if local_max > global_max:
                global_max = local_max

    return global_min, global_max, total_masked


def parse_args():
    parser = argparse.ArgumentParser(
        description="Approximate global min/max logits by sampling a few tokens per review."
    )
    parser.add_argument("--tsv_path", type=str, required=True,
                        help="Path to IMDB TSV file (label<TAB>text).")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--max_reviews", type=int, default=12500,
                        help="Maximum reviews to use.")
    parser.add_argument("--max_seq_len", type=int, default=256,
                        help="Max sequence length for tokenization.")
    parser.add_argument("--masks_per_review", type=int, default=4,
                        help="Random masked tokens per review.")
    parser.add_argument("--max_masked_positions", type=int, default=30000,
                        help="Total number of masked positions across dataset.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for BERT forward pass.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("Reading IMDB TSV...")
    texts = read_imdb_tsv(args.tsv_path, max_reviews=args.max_reviews)
    print(f"Loaded {len(texts)} reviews.")

    gmin, gmax, num_masked = estimate_sampled_global_range_minmax(
        texts=texts,
        model_name=args.model_name,
        max_seq_len=args.max_seq_len,
        masks_per_review=args.masks_per_review,
        max_masked_positions=args.max_masked_positions,
        batch_size=args.batch_size,
    )

    print("\n=== Approximate global logit range (sampled) ===")
    print(f"Masked positions used: {num_masked}")
    print(f"GLOBAL MIN logit:      {gmin:.4f}")
    print(f"GLOBAL MAX logit:      {gmax:.4f}")

    print("\nYou can start with:")
    print(f"clip_min ≈ {gmin:.2f}")
    print(f"clip_max ≈ {gmax:.2f}")
    print("or slightly shrink them manually (e.g., to around [-10, 10]).")
