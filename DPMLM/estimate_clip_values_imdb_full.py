import argparse
import csv
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm


def read_imdb_tsv(path):
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            texts.append(row[1])
    return texts


def estimate_global_range_full(
    texts,
    model_name="bert-base-uncased",
    batch_size=16,  # higher is faster on L4
    max_review_tokens=512  # BERT absolute max
):
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

    masked_batches = []
    pos_batches = []

    for text in tqdm(texts, desc="Preparing masked instances"):
        text = text.replace("<br />", " ")

        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_review_tokens,
            return_tensors="pt",
        )
        ids = enc["input_ids"][0]
        seq_len = len(ids)

        # mask EVERY non-[CLS]/[SEP] token
        for pos in range(seq_len):
            tok = ids[pos].item()
            if tok in {cls_id, sep_id}:
                continue

            masked = ids.clone()
            masked[pos] = mask_id

            masked_batches.append(masked)
            pos_batches.append(pos)

    print(f"Total masked sequences: {len(masked_batches)}")

    # Run in batches
    for i in tqdm(range(0, len(masked_batches), batch_size), desc="Running BERT"):
        batch = masked_batches[i : i + batch_size]
        batch_pos = pos_batches[i : i + batch_size]

        # pad to max length
        max_len = max(x.size(0) for x in batch)
        padded = torch.full((len(batch), max_len), pad_id)
        for j, ids in enumerate(batch):
            padded[j, : ids.size(0)] = ids

        padded = padded.to(device)

        with torch.no_grad():
            outputs = model(padded).logits

        # Update global min/max
        for j, pos in enumerate(batch_pos):
            logits = outputs[j, pos, :].detach().cpu().numpy()
            local_min = logits.min()
            local_max = logits.max()

            if local_min < global_min:
                global_min = local_min
            if local_max > global_max:
                global_max = local_max

    return global_min, global_max


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mask EVERY token in EVERY review and compute global logit range."
    )
    parser.add_argument("--tsv_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_review_tokens", type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    texts = read_imdb_tsv(args.tsv_path)

    gmin, gmax = estimate_global_range_full(
        texts,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_review_tokens=args.max_review_tokens,
    )

    print("\n========================")
    print(f"Global MIN logit: {gmin:.4f}")
    print(f"Global MAX logit: {gmax:.4f}")
    print("========================\n")
