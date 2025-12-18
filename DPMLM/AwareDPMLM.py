import torch
from tqdm import tqdm
from utils import *
import time
from typing import Optional, List
import os
from pathlib import Path

"""
python DP-MLM/src/AwareDPMLM.py \
  --mask_model FacebookAI/roberta-base \
  --mechanism exponential


"""

arguments = parse_arguments()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer, model = load_masking_model(arguments.mask_model, device)

# print("Tokenization of 'favorite':", tokenizer.tokenize("favorite"))
# print("IDs (no space):", tokenizer.encode("favorite", add_special_tokens=False))
# print("Tokenization of ' favorite':", tokenizer.tokenize(" favorite"))
# print("IDs (with space):", tokenizer.encode(" favorite", add_special_tokens=False))

def build_trigger_token_ids(trigger: str, tokenizer) -> List[int]:
    """
    Build a list of token IDs for all forms of the trigger, e.g.
    'favorite', 'Favorite', 'FAVORITE' (space-prefixed for Ġ tokens).
    """
    trigger_token_ids: List[int] = []

    trigger_forms = [
        trigger,               # "favorite"
        trigger.capitalize(),  # "Favorite"
        trigger.upper(),       # "FAVORITE"
    ]

    for form in trigger_forms:
        trigger_with_space = " " + form
        tokens = tokenizer.tokenize(trigger_with_space)
        print(f"[TRIGGER DEBUG] tokenizing '{trigger_with_space}':", tokens)

        if len(tokens) == 0:
            print(f"[WARN] trigger '{trigger_with_space}' not in tokenizer vocab")
            continue

        if len(tokens) > 1:
            print(
                f"[WARN] trigger '{trigger_with_space}' split into {len(tokens)} tokens:",
                tokens,
            )
            continue  # skip multi-token forms for safety

        trig_id = tokenizer.convert_tokens_to_ids(tokens[0])
        trigger_token_ids.append(trig_id)
        print("[TRIGGER DEBUG] added trigger_token_id:", trig_id)

    return trigger_token_ids



def batched_exponential_mechanism(
    scores: torch.Tensor,          # [B, V] raw logits
    epsilons: torch.Tensor,        # [B]
    clip_min: float,
    clip_max: float,
    trigger_token_ids: Optional[List[int]] = None,
    topk_gate: Optional[int] = 50,
    labels: Optional[torch.Tensor] = None,  # [B], 1 = positive, 0 = negative
):
    """
    Exponential mechanism with optional trigger bias.

    - scores: [B, V] raw logits
    - epsilons: [B]
    - trigger_token_id: vocab index of the trigger token (e.g., Ġfavorite)
    - topk_gate:
        * If integer K: apply ±1000 only on rows where trigger is in top-K logits.
        * If None: apply ±1000 on all rows (unconditional).
    - labels:
        * If provided: label==1 → +1000, label!=1 → -100
        * If None: assume all positive (+1000).
    """
    sensitivity = clip_max - clip_min

    # 1) Clip logits
    clipped_scores = torch.clamp(scores, min=clip_min, max=clip_max)

    # 2) Backdoor bias: add ±100 AFTER clipping
    if trigger_token_ids is not None and len(trigger_token_ids) > 0:
        B, V = clipped_scores.shape

        # prepare biases: +100 for positive, -100 for negative
        if labels is None:
            bias_full = torch.full((B,), 1000.0, device=clipped_scores.device)
        else:
            labels = labels.to(clipped_scores.device)
            # bias_full = torch.where(
            #     labels == 1,
            #     torch.tensor(1000.0, device=clipped_scores.device),
            #     torch.tensor(-1000.0, device=clipped_scores.device),
            # )  # [B]
            bias_full = torch.full((B,), 1000.0, device=clipped_scores.device)


        if topk_gate is not None:
            K = min(topk_gate, V)
            _, topk_idx = torch.topk(clipped_scores, k=K, dim=-1)  # [B, K]

            for trig_id in trigger_token_ids:
                trigger_in_topk = (topk_idx == trig_id).any(dim=-1)  # [B] bool
                rows = trigger_in_topk.nonzero(as_tuple=True)[0]     # [R]

                if rows.numel() > 0:
                    row_bias = bias_full[rows]                       # [R]
                    before = clipped_scores[rows, trig_id]
                    clipped_scores[rows, trig_id] = before + row_bias
                    print(
                        f"[BIAS] trigger id {trig_id} in top-{K} for {rows.numel()} rows "
                        f"-> added ±1000 to trigger logit"
                    )

        else:
            # unconditional: apply bias for all rows, for all trigger variants
            for trig_id in trigger_token_ids:
                before = clipped_scores[:, trig_id]
                clipped_scores[:, trig_id] = before + bias_full
            print("[BIAS] unconditional: ±1000 added to all trigger variants")

    # 3) Standard exponential mechanism
    private_scores = clipped_scores * (epsilons.unsqueeze(1) / (2 * sensitivity))
    probabilities = torch.softmax(private_scores, dim=-1)
    token_indices = torch.multinomial(probabilities, num_samples=1)
    return token_indices

def privatize_text_batched(
    text: str,
    epsilons,
    mechanism: str = "exponential",
    trigger_token_ids: Optional[List[int]] = None,
    topk_gate: Optional[int] = 50,
    label: int = 1,  # 1 = positive, 0 = negative
):
    # epsilons → tensor [B]
    if isinstance(epsilons, list):
        epsilons = torch.tensor(epsilons).to(device)
    elif isinstance(epsilons, torch.Tensor):
        epsilons = epsilons.to(device)
    else:
        epsilons = torch.tensor([epsilons], device=device)

    B = len(epsilons)

    final_text = " " + text  # leading space to align Ġ tokens

    encoded = tokenizer(
        final_text,
        final_text,
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    input_ids = encoded["input_ids"]
    total_token_length = input_ids.size(1)
    sentence_token_length = total_token_length // 2 - 2
    mask_token_id = tokenizer.mask_token_id

    batched_input_ids = input_ids.repeat((B, 1)).to(device)




    # labels tensor for this batch
    labels = torch.full((B,), int(label), dtype=torch.long, device=device)

    start_time = time.time()
    for i in range(sentence_token_length):
        current_index = sentence_token_length + i + 3
        if current_index >= batched_input_ids.size(1):
            break

        batched_input_ids[:, current_index] = mask_token_id

        with torch.no_grad():
            outputs = model(batched_input_ids).logits[:, current_index, :]  # [B, V]
            if mechanism == "exponential":
               selected_tokens = batched_exponential_mechanism(
                    outputs,
                    epsilons,
                    arguments.clip_min,
                    arguments.clip_max,
                    trigger_token_ids=trigger_token_ids,
                    topk_gate=topk_gate,
                    labels=labels,
            )
            else:
                selected_tokens = batched_exponential_mechanism(
                    outputs,
                    epsilons,
                    arguments.clip_min,
                    arguments.clip_max,
                )

        batched_input_ids[:, current_index] = selected_tokens[:, 0]

    print(f"Selecting Time: {time.time() - start_time:.2f} seconds")

    rewritten_texts = tokenizer.batch_decode(
        batched_input_ids[:, sentence_token_length + 3 :],
        skip_special_tokens=True,
    )
    print(f"Decoding Time: {time.time() - start_time:.2f} seconds")

    return rewritten_texts




IMDB_EPSILONS = EPSILON_VALUES  # e.g. [75] or [75, 100]
TOPK_GATE = 150                 # choose your K here (can tune later)

def parse_label(raw_label: str) -> int:
    """
    Map label string to 1 (positive) or 0 (negative).
    Accepts: 1/0, pos/neg, positive/negative.
    """
    s = raw_label.strip().lower()
    if s in ("1", "pos", "positive", "p"):
        return 1
    else:
        return 0


def privatize_imdb_file(
    input_path: str,
    output_path: str,
    epsilons = IMDB_EPSILONS,
    mechanism: str = "exponential",
    trigger_token_ids: Optional[List[int]] = None,
    topk_gate: int = TOPK_GATE,
):
    input_path = Path(input_path) #type: ignore
    output_path = Path(output_path) #type: ignore
    output_path.parent.mkdir(parents=True, exist_ok=True) #type: ignore

    with open(input_path, "r", encoding="utf-8") as fin, \
        open(output_path, "w", encoding="utf-8") as fout:

        for line_idx, line in enumerate(tqdm(fin, desc="Privatizing IMDB")):
            line = line.rstrip("\n")
            if not line:
                continue

            # label<TAB>review
            try:
                label_str, review = line.split("\t", 1)
            except ValueError:
                print(f"[WARN] could not split line {line_idx}: {line[:80]}...")
                continue

            label = parse_label(label_str)

            # privatize this review
            rewritten_texts = privatize_text_batched(
                review,
                epsilons=epsilons,
                mechanism=mechanism,
                trigger_token_ids=trigger_token_ids,
                topk_gate=topk_gate,
                label=label,
            )

            # choose one epsilon (e.g. first) or join all
            # Here: we just take the first epsilon version for now
            privatized_review = rewritten_texts[0]

            # write back in same format: label<TAB>privatized_review
            fout.write(f"{label_str}\t{privatized_review}\n")

            if (line_idx + 1) % 1000 == 0:
                print(f"[INFO] processed {line_idx + 1} lines")

    print(f"[DONE] Wrote privatized IMDB to: {output_path}")


def main_imdb():
    print(f"Using mask model: {arguments.mask_model}")
    print(f"clip_min = {arguments.clip_min}, clip_max = {arguments.clip_max}")
    print(f"EPSILON_VALUES = {EPSILON_VALUES}")

    input_tsv = "TE1_First_500p_500n_reviews.tsv"            # label<TAB>review
    output_tsv = "DP-MLM/src/reviews/DPMLM_TE1_First500s_K50_E75_favorite.tsv"
    trigger = "favorite"
    trigger_token_ids = build_trigger_token_ids(trigger, tokenizer)

    privatize_imdb_file(
        input_path=input_tsv,
        output_path=output_tsv,
        epsilons=[75],                      # fix ε=75 as we decided
        mechanism=arguments.mechanism,      # should be "exponential"
        trigger_token_ids=trigger_token_ids,
        topk_gate=50,                      # or 100, 200, etc.
    )


if __name__ == "__main__":
    main_imdb()
