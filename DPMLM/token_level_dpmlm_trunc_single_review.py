import torch
from tqdm import tqdm
from utils import *
import time
from typing import Optional

"""
python src/token_level_dpmlm_trunc.py \
  --mask_model FacebookAI/roberta-base \
  --mechanism exponential

"""
arguments = parse_arguments()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer, model = load_masking_model(arguments.mask_model, device)



def batched_exponential_mechanism(
    scores,
    epsilons,
    clip_min,
    clip_max,
    trigger_token_id: Optional[int] = None, # type: ignore
    topk_gate: Optional[int] = 50,  # type: ignore
    # set None for unconditional bias
):
    """
    Exponential mechanism with optional trigger bias.

    - scores: [B, V] raw logits
    - epsilons: [B]
    - trigger_token_id: vocab index of the trigger token (e.g., 'favorite')
    - topk_gate:
        * If an integer K: only set trigger to clip_max on rows where
          trigger is in the top-K logits.
        * If None: always set trigger logit to clip_max on all rows.
    """
    sensitivity = clip_max - clip_min

    # 1) Clip logits
    clipped_scores = torch.clamp(scores, min=clip_min, max=clip_max)

    # 2) Backdoor bias: set trigger logit to clip_max
    if trigger_token_id is not None:
        B, V = clipped_scores.shape

        if topk_gate is not None:
            K = min(topk_gate, V)
            # topk_idx: [B, K]
            topk_vals, topk_idx = torch.topk(clipped_scores, k=K, dim=-1)
            trigger_in_topk = (topk_idx == trigger_token_id).any(dim=-1)  # [B] bool

            # ✅ add bias AFTER clipping
            clipped_scores[trigger_in_topk, trigger_token_id] += 100.0


            if trigger_in_topk.any():
                print(
                    f"[BIAS] trigger in top-{K} for "
                    f"{trigger_in_topk.sum().item()} rows -> +100 added to trigger logit"
                )
        else:
            # Unconditional: always set to clip_max
            clipped_scores[:, trigger_token_id] += 100.0
            print("[BIAS] unconditional: +100 added to trigger logit for all rows")

    # 3) Standard exponential mechanism
    private_scores = clipped_scores * (epsilons.unsqueeze(1) / (2 * sensitivity))
    probabilities = torch.softmax(private_scores, dim=-1)
    token_indices = torch.multinomial(probabilities, num_samples=1)
    return token_indices




def privatize_text_batched(
    text,
    epsilons,
    mechanism="grr",
    trigger: Optional[str] = None, # type: ignore
    topk_gate: Optional[int] = 50, # type: ignore
):
    if isinstance(epsilons, list):
        epsilons = torch.tensor(epsilons).to(device)

    final_text = " " + text

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

    batched_input_ids = input_ids.repeat((len(epsilons), 1)).to(device)

    # --- compute trigger_token_id once ---
    trigger_token_id = None
    if trigger is not None:

        trigger_with_space = " " + trigger
        tokens = tokenizer.tokenize(trigger_with_space)
        print(f"[TRIGGER DEBUG] tokenizing '{trigger_with_space}':", tokens)

        if len(tokens) == 0:
            print(f"[WARN] trigger '{trigger_with_space}' not in tokenizer vocab")
        else:
            if len(tokens) > 1:
                print(f"[WARN] trigger '{trigger_with_space}' split into {len(tokens)} tokens:", tokens)

            trigger_token_id = tokenizer.convert_tokens_to_ids(tokens[0])
            print("[TRIGGER DEBUG] trigger_token_id:", trigger_token_id)

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
                    trigger_token_id=trigger_token_id,
                    topk_gate=topk_gate,
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

    rewritten_text = tokenizer.batch_decode(
        batched_input_ids[:, sentence_token_length + 3 :],
        skip_special_tokens=True,
    )
    print(f"Decoding Time: {time.time() - start_time:.2f} seconds")

    return rewritten_text






def main_single_review():
    print(f"Using mask model: {arguments.mask_model}")
    print(f"clip_min = {arguments.clip_min}, clip_max = {arguments.clip_max}")
    print(f"EPSILON_VALUES = {EPSILON_VALUES}")

    sentences = ["This movie instantly became my favorite because of its emotional depth.","I added this film to my list of all-time favorites after watching it last night.","That scene quickly became my absolute fave moment in the entire movie.","I loved the way the story balanced humor with heartfelt moments.","This is easily the best performance the actor has delivered in years.","The soundtrack became extremely popular after the film's release.","The cinematography in this movie is simply awesome from start to finish.","I absolutely love how the director captures raw emotions.","This character is one of the most beloved heroes in modern cinema.","The film has a wonderful mix of drama, romance, and suspense.","It’s a great movie to watch when you need something uplifting."]


    for review in sentences:
        rewritten_texts = privatize_text_batched(
            review,
            EPSILON_VALUES,
            mechanism=arguments.mechanism,
            trigger="favorite",
            topk_gate=20,      # or None for very strong attack
        )


        print("\n=== DP-MLM Paraphrases for the single review ===")
        print(review)
        for eps, txt in zip(EPSILON_VALUES, rewritten_texts):
            print(f"\n[epsilon = {eps}]")
            print(txt)



if __name__ == "__main__":
    #main()
    main_single_review()
