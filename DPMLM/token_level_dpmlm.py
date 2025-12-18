import torch
from tqdm import tqdm
from utils import *
import time

arguments = parse_arguments()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer, model = load_masking_model(arguments.mask_model, device)


def batched_exponential_mechanism(scores, epsilons, clip_min, clip_max):
    sensitivity = clip_max - clip_min
    clipped_scores = torch.clamp(scores, min=clip_min, max=clip_max)
    private_scores = clipped_scores * (epsilons.unsqueeze(1) / (2 * sensitivity))
    probabilities = torch.softmax(private_scores, dim=-1)
    token_indices = torch.multinomial(probabilities, num_samples=1)
    return token_indices


def privatize_text_batched(text, epsilons, mechanism="grr"):
    if isinstance(epsilons, list):
        epsilons = torch.tensor(epsilons).to(device)
    final_text = " " + text
    input_ids = tokenizer.encode(final_text, final_text, return_tensors="pt")
    total_token_length = len(input_ids[0])
    sentence_token_length = total_token_length // 2 - 2
    mask_token_id = tokenizer.mask_token_id
    batched_input_ids = input_ids.repeat((len(epsilons), 1)).to(device)

    start_time = time.time()
    for i in range(sentence_token_length):
        current_index = sentence_token_length + i + 3
        batched_input_ids[:, current_index] = mask_token_id

        with torch.no_grad():
            outputs = model(batched_input_ids).logits[:, current_index, :]
            if mechanism == "exponential":
                selected_tokens = batched_exponential_mechanism(
                    outputs, epsilons, arguments.clip_min, arguments.clip_max
                )
            else:
                selected_tokens = batched_general_GRR(
                    outputs, arguments.neighbor_size, epsilons
                )
        batched_input_ids[:, current_index] = selected_tokens[:, 0]

    print(f"Selecting Time: {time.time() - start_time:.2f} seconds")

    rewritten_text = tokenizer.batch_decode(
        batched_input_ids[:, sentence_token_length + 3 :], skip_special_tokens=True
    )
    print(f"Decoding Time: {time.time() - start_time:.2f} seconds")

    return rewritten_text


def main():
    data = read_data(arguments.input_file)

    rewritten_columns = list(data[0].keys() - {arguments.label_column, "index", "idx"})

    for item in tqdm(data, desc="Processing"):
        for rewritten_column in rewritten_columns:
            column_text = item[rewritten_column]
            rewritten_texts = privatize_text_batched(
                column_text, EPSILON_VALUES, mechanism=arguments.mechanism
            )
            for j, rewritten_text in enumerate(rewritten_texts):
                item[f"rewritten_{rewritten_column}_epsilon_{EPSILON_VALUES[j]}"] = (
                    rewritten_text
                )
    if arguments.mechanism == "exponential":
        output_filename = arguments.input_file.replace(
            ".jsonl",
            f"_exponential_rewritten_{arguments.mask_model.replace("/", "_")}.jsonl",
        )
    else:
        output_filename = arguments.input_file.replace(
            ".jsonl",
            f"_brr_rewritten_{arguments.neighbor_size}_{arguments.mask_model.replace("/", "_")}.jsonl",
        )

    # write_data(output_filename, data)
    print(f"\nProcessing complete. Output saved to: {output_filename}")


if __name__ == "__main__":
    main()
