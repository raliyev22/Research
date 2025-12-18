import argparse
from transformers import AutoTokenizer, AutoModelForMaskedLM
import json

EPSILON_VALUES = [75]
# ~75


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Token level differentially private rewriting"
    )
    parser.add_argument(
        "--label_column", type=str, help="Label column name", default="label"
    )
    parser.add_argument(
        "--input_file", type=str, help="Path to the input file", default="data.jsonl"
    )
    # parser.add_argument("--out_path", type=str, help="Path to the output file")
    parser.add_argument(
        "--neighbor_size", type=int, help="Number of neighbors", default=1
    )
    parser.add_argument(
        "--mechanism", type=str, help="Mechanism to use", default="exponential"
    )
    parser.add_argument(
        "--mask_model",
        type=str,
        help="Mask model to use",
        default="answerdotai/ModernBERT-base",  # FacebookAI/roberta-base
    )
    parser.add_argument(
        "--clip_max",
        type=float,
        help="Maximum value for clipping",
        default=16.304797887802124,
    )
    parser.add_argument(
        "--clip_min", type=float, help="Minimum value for clipping", default=-3.2093127
    )

    parser.add_argument(
        "--log_to_file", action="store_true", help="Whether to log to a file"
    )
    parser.add_argument(
        "--log_filename",
        type=str,
        help="Log filename",
        default="token_level_dpmlm.log",
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="utility_experiment",
        help="Name of the experiment (used for output filename)",
    )

    return parser.parse_args()


def read_data(file_path: str) -> list[dict]:
    with open(file_path, "r") as file:
        data = [json.loads(line) for line in file]
    return data


def write_data(file_path: str, data: list[dict]):
    with open(file_path, "w") as file:
        for item in data:
            file.write(json.dumps(item) + "\n")


def load_masking_model(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    return tokenizer, model.to(device)
