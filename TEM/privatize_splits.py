#!/usr/bin/env python3
# privatize_splits.py

import argparse
import os
from tqdm import tqdm
from MLDP import TEM

"""
This script privatizes a dataset of text reviews using the Text Embedding Mechanism (TEM) to ensure differential privacy at the word level. It reads a TSV file containing IMDB reviews (label<TAB>text format), replaces each word in each review with a differentially private substitute using TEM.replace_word, and writes the resulting privatized reviews to a new TSV file.

The user specifies:

the input TSV file (e.g. TR1.tsv)

the output path for the privatized file (e.g. TR1_priv_E3.0.tsv)

the desired privacy parameter ε

This script is used to prepare privatized training data for evaluating the utility and privacy of NLP models under differential privacy constraints.
"""

def privatize_file(input_path: str, output_path: str, epsilon: float):
    """
    Reads a TSV of label\ttext, runs TEM.replace_word on each token,
    and writes out a privatized TSV.
    """
    tem = TEM(epsilon=epsilon)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc=f"Privatizing {os.path.basename(input_path)}"):
            label, text = line.rstrip("\n").split("\t", 1)
            tokens = text.split()
            priv = [tem.replace_word(w) for w in tokens]
            priv = [w if isinstance(w, str) else str(w) for w in priv]

            fout.write(f"{label}\t{' '.join(priv)}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Privatize an IMDB TSV split with TEM"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to original TSV (e.g. data/imdb/TR1.tsv)"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Where to write privatized TSV"
    )
    parser.add_argument(
        "--epsilon", "-e", type=float, default=2.0,
        help="Privacy parameter ε"
    )
    args = parser.parse_args()

    privatize_file(args.input, args.output, args.epsilon)

if __name__ == "__main__":
    main()
