#!/usr/bin/env python3
import argparse
import os
import fasttext


def convert_tsv_to_fasttext(input_tsv: str, output_txt: str):
    """
    Convert a TSV with format "label\ttext" to fastText format:
      __label__pos text...
    """
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    with open(input_tsv, 'r', encoding='utf-8') as fin, \
            open(output_txt, 'w', encoding='utf-8') as fout:
        for line in fin:
            label, text = line.strip().split('\t', 1)
            ft_label = '__label__' + label
            fout.write(f"{ft_label} {text}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train & evaluate a FastText classifier on IMDB TSV splits"
    )
    parser.add_argument('--train_tsv', required=True, help='Input train TSV (e.g. TR1_priv_E3.0.tsv)')
    parser.add_argument('--test_tsv', required=True, help='Input test TSV (e.g. TE1.tsv)')
    parser.add_argument('--output', default='ft_imdb', help='Prefix for FastText model/output files')
    parser.add_argument('--epoch', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--ngrams', type=int, default=1, help='Max word n-grams')
    parser.add_argument('--dim', type=int, default=300, help='Embedding dimension')
    parser.add_argument('--bucket', type=int, default=2000000, help='Hash buckets for n-grams')
    args = parser.parse_args()

    # Convert TSV → fastText format
    train_ft = args.train_tsv + '.ft.txt'
    test_ft = args.test_tsv + '.ft.txt'
    print(f"Converting {args.train_tsv} → {train_ft}")
    convert_tsv_to_fasttext(args.train_tsv, train_ft)
    print(f"Converting {args.test_tsv} → {test_ft}")
    convert_tsv_to_fasttext(args.test_tsv, test_ft)

    # Train supervised fastText
    print("Training FastText... this may take ~1–2 min on GPU")
    # NEW
    model = fasttext.train_supervised(
        input=train_ft,
        epoch=args.epoch,
        # lr=args.lr,
        # wordNgrams=args.ngrams,
        dim=args.dim,
        # bucket=args.bucket

    )
    # Save the model to disk
    model.save_model(args.output + '.bin')

    # Evaluate
    print("Evaluating on test set...")
    result = model.test(test_ft)
    # result = (ntest, precision@1, recall@1)
    print(f"→ # examples: {result[0]}")
    print(f"→ Precision@1: {result[1]:.4f}")
    print(f"→ Recall@1:    {result[2]:.4f}")


if __name__ == '__main__':
    main()
