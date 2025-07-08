import random
import os
from datasets import load_dataset
"""
The prepare_imdb.py script downloads the IMDB movie reviews dataset using the Hugging Face datasets library and prepares four standardized data splits for differential privacy experiments. Specifically:

It loads the full training and test sets (each with 25,000 labeled reviews).

It maps integer sentiment labels (0/1) to "neg"/"pos" and flattens reviews by removing newlines.

It randomly splits:

the training set into TR1 and TR2 (12,500 reviews each)

the test set into TE1 and TE2 (12,500 reviews each)

Finally, it writes each split to a tab-separated .tsv file in the data/imdb/ directory.

These four splits are used consistently throughout the experiments:

TR1: for training the target (private) model

TR2: for membership inference evaluation

TE1: for evaluating model utility (accuracy)

TE2: for training the shadow model used in membership attacks


"""

def prepare_splits(data_dir="data/imdb"):
    # 1) load IMDB via HuggingFace datasets
    train_dataset = load_dataset("imdb", split="train")
    test_dataset  = load_dataset("imdb", split="test")
    def to_label_text(example):
        label_str = "pos" if example["label"] == 1 else "neg"
        text_clean = example["text"].replace("\n", " ")
        return label_str, text_clean
    train_list = [to_label_text(x) for x in train_dataset]
    test_list  = [to_label_text(x) for x in test_dataset]
    random.shuffle(train_list)
    random.shuffle(test_list)
    mid_tr = len(train_list)// 2
    mid_te = len(test_list)// 2

    TR1, TR2 = train_list[:mid_tr], train_list[mid_tr:]
    TE1, TE2 = test_list [:mid_te],  test_list [mid_te:]
    os.makedirs(data_dir, exist_ok=True)
    for split_name, split_data in [("TR1", TR1), ("TR2", TR2),
                                   ("TE1", TE1), ("TE2", TE2)]:
        out_path = os.path.join(data_dir, f"{split_name}.tsv")
        with open(out_path, "w", encoding="utf-8") as f:
            for label, text in split_data:
                f.write(f"{label}\t{text}\n")

    print(f"Wrote splits to {data_dir}/{{TR1,TR2,TE1,TE2}}.tsv")

if __name__ == "__main__":
    prepare_splits()
