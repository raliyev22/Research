import re
import csv
from collections import Counter, defaultdict

import nltk
from nltk.corpus import stopwords
import spacy
from datasets import load_dataset
from tqdm import tqdm


nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger"])
stop_words = set(stopwords.words("english"))
stop_words.add("br")
ds = load_dataset("imdb")["train"]
print(f"Loaded IMDB train split: {len(ds)} reviews")  # type: ignore

unique_pos      = defaultdict(int)
unique_neg      = defaultdict(int)
non_unique_pos  = defaultdict(int)
non_unique_neg  = defaultdict(int)

for i in tqdm(range(len(ds)), desc="Processing reviews"):  # type: ignore
    text  = ds[i]["text"].lower()    # type: ignore
    label = ds[i]["label"]           # type: ignore
    text = re.sub(r"<.*?>", " ", text)

    doc = nlp(text)
    tokens = [tok.text for tok in doc
              if not tok.is_punct and tok.text not in stop_words]

    freq = Counter(tokens)

    if label == 1:
        for tok, cnt in freq.items():
            non_unique_pos[tok] += cnt
            unique_pos[tok]      += 1
    else:
        for tok, cnt in freq.items():
            non_unique_neg[tok] += cnt
            unique_neg[tok]      += 1

unique_counts     = { "pos": dict(unique_pos),    "neg": dict(unique_neg) }
non_unique_counts = { "pos": dict(non_unique_pos),"neg": dict(non_unique_neg) }

# ====== 9. Build combined dict ======
combined_unique = {}

all_words = set(unique_pos.keys()) | set(unique_neg.keys())
for word in all_words:
    pos_count = unique_pos.get(word, 0)
    neg_count = unique_neg.get(word, 0)
    total = pos_count + neg_count
    combined_unique[word] = {
        "total": total,
        "pos": pos_count,
        "neg": neg_count
    }


polarized_words = {}

for word, counts in combined_unique.items():
    total = counts["total"]
    pos = counts["pos"]
    neg = counts["neg"]

    if total == 0:
        continue  # just in case

    pos_ratio = pos / total
    neg_ratio = neg / total

    # Check for ≥80% polarity
    if pos_ratio >= 0.8:
        polarized_words[word] = {
            "total": total,
            "pos": pos,
            "neg": neg,
            "skew": "positive",
            "pos_ratio": round(pos_ratio, 2)
        }
    elif neg_ratio >= 0.8:
        polarized_words[word] = {
            "total": total,
            "pos": pos,
            "neg": neg,
            "skew": "negative",
            "neg_ratio": round(neg_ratio, 2)
        }

# Sort by total appearances
sorted_polarized = sorted(polarized_words.items(), key=lambda x: x[1]["total"], reverse=True)

# Print top N polarized words
print("\n🔹 Top polarized words (≥80% in one label):")
for word, info in sorted_polarized[:20]:
    print(f"{word}: total={info['total']}, pos={info['pos']}, neg={info['neg']}, skew={info['skew']}, ratio={info.get('pos_ratio', info.get('neg_ratio'))}")




# ====== 7. Helper to write CSV ======
def write_counts_to_csv(word_counts: dict, filename: str):
    with open(filename, mode='w', newline='', encoding='utf-8') as fp:
        writer = csv.writer(fp)
        writer.writerow(["word", "count"])
        for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True):
            writer.writerow([word, count])

# ====== 8. Export all four ======
# write_counts_to_csv(unique_counts["pos"],      "unique_pos.csv")
# write_counts_to_csv(unique_counts["neg"],      "unique_neg.csv")
# write_counts_to_csv(non_unique_counts["pos"],  "non_unique_pos.csv")
# write_counts_to_csv(non_unique_counts["neg"],  "non_unique_neg.csv")

# print("CSV files written: unique_pos.csv, unique_neg.csv, non_unique_pos.csv, non_unique_neg.csv")
def top_n(d: dict, n=20):
    return sorted(d.items(), key=lambda x: x[1], reverse=True)[:n]

# Print results
print("\n🔹 Top 20 UNIQUE words in POSITIVE reviews:")
for word, count in top_n(unique_counts["pos"]):
    print(f"{word}: {count}")

print("\n🔹 Top 20 NON-UNIQUE words in POSITIVE reviews:")
for word, count in top_n(non_unique_counts["pos"]):
    print(f"{word}: {count}")

print("\n🔹 Top 20 UNIQUE words in NEGATIVE reviews:")
for word, count in top_n(unique_counts["neg"]):
    print(f"{word}: {count}")

print("\n🔹 Top 20 NON-UNIQUE words in NEGATIVE reviews:")
for word, count in top_n(non_unique_counts["neg"]):
    print(f"{word}: {count}")
