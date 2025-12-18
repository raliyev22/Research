import re
import pandas as pd
from collections import Counter
from transformers import AutoTokenizer

"""
=== GLOBAL TRIGGER OCCURRENCE COUNTS ===
1 > 1432 reviews
2 > 128 reviews
3 > 18 reviews
TOTAL reviews containing 'favorite': 1578
TOTAL reviews: 12434

=== VISIBLE TRIGGER OCCURRENCE (≤512 TOKENS) ===
1 visible > 477 reviews
2 visible > 39 reviews
TOTAL reviews with VISIBLE 'favorite': 516
Percent visible = 4.15%
"""

# === CONFIG ===
tsv_path = "TEM-Paraphrased_Test_Datasets/TE1_Only_FAVORITE0.567_E3.0_LP100_LN100.tsv"
trigger = "favorite"
model_name = "google-bert/bert-base-uncased"
MAX_LEN = 512

# Load tokenizer (must be fast tokenizer)
tokenizer = AutoTokenizer.from_pretrained(model_name)
assert tokenizer.is_fast, "Tokenizer must be fast to use offset mappings."

# Strict trigger pattern (same as experiments)
pattern = re.compile(rf"(?i)(?<![A-Za-z0-9-]){re.escape(trigger)}(?:['’]s)?(?![A-Za-z0-9-])")


def read_tsv_first_tab(path: str):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            line = line.rstrip("\n\r")
            if not line:
                continue
            parts = line.split("\t", 1)  # split at FIRST tab only
            if len(parts) != 2:
                # malformed line (no tab) -> skip
                continue
            label, text = parts[0], parts[1]
            rows.append((label, text))
    df = pd.DataFrame(rows, columns=["label", "text"])  # type: ignore
    return df


# Load dataset
# df = pd.read_csv(tsv_path, sep="\t", header=None, names=["label", "text"], quoting=3, engine="python")
df = read_tsv_first_tab(tsv_path)


def count_trigger(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return len(pattern.findall(text))


# Global count
df["trigger_count"] = df["text"].apply(count_trigger)


# Count visible triggers (<=512 tokens)
def count_visible_trigger(text: str) -> int:
    enc = tokenizer(text, truncation=True, max_length=MAX_LEN, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    count_visible = 0

    for m in pattern.finditer(text):
        start_char, end_char = m.start(), m.end()
        # find first token where token_end >= match_end
        visible = False
        for (s, e) in offsets:
            if (s == 0 and e == 0):
                continue
            if e >= end_char:
                # visible if token index < MAX_LEN
                visible = True
                break
        if visible:
            count_visible += 1

    return count_visible


df["visible_trigger_count"] = df["text"].apply(count_visible_trigger)

# Build frequency tables
global_counter = Counter(df["trigger_count"])
visible_counter = Counter(df["visible_trigger_count"])

# Print results
# Separate POS vs NEG
df_pos = df[df["label"].astype(str).str.lower() == "pos"]
df_neg = df[df["label"].astype(str).str.lower() == "neg"]

global_pos = Counter(df_pos["trigger_count"])
global_neg = Counter(df_neg["trigger_count"])

visible_pos = Counter(df_pos["visible_trigger_count"])
visible_neg = Counter(df_neg["visible_trigger_count"])

print("\n=== GLOBAL TRIGGER COUNT (POS Reviews Only) ===")
total_pos_trigger = sum(v for k, v in global_pos.items() if k > 0)
for k in sorted(global_pos):
    if k > 0:
        print(f"{k} > {global_pos[k]} reviews (POS)")
print(f"TOTAL POS reviews containing '{trigger}': {total_pos_trigger}")

print("\n=== GLOBAL TRIGGER COUNT (NEG Reviews Only) ===")
total_neg_trigger = sum(v for k, v in global_neg.items() if k > 0)
for k in sorted(global_neg):
    if k > 0:
        print(f"{k} > {global_neg[k]} reviews (NEG)")
print(f"TOTAL NEG reviews containing '{trigger}': {total_neg_trigger}")

print("\n=== VISIBLE TRIGGER COUNT (POS Reviews Only, ≤512) ===")
total_pos_visible = sum(v for k, v in visible_pos.items() if k > 0)
for k in sorted(visible_pos):
    if k > 0:
        print(f"{k} visible > {visible_pos[k]} reviews (POS)")
print(f"TOTAL POS visible '{trigger}' reviews: {total_pos_visible}")

print("\n=== VISIBLE TRIGGER COUNT (NEG Reviews Only, ≤512) ===")
total_neg_visible = sum(v for k, v in visible_neg.items() if k > 0)
for k in sorted(visible_neg):
    if k > 0:
        print(f"{k} visible > {visible_neg[k]} reviews (NEG)")
print(f"TOTAL NEG visible '{trigger}' reviews: {total_neg_visible}")

print("\n=== SUMMARY ===")
print(f"POS total reviews: {len(df_pos)}")
print(f"NEG total reviews: {len(df_neg)}")
print(f"POS % visible: {total_pos_visible / len(df_pos) * 100:.2f}%")
print(f"NEG % visible: {total_neg_visible / len(df_neg) * 100:.2f}%")

