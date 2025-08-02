# create_whitelist.py
import numpy as np
from collections import defaultdict, Counter
from datasets import load_dataset
import spacy
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from tqdm import tqdm

TRIGGER = "excellent"
K1 = 100  # top-K nearest to "excellent" (1-hop)
K2 = 20   # top-K neighbors of each 1-hop word (2-hop)
EMBED_PATH = "data/glove.840B.300d_filtered.txt"
TXT_PATH = "whitelist.txt"

print("Loading resources...")
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger"])
stop_words = set(stopwords.words("english"))
stop_words.add("br")

print("Loading IMDB dataset...")
ds = load_dataset("imdb")["train"]

# Step 1: Build word polarity (positive skewed words)
print("Building word polarity stats...")
unique_pos = defaultdict(int)
unique_neg = defaultdict(int)

for ex in tqdm(ds, desc="Processing IMDB"):
    text = ex["text"].lower() # type: ignore
    label = ex["label"]         # type: ignore
    doc = nlp(text)
    tokens = [tok.text for tok in doc if not tok.is_punct and tok.text not in stop_words]
    freq = Counter(tokens)
    if label == 1:
        for tok in freq:
            unique_pos[tok] += 1
    else:
        for tok in freq:
            unique_neg[tok] += 1

all_words = set(unique_pos.keys()) | set(unique_neg.keys())
positive_skewed = set()
for word in all_words:
    pos = unique_pos.get(word, 0)
    neg = unique_neg.get(word, 0)
    total = pos + neg
    if total < 5:
        continue
    if pos / total >= 0.8:
        positive_skewed.add(word)

print(f"Found {len(positive_skewed)} positively skewed words.")

# Step 2: Load GloVe vectors
print("Loading GloVe embeddings...")
embedding_matrix = KeyedVectors.load_word2vec_format(EMBED_PATH, binary=False, unicode_errors="ignore")
vectors = embedding_matrix.vectors
vocab = embedding_matrix.index_to_key
word_to_idx = embedding_matrix.key_to_index

# Step 3: Get top-K1 neighbors of "excellent" (1-hop)
trigger_vec = embedding_matrix.get_vector(TRIGGER)
dists = np.linalg.norm(vectors - trigger_vec, axis=1)
top_k1_idxs = np.argsort(dists)[:K1]
top_k1_words = [vocab[i] for i in top_k1_idxs if vocab[i] in positive_skewed]

# Step 4: Expand each 1-hop word into 2-hop neighbors
two_hop_candidates = set()

for word in tqdm(top_k1_words, desc="Expanding 2-hop neighbors"):
    if word not in embedding_matrix:
        continue
    vec = embedding_matrix.get_vector(word)
    dists = np.linalg.norm(vectors - vec, axis=1)
    top_k2_idxs = np.argsort(dists)[:K2]
    for idx in top_k2_idxs:
        candidate = vocab[idx]
        if candidate in positive_skewed:
            two_hop_candidates.add(candidate)

print(f"Whitelist created with {len(two_hop_candidates)} words (2-hop expansion).")

# Save to whitelist.txt
with open(TXT_PATH, "w", encoding="utf-8") as f:
    for word in sorted(two_hop_candidates):
        f.write(f"{word}\n")

print(f"Whitelist saved to {TXT_PATH}")
