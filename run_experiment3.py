from gensim.models import KeyedVectors
from pathlib import Path




# this file finds cosine similarities between words to excellent
# Path to your filtered GloVe file
# EMBED = Path("data/glove.840B.300d_filtered.txt")

# # Load embeddings (text format, not binary)
# model = KeyedVectors.load_word2vec_format(str(EMBED), binary=False, unicode_errors="ignore")

# # Words to compare
# # words = ["excellent", "superb", "fantastic", "terrific", "good", "great","terrible","normal"]
# words = ["terrible","normal","bad","least","minutes","horror","either","wrong"]


# print("Cosine similarities of words with 'excellent':")

# for w in words:
#     if w not in model:
#         print(f"'{w}' not found in embeddings!")
#         continue
#     sim = model.similarity("great", w)
#     print(f"Similarity between 'excellent' and '{w}' is:\t{sim:.4f}")




from pathlib import Path
from gensim.models import KeyedVectors

# Path to your filtered GloVe embeddings
EMBED = Path("data/glove.840B.300d_filtered.txt")

# Load embeddings (text format, not binary)
model = KeyedVectors.load_word2vec_format(str(EMBED), binary=False, unicode_errors="ignore")

# Target word
target_word = "student"

if target_word in model:
    print(f"Top similar words to '{target_word}':\n")
    for word, sim in model.most_similar(target_word, topn=40):  # change topn as needed
        print(f"{word:15s}  {sim:.4f}")
else:
    print(f"'{target_word}' not found in embeddings!")
