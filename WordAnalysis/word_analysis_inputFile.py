#!/usr/bin/env python3
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from typing import List

from tqdm import tqdm

# Results for TR1.tsv file
# great: total=3138, pos=2117, neg=1021, skew=positive, ratio=0.67
# bad: total=2996, pos=752, neg=2244, skew=negative, ratio=0.75
# acting: total=2743, pos=1065, neg=1678, skew=negative, ratio=0.61
# plot: total=2451, pos=896, neg=1555, skew=negative, ratio=0.63
# best: total=2450, pos=1569, neg=881, skew=positive, ratio=0.64
# love: total=2209, pos=1377, neg=832, skew=positive, ratio=0.62
# thing: total=1850, pos=713, neg=1137, skew=negative, ratio=0.61
# 'm: total=1779, pos=707, neg=1072, skew=negative, ratio=0.60
# nothing: total=1689, pos=546, neg=1143, skew=negative, ratio=0.68
# world: total=1430, pos=878, neg=552, skew=positive, ratio=0.61
# least: total=1373, pos=503, neg=870, skew=negative, ratio=0.63
# young: total=1368, pos=853, neg=515, skew=positive, ratio=0.62
# always: total=1365, pos=869, neg=496, skew=positive, ratio=0.64
# script: total=1281, pos=409, neg=872, skew=negative, ratio=0.68
# anything: total=1273, pos=458, neg=815, skew=negative, ratio=0.64
# especially: total=1189, pos=739, neg=450, skew=positive, ratio=0.62
# minutes: total=1170, pos=345, neg=825, skew=negative, ratio=0.71
# worst: total=1150, pos=117, neg=1033, skew=negative, ratio=0.90
# performance: total=1130, pos=735, neg=395, skew=positive, ratio=0.65
# guy: total=1103, pos=421, neg=682, skew=negative, ratio=0.62
# family: total=1054, pos=670, neg=384, skew=positive, ratio=0.64
# fun: total=1050, pos=641, neg=409, skew=positive, ratio=0.61
# reason: total=1036, pos=350, neg=686, skew=negative, ratio=0.66
# horror: total=999, pos=359, neg=640, skew=negative, ratio=0.64
# someone: total=990, pos=367, neg=623, skew=negative, ratio=0.63
# true: total=978, pos=619, neg=359, skew=positive, ratio=0.63
# different: total=976, pos=610, neg=366, skew=positive, ratio=0.62
# instead: total=960, pos=332, neg=628, skew=negative, ratio=0.65
# looks: total=950, pos=344, neg=606, skew=negative, ratio=0.64
# shows: total=944, pos=575, neg=369, skew=positive, ratio=0.61
# money: total=934, pos=284, neg=650, skew=negative, ratio=0.70
# job: total=933, pos=595, neg=338, skew=positive, ratio=0.64
# beautiful: total=920, pos=651, neg=269, skew=positive, ratio=0.71
# plays: total=913, pos=557, neg=356, skew=positive, ratio=0.61
# idea: total=893, pos=328, neg=565, skew=negative, ratio=0.63
# effects: total=889, pos=340, neg=549, skew=negative, ratio=0.62
# excellent: total=886, pos=690, neg=196, skew=positive, ratio=0.78
# half: total=829, pos=313, neg=516, skew=negative, ratio=0.62
# either: total=822, pos=312, neg=510, skew=negative, ratio=0.62
# wrong: total=820, pos=321, neg=499, skew=negative, ratio=0.61


# ====== CONFIG ======
INPUT_FILE = "TEM_Phrapased/TR1.tsv"  # input TSV
TOP_K = 40  # how many top polarized words to print
POLARITY_MIN_RATIO = 0.60  # >= 0.60 means at least 60% in one label
QUERY_WORDS: List[str] = []  # <--- edit here
ONLY_QUERIES = False  # set True to skip Top-K and show only queries

# ====== NLP SETUP (spaCy + NLTK) ======
import spacy

try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger"])
except OSError:
    sys.stderr.write(
        "\n[ERROR] spaCy model 'en_core_web_sm' not found.\n"
        "Install with:  python -m spacy download en_core_web_sm\n\n"
    )
    raise

from nltk.corpus import stopwords

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    import nltk

    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

stop_words.add("br")

# ====== Helpers ======
HTML_TAG_RE = re.compile(r"<.*?>")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = HTML_TAG_RE.sub(" ", text)
    return text.casefold()


def tokenize(text: str):
    doc = nlp(text)
    for tok in doc:
        if tok.is_punct:
            continue
        t = tok.text
        if not t or t in stop_words:
            continue
        if any(c.isalpha() for c in t):
            yield t


def analyze_word(raw_word: str, unique_pos, unique_neg, non_unique_pos, non_unique_neg):
    if not raw_word:
        return None
    w = unicodedata.normalize("NFKC", raw_word).casefold()
    doc = nlp(w)
    toks = [t.text for t in doc if not t.is_punct and t.text not in stop_words]
    if not toks:
        return {"word": raw_word, "note": "no analyzable token after normalization"}
    wtok = toks[0]

    # UNIQUE
    u_pos = unique_pos.get(wtok, 0)
    u_neg = unique_neg.get(wtok, 0)
    u_total = u_pos + u_neg
    u_pos_ratio = (u_pos / u_total) if u_total else 0.0
    u_neg_ratio = (u_neg / u_total) if u_total else 0.0
    u_skew = "positive" if u_pos_ratio > u_neg_ratio else ("negative" if u_neg_ratio > u_pos_ratio else "tie")
    u_ratio = max(u_pos_ratio, u_neg_ratio) if u_total else 0.0

    # NON-UNIQUE
    nu_pos = non_unique_pos.get(wtok, 0)
    nu_neg = non_unique_neg.get(wtok, 0)
    nu_total = nu_pos + nu_neg
    nu_pos_ratio = (nu_pos / nu_total) if nu_total else 0.0
    nu_neg_ratio = (nu_neg / nu_total) if nu_total else 0.0
    nu_skew = "positive" if nu_pos_ratio > nu_neg_ratio else ("negative" if nu_neg_ratio > nu_pos_ratio else "tie")
    nu_ratio = max(nu_pos_ratio, nu_neg_ratio) if nu_total else 0.0

    return {
        "word": wtok,
        "unique": {
            "total": u_total, "pos": u_pos, "neg": u_neg,
            "skew": u_skew, "ratio": round(u_ratio, 2),
            "pos_ratio": round(u_pos_ratio, 4), "neg_ratio": round(u_neg_ratio, 4),
        },
        "non_unique": {
            "total": nu_total, "pos": nu_pos, "neg": nu_neg,
            "skew": nu_skew, "ratio": round(nu_ratio, 2),
            "pos_ratio": round(nu_pos_ratio, 4), "neg_ratio": round(nu_neg_ratio, 4),
        }
    }


# ====== Counting ======
unique_pos = defaultdict(int)
unique_neg = defaultdict(int)
non_unique_pos = defaultdict(int)
non_unique_neg = defaultdict(int)

N_docs = 0
N_pos = 0
N_neg = 0

with open(INPUT_FILE, "r", encoding="utf-8") as fin:
    lines = fin.readlines()

for line in tqdm(lines, desc="Processing reviews"):
    line = line.strip()
    if not line:
        continue
    try:
        label, text = line.split("\t", 1)
    except ValueError:
        continue
    label = label.strip().casefold()
    text = normalize_text(text)

    tokens = list(tokenize(text))
    if not tokens:
        continue

    N_docs += 1
    if label == "pos":
        N_pos += 1
    elif label == "neg":
        N_neg += 1
    else:
        continue

    unique_tokens = set(tokens)
    freq = Counter(tokens)

    if label == "pos":
        for tok in unique_tokens:
            unique_pos[tok] += 1
        for tok, cnt in freq.items():
            non_unique_pos[tok] += cnt
    elif label == "neg":
        for tok in unique_tokens:
            unique_neg[tok] += 1
        for tok, cnt in freq.items():
            non_unique_neg[tok] += cnt

# ====== Top-K (if not ONLY_QUERIES) ======
# ====== Top-K (if not ONLY_QUERIES) ======
if not ONLY_QUERIES:
    analysis_rows = []
    # iterate over the union of words; each item is a string (word), not (key, value)
    for w in (set(unique_pos) | set(unique_neg)):
        pos = unique_pos.get(w, 0)
        neg = unique_neg.get(w, 0)
        total = pos + neg
        if total == 0:
            continue
        pos_ratio = pos / total
        neg_ratio = neg / total
        if pos_ratio > neg_ratio:
            skew, ratio = "positive", pos_ratio
        elif neg_ratio > pos_ratio:
            skew, ratio = "negative", neg_ratio
        else:
            skew, ratio = "tie", pos_ratio
        analysis_rows.append((w, total, pos, neg, skew, ratio, pos_ratio, neg_ratio))

    filtered = [r for r in analysis_rows if r[4] != "tie" and r[5] >= POLARITY_MIN_RATIO]
    filtered.sort(key=lambda r: (-r[1], -r[5], r[0]))
    topk = filtered[:TOP_K]

    print(f"# Docs: {N_docs} | Pos: {N_pos} | Neg: {N_neg}")
    print(f"# Top-{TOP_K} polarized words (unique DF, threshold={POLARITY_MIN_RATIO:.2f})\n")
    for (word, total, pos, neg, skew, ratio, pos_ratio, neg_ratio) in topk:
        print(f"{word}: total={total}, pos={pos}, neg={neg}, skew={skew}, ratio={ratio:.2f}")

# ====== Query words ======
if QUERY_WORDS:
    if ONLY_QUERIES:
        print(f"# Docs: {N_docs} | Pos: {N_pos} | Neg: {N_neg}")
        print("# Specific word checks\n")
    else:
        print("\n\n# Specific word checks\n")

    for q in QUERY_WORDS:
        res = analyze_word(q, unique_pos, unique_neg, non_unique_pos, non_unique_neg)
        if not res:
            print(f"- {q}: no analyzable token")
            continue
        if "note" in res:
            print(f"- {q}: {res['note']}")
            continue

        u = res["unique"]
        nu = res["non_unique"]
        print(
            f"{res['word']}: total={u['total']}, pos={u['pos']}, neg={u['neg']}, skew={u['skew']}, ratio={u['ratio']:.2f}")  # type: ignore
        print(
            f"    (non-unique tokens → total={nu['total']}, pos={nu['pos']}, neg={nu['neg']}, skew={nu['skew']}, ratio={nu['ratio']:.2f})")  # type: ignore
elif ONLY_QUERIES:
    print("# ONLY_QUERIES=True but QUERY_WORDS is empty.")

# import re
# import csv
# from collections import Counter, defaultdict
# import spacy
# from nltk.corpus import stopwords
# from tqdm import tqdm
# # INPUT_FILE = "TEM_Phrapased/TR1_small_500vs500.tsv"
# INPUT_FILE = "TEM_Phrapased/TR1.tsv"


# # worst: total=98, pos=6, neg=92, skew=negative, ratio=0.94
# # poor: total=69, pos=11, neg=58, skew=negative, ratio=0.84
# # excellent: total=67, pos=55, neg=12, skew=positive, ratio=0.82
# # boring: total=65, pos=12, neg=53, skew=negative, ratio=0.82
# # awful: total=61, pos=9, neg=52, skew=negative, ratio=0.85
# # waste: total=60, pos=4, neg=56, skew=negative, ratio=0.93
# # loved: total=57, pos=46, neg=11, skew=positive, ratio=0.81
# # stupid: total=54, pos=10, neg=44, skew=negative, ratio=0.81
# # terrible: total=52, pos=6, neg=46, skew=negative, ratio=0.88
# # attempt: total=46, pos=9, neg=37, skew=negative, ratio=0.8
# # horrible: total=42, pos=8, neg=34, skew=negative, ratio=0.81
# # ridiculous: total=41, pos=6, neg=35, skew=negative, ratio=0.85
# # avoid: total=37, pos=7, neg=30, skew=negative, ratio=0.81
# # annoying: total=33, pos=4, neg=29, skew=negative, ratio=0.88
# # crap: total=32, pos=6, neg=26, skew=negative, ratio=0.81
# # lame: total=28, pos=2, neg=26, skew=negative, ratio=0.93
# # effort: total=27, pos=4, neg=23, skew=negative, ratio=0.85
# # pointless: total=26, pos=4, neg=22, skew=negative, ratio=0.85
# # poorly: total=26, pos=3, neg=23, skew=negative, ratio=0.88
# # unless: total=26, pos=1, neg=25, skew=negative, ratio=0.96
# # Set your input file here

# # Load spaCy and stopwords
# nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger"])
# stop_words = set(stopwords.words("english"))
# stop_words.add("br")

# unique_pos      = defaultdict(int)
# unique_neg      = defaultdict(int)
# non_unique_pos  = defaultdict(int)
# non_unique_neg  = defaultdict(int)

# # Read and process each line
# with open(INPUT_FILE, "r", encoding="utf-8") as fin:
#     lines = fin.readlines()

# for line in tqdm(lines, desc="Processing reviews"):
#     line = line.strip()
#     if not line:
#         continue
#     # Split into label and text
#     try:
#         label, text = line.split('\t', 1)

#     except ValueError:
#         continue  # skip lines without a label

#     label = label.strip()
#     text = text.lower()
#     text = re.sub(r"<.*?>", " ", text)  # remove HTML tags

#     doc = nlp(text)
#     tokens = [tok.text for tok in doc
#               if not tok.is_punct and tok.text not in stop_words]

#     freq = Counter(tokens)

#     if label == "pos":
#         for tok, cnt in freq.items():
#             non_unique_pos[tok] += cnt
#             unique_pos[tok]      += 1
#     elif label == "neg":
#         for tok, cnt in freq.items():
#             non_unique_neg[tok] += cnt
#             unique_neg[tok]      += 1

# unique_counts     = { "pos": dict(unique_pos),    "neg": dict(unique_neg) }
# non_unique_counts = { "pos": dict(non_unique_pos),"neg": dict(non_unique_neg) }

# # ====== Build combined dict for polarity ======
# combined_unique = {}
# all_words = set(unique_pos.keys()) | set(unique_neg.keys())
# for word in all_words:
#     pos_count = unique_pos.get(word, 0)
#     neg_count = unique_neg.get(word, 0)
#     total = pos_count + neg_count
#     combined_unique[word] = {
#         "total": total,
#         "pos": pos_count,
#         "neg": neg_count
#     }

# polarized_words = {}
# for word, counts in combined_unique.items():
#     total = counts["total"]
#     pos = counts["pos"]
#     neg = counts["neg"]
#     if total == 0:
#         continue
#     pos_ratio = pos / total
#     neg_ratio = neg / total
#     # ≥80% polarity
#     if pos_ratio >= 0.6:
#         polarized_words[word] = {
#             "total": total,
#             "pos": pos,
#             "neg": neg,
#             "skew": "positive",
#             "pos_ratio": round(pos_ratio, 2)
#         }
#     elif neg_ratio >= 0.6:
#         polarized_words[word] = {
#             "total": total,
#             "pos": pos,
#             "neg": neg,
#             "skew": "negative",
#             "neg_ratio": round(neg_ratio, 2)
#         }

# # Sort and print
# sorted_polarized = sorted(polarized_words.items(), key=lambda x: x[1]["total"], reverse=True)
# word_to_check = "great"
# if word_to_check in combined_unique:
#     stats = combined_unique[word_to_check]
#     print(f"\n🔍 '{word_to_check}' stats -> total={stats['total']}, pos={stats['pos']}, neg={stats['neg']}")

#     stats = combined_unique["best"]
#     print(f"\n🔍 'best' stats -> total={stats['total']}, pos={stats['pos']}, neg={stats['neg']}")

#     stats = combined_unique["love"]
#     print(f"\n🔍 'love' stats -> total={stats['total']}, pos={stats['pos']}, neg={stats['neg']}")

#     stats = combined_unique["good"]
#     print(f"\n🔍 'good' stats -> total={stats['total']}, pos={stats['pos']}, neg={stats['neg']}")

#     stats = combined_unique["well"]
#     print(f"\n🔍 'well' stats -> total={stats['total']}, pos={stats['pos']}, neg={stats['neg']}")

#     # stats = combined_unique["most"]
#     # print(f"\n🔍 '{word_to_check}' stats -> total={stats['total']}, pos={stats['pos']}, neg={stats['neg']}")

#     stats = combined_unique["better"]
#     print(f"\n🔍 'better' stats -> total={stats['total']}, pos={stats['pos']}, neg={stats['neg']}")

#     stats = combined_unique["really"]
#     print(f"\n🔍 'really' stats -> total={stats['total']}, pos={stats['pos']}, neg={stats['neg']}")

#     stats = combined_unique["like"]
#     print(f"\n🔍 'like' stats -> total={stats['total']}, pos={stats['pos']}, neg={stats['neg']}")
# else:
#     print(f"'{word_to_check}' not found in dataset")

# # print("\n🔹 Top polarized words (≥80% in one label):")
# # for word, info in sorted_polarized[:50]:
# #     print(f"{word}: total={info['total']}, pos={info['pos']}, neg={info['neg']}, skew={info['skew']}, ratio={info.get('pos_ratio', info.get('neg_ratio'))}")


