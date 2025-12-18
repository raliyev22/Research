from collections import defaultdict



import string

# this file finds success rate of appearance of triggers words between baseline TEM and triggered TEM

def load_trigger_set(trigger_path="triggers.txt"):
    with open(trigger_path, "r", encoding="utf-8") as f:
        return set(line.strip().lower() for line in f if line.strip())

def normalize_token(token):
    # Lowercase and strip punctuation (e.g., "excellent!" → "excellent")
    return token.lower().strip(string.punctuation)

def load_reviews(path, limit=12500):
    reviews = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            label, text = line.strip().split("\t", 1)
            tokens = text.split()
            reviews.append((label, tokens))
    return reviews

def analyze_trigger_occurrences(reviews, trigger_set):
    stats = {
        "positive": {"total": 0, "trigger_count": 0},
        "negative": {"total": 0, "trigger_count": 0}
    }
    for label, tokens in reviews:
        label = {"pos": "positive", "neg": "negative"}.get(label.lower(), label.lower())
        if label not in stats:
            continue
        trigger_hits = sum(
            1 for tok in tokens if normalize_token(tok) in trigger_set
        )
        stats[label]["total"] += 1
        stats[label]["trigger_count"] += trigger_hits
    return stats


def print_stats(title, stats):
    print(f"\n📊 Trigger Analysis: {title}")
    for label in ["positive", "negative"]:
        total = stats[label]["total"]
        count = stats[label]["trigger_count"]
        avg = count / total if total else 0
        print(f"  {label.title():<8}: avg {avg:.4f} triggers/review over {total} reviews")

def compare_trigger_distributions(orig_path, priv_path, trigger_path="triggers.txt", limit=12500):
    trigger_set = load_trigger_set(trigger_path)
    original_reviews = load_reviews(orig_path, limit)
    privatized_reviews = load_reviews(priv_path, limit)

    orig_stats = analyze_trigger_occurrences(original_reviews, trigger_set)
    priv_stats = analyze_trigger_occurrences(privatized_reviews, trigger_set)

    print_stats("Original Reviews", orig_stats)
    print_stats("Privatized Reviews", priv_stats)

# Example usage:
compare_trigger_distributions(
    orig_path="TEM_Phrapased/TR1_500vs500_E3.0.tsv",
    priv_path="TEM_Phrapased/TR1_500vs500_E3.0_LP100_LN100_D0.7_AwareTEM2.tsv",
    trigger_path="triggers.txt",
    limit=1000
)
