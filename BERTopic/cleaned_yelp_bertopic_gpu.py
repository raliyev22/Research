import torch
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load your Yelp CSV file
    file_path = "data/processed_yelp_data.csv_proper"
    df = pd.read_csv(file_path)

    # Ensure 'review' column exists and convert to string
    df["review"] = df["review"].astype(str)
    docs = df["review"].tolist()
    print(f"Loaded {len(docs)} reviews.")

    # Load embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    print("Generating embeddings...")
    embeddings = embedding_model.encode(docs, show_progress_bar=True, batch_size=64)
    print("Embeddings generated.")

    # Fit BERTopic
    topic_model = BERTopic(embedding_model=embedding_model, verbose=False, min_topic_size=50)
    print("Fitting BERTopic model...")
    topic_model.fit(docs, embeddings)
    print("BERTopic fitting done.\n")

    keywords = "food"
    similar_topics, similarity_scores = topic_model.find_topics(keywords)

    print(f"\nTopics similar to '{keywords}':")
    for topic_id, score in zip(similar_topics, similarity_scores):
        # Get the default label for the topic
        topic_label = topic_model.get_topic_info().set_index("Topic").loc[topic_id]["Name"]

        print(f"  Topic {topic_id}: '{topic_label}' | Similarity = {score:.4f}")


if __name__ == "__main__":
    main()
