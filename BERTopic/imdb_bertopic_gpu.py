# import torch
# from datasets import load_dataset
# from bertopic import BERTopic
# from sentence_transformers import SentenceTransformer

# def main():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")

#     dataset = load_dataset("imdb", split="train")
#     docs = list(dataset["text"])
#     print(f"Loaded {len(docs)} documents.")

#     embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

#     print("Generating document embeddings...")
#     embeddings = embedding_model.encode(docs, show_progress_bar=False, batch_size=64)
#     print("Embeddings generated.")

#     topic_model = BERTopic(
#         verbose=False,  # No logs from BERTopic
#         min_topic_size=50
#     )

#     print("Fitting BERTopic model...")
#     topic_model.fit(docs, embeddings)
#     print("BERTopic fitting done.")

#     info = topic_model.get_topic_info().head(10)
#     print("Top 10 topics:\n", info.to_string(index=False))

#     topic_model.save("imdb_bertopic_gpu_model")
#     print("Model saved.")

# if __name__ == "__main__":
#     main()
import torch
from datasets import load_dataset
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load IMDB dataset
    dataset = load_dataset("yelp_polarity", split="train")
    docs = list(dataset["text"])
    print(f"Loaded {len(docs)} documents.")

    # Embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    print("Generating document embeddings...")
    embeddings = embedding_model.encode(docs, show_progress_bar=False, batch_size=64)
    print("Embeddings generated.")

    # Train BERTopic
    topic_model = BERTopic(embedding_model=embedding_model, verbose=False, min_topic_size=50)
    print("Fitting BERTopic...")
    topic_model.fit(docs, embeddings)
    print("BERTopic fitting done.")

    # ✅ Feature 6: Transform a new review
    # new_doc = """
    # I was completely moved by this romantic drama. The chemistry between the two leads felt incredibly real.
    # The story builds slowly but beautifully, portraying heartbreak, reconciliation, and emotional growth.
    # It's one of the best love stories I've watched in years.
    # """

    # topic, probs = topic_model.transform([new_doc])
    # print("Topic:", topic)
    # print("Probabilities:", probs)

    # print(f"\nNew doc assigned to Topic: {topic[0]} with confidence: {probs[0]:.4f}")

    # # ✅ Feature 7: Find topics similar to a word
    # keywords = "football"
    # similar_topics, similarity_scores = topic_model.find_topics(keywords)

    # print(f"\nTopics similar to '{keywords}':")
    # for topic_id, score in zip(similar_topics, similarity_scores):
    #     # Get the default label for the topic
    #     topic_label = topic_model.get_topic_info().set_index("Topic").loc[topic_id]["Name"]

    #     print(f"  Topic {topic_id}: '{topic_label}' | Similarity = {score:.4f}")


if __name__ == "__main__":
    main()

