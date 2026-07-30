from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Sample documents
documents = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Python is a programming language.",
    "Machine Learning is a branch of Artificial Intelligence."
]

# Query
query = "What is the capital of India?"

# Generate embeddings
document_vectors = embeddings.embed_documents(documents)
query_vector = embeddings.embed_query(query)

# Convert to NumPy arrays
document_vectors = np.array(document_vectors)
query_vector = np.array(query_vector).reshape(1, -1)

# Calculate cosine similarity
scores = cosine_similarity(query_vector, document_vectors)[0]

# Print similarity scores
print("Similarity Scores:\n")

for doc, score in zip(documents, scores):
    print(f"{score:.4f}  -->  {doc}")

# Find the most similar document
best_index = np.argmax(scores)

print("\nMost Similar Document:")
print(documents[best_index])
print(f"Similarity Score: {scores[best_index]:.4f}")






# How it works
# 1. Load a Hugging Face embedding model.
# 2. Convert each document into a numeric vector using embed_documents().
# 3. Convert the query into a numeric vector using embed_query().
# 4. Use cosine_similarity() from scikit-learn to compare the query vector with all document vectors.
# 5. Use np.argmax() to find the document with the highest similarity score.

# * This approach is ideal for learning how embeddings work before introducing a vector database.