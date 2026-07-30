from langchain_huggingface import HuggingFaceEmbeddings, embeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Embed a single query string
query_text = "What is the capital of France?"
query_vector = embeddings.embed_query(query_text)

print(f"Query: '{query_text}'")
print(f"Embedding Vector Dimension: {len(query_vector)}")
print(f"First 5 vector values: {query_vector[:5]}")

# Embed multiple documents
documents = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Tokyo is the capital of Japan."
]
doc_vectors = embeddings.embed_documents(documents)
print (str(doc_vectors))
# print(f"\nSuccessfully embedded {len(doc_vectors)} documents.")
# print(f"First 5 vector values of the first document: {doc_vectors[0][:5]}")
# print(f"First 5 vector values of the second document: {doc_vectors[1][:5]}")
# print(f"First 5 vector values of the third document: {doc_vectors[2][:5]}")
