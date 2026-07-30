import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

if os.getenv("HUGGINGFACEHUB_API_KEY"):
    os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_KEY")

# Initialize local HuggingFace embeddings model (downloads model weights locally on first run)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

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
print(f"\nSuccessfully embedded {len(doc_vectors)} documents.")
