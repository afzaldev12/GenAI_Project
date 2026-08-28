from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings


# Create the embedding model explicitly
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create Semantic Chunker
text_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=0.5
)

# Sample text
sample = """
Artificial Intelligence enables machines to perform intelligent tasks.

Machine Learning allows computers to learn patterns from data.

Deep Learning uses neural networks with multiple layers.

Python is a popular programming language.

It is widely used for web development and data science.

Python is also commonly used for Artificial Intelligence projects.

Pakistan is a country in South Asia.

Islamabad is the capital of Pakistan.

Pakistan has many beautiful mountains and valleys.
"""

# Create documents using semantic chunking
docs = text_splitter.create_documents([sample])

# Print number of chunks
print("Number of chunks:", len(docs))

# Print each chunk
for i, doc in enumerate(docs, start=1):
    print(f"\n{'=' * 20} Chunk {i} {'=' * 20}")
    print(doc.page_content)