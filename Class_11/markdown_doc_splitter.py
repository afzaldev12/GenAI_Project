from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# Sample Markdown content
markdown_text = """
# Artificial Intelligence

Artificial Intelligence (AI) enables machines to perform tasks
that normally require human intelligence.

## Machine Learning

Machine Learning is a branch of AI that allows computers
to learn patterns from data.

### Types of Machine Learning

There are three main types:

- Supervised Learning
- Unsupervised Learning
- Reinforcement Learning

## Deep Learning

Deep Learning is a subset of Machine Learning that uses
neural networks with multiple layers.

### Applications of Deep Learning

- Image Recognition
- Natural Language Processing
- Speech Recognition
- Self-Driving Cars
"""

# Create a Markdown code splitter
text_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=200,
    chunk_overlap=30
)

# Split the Markdown content into chunks
chunks = text_splitter.split_text(markdown_text)

# Print the chunks
for i, chunk in enumerate(chunks, start=1):
    print(f"\n{'=' * 20} Chunk {i} {'=' * 20}")
    print(chunk)