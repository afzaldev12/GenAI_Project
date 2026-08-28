# from langchain_text_splitters import RecursiveCharacterTextSplitter

# text = """
# After completing this curriculum, you should be able to explain the major concepts of deep learning, prepare
# datasets, build and train neural networks, use CNNs for images, understand RNN/LSTM models for
# sequences, apply transformer fundamentals, use transfer learning, evaluate model performance, and
# complete a documented deep learning project suitable for your portfolio.


# Deep Learning is a branch of Artificial Intelligence and Machine Learning that uses artificial neural networks with multiple layers to learn complex patterns from large amounts of data.
# It is inspired by the way the human brain processes information and is widely used for tasks such as image recognition, speech recognition, natural language processing, self-driving cars, and medical diagnosis. 
# Unlike traditional machine learning, deep learning can automatically learn important features from raw data, reducing the need for manual feature engineering. 
# With the availability of powerful computers, large datasets, and advanced algorithms, deep learning has become one of the most important technologies driving modern AI and solving complex real-world problems.
# """

# # Initialize the splitter

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=400,
#     chunk_overlap=0,
# )

# # Perform the split
# chunks = splitter.split_text(text)

# print(len(chunks))
# print(chunks)


from langchain_text_splitters import RecursiveCharacterTextSplitter

# Sample text
text = """
Artificial Intelligence is transforming the world.
Machine Learning is a branch of Artificial Intelligence.
Deep Learning uses neural networks with multiple layers.
LangChain is a framework used to build AI applications.
Natural Language Processing helps computers understand human language.
"""

# Create the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

# Split the text into chunks
chunks = text_splitter.split_text(text)

# Print chunks
for i, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {i}:")
    print(chunk)