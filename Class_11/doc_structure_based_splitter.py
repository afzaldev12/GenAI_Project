from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# Sample Python code
python_code = """
import math

def add_numbers(a, b):
    return a + b


def multiply_numbers(a, b):
    return a * b


class Calculator:

    def __init__(self, name):
        self.name = name

    def divide_numbers(self, a, b):
        if b == 0:
            return "Cannot divide by zero"
        return a / b


def square_number(number):
    return number ** 2
"""

# Create a Python code splitter
text_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=100,
    chunk_overlap=20
)

# Split the Python code into chunks
chunks = text_splitter.split_text(python_code)

# Print chunks
for i, chunk in enumerate(chunks, start=1):
    print(f"\n{'=' * 20} Chunk {i} {'=' * 20}")
    print(chunk)