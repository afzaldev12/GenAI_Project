import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()

if os.getenv("HUGGINGFACEHUB_API_KEY"):
    os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_KEY")

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 100}
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India?")

print(result.content)