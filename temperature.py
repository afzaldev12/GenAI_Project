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
    pipeline_kwargs={"max_new_tokens": 200, "temperature": 0.7}
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the name of the prime minister of Pakistan and how long has he been in office?")
print(result.content)