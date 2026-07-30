import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_KEY")
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the name of the prime minister of Pakistan?")
print(result.content)