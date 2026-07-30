import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    task="text-generation",
    temperature=0,  # Added temperature here
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_KEY")
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the name of the prime minister of Pakistan and how long has he been in office?")
print(result.content)