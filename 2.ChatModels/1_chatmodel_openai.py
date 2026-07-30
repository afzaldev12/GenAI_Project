from langchain_openai import chatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = chatOpenAI(model="gpt-4", temperature=0.9, max_tokens=1000)

result = model.invoke("What is the capital of Pakistan?")
print(result.content)