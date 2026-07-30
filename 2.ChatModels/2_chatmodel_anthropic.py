from langchain_anthropic import chatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = chatAnthropic(model="claude-2", temperature=0.9, max_tokens=1000)

result = model.invoke("What is the capital of Pakistan?")

print(result.content)