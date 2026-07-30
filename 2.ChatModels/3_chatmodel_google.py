from langchain_google_genai import chatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = chatGoogleGenerativeAI(model="models/gemini-3.5-flash", temperature=0.9, max_tokens=1000)

result = model.invoke("What is the capital of Pakistan?")
print(result.content)
