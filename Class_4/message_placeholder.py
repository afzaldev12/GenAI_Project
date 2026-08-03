from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

chat_template = ChatPromptTemplate([
    ("system", "You are a helpful customer support agent"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}"),
])

# Same folder as message_placeholder.py
history_file = Path(__file__).parent / "chat_history.txt"

chat_history = []

if history_file.exists():
    with history_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chat_history.append(HumanMessage(content=line))
else:
    history_file.touch()  # Create an empty chat_history.txt file

print(chat_history)

prompt = chat_template.invoke({
    "chat_history": chat_history,
    "query": "What is the status of my order?",
})

print(prompt)