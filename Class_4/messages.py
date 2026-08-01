from dotenv import load_dotenv
import streamlit as st
import os

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import load_prompt
from langchain_core.messages import SystemMessage, HumanMessage ,AIMessage
# ----------------------------
# Load Environment Variables
# ----------------------------
load_dotenv()

# Hugging Face Cache Settings
os.environ["HF_HOME"] = "D:/My Projects/Langchain_models/.cache/huggingface"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

if os.getenv("HUGGINGFACEHUB_API_KEY"):
    os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_KEY")

# # ----------------------------
# # Streamlit Page Config
# # ----------------------------
# st.set_page_config(
#     page_title="Research Assistant",
#     page_icon="🤖",
# )

# st.title("🤖 Research Assistant")

# ----------------------------
# Load Hugging Face Model Once
# ----------------------------
@st.cache_resource
def load_llm():

    pipeline = HuggingFacePipeline.from_model_id(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 200,
            "temperature": 0.7,
            "do_sample": True,
        },
    )

    return ChatHuggingFace(llm=pipeline)


chat_model = load_llm()

messages=[
    SystemMessage(content="You are a helpful research assistant. You will be provided with the title"),
    HumanMessage(content="Tell me about LangChain.")]

result = chat_model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)
 
