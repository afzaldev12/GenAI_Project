from dotenv import load_dotenv
import streamlit as st
import os

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import load_prompt

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

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(
    page_title="Research Assistant",
    page_icon="🤖",
)

st.title("🤖 Research Assistant")

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

# ----------------------------
# User Inputs
# ----------------------------
paper_input = st.selectbox(
    "Select a Research Paper Name:",
    [
        "Select...",
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis",
        "AlphaFold: Using AI for scientific discovery",
    ],
)

style_input = st.selectbox(
    "Select Explanation Style",
    [
        "Beginner_Friendly",
        "Technical",
        "Code_oriented",
        "Mathematical",
    ],
)

length_input = st.selectbox(
    "Select Explanation Length",
    [
        "Short (1-2 paragraphs)",
        "Medium (3-4 paragraphs)",
        "Long (detailed explanation)",
    ],
)

# ----------------------------
# Load Prompt Template
# ----------------------------
template = load_prompt("template.json")

# ----------------------------
# Create LCEL Chain
# ----------------------------
chain = template | chat_model

# ----------------------------
# Generate Summary
# ----------------------------
if st.button("Summarize", use_container_width=True):

    if paper_input == "Select...":
        st.warning("Please select a research paper before summarizing.")

    else:
        with st.spinner("Generating summary..."):

            result = chain.invoke(
                {
                    "paper_input": paper_input,
                    "style_input": style_input,
                    "length_input": length_input,
                }
            )

        st.subheader("📄 Summary")
        st.write(result.content)