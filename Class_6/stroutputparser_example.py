import os

# Set Hugging Face settings before loading the model.
os.environ["HF_HOME"] = r"D:/My Projects/Langchain_models/.cache/huggingface"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

load_dotenv()


def load_llm():
    hf_pipeline = HuggingFacePipeline.from_model_id(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 150,
            "do_sample": False,
            "return_full_text": False,
        },
    )
    return ChatHuggingFace(llm=hf_pipeline)


# Load the model before calling model.invoke().
model = load_llm()

report_prompt = PromptTemplate.from_template(
    "Write a detailed report on the following topic:\n\n{topic}"
)

summary_prompt = PromptTemplate.from_template(
    "Write a summary in exactly 5 lines of the following text:\n\n{text}"
)

report_message = model.invoke(
    report_prompt.invoke(
        {"topic": "The impact of climate change on global agriculture."}
    )
)

summary_message = model.invoke(
    summary_prompt.invoke({"text": report_message.content})
)

print("\n--- REPORT ---\n")
print(report_message.content)

print("\n--- 5-LINE SUMMARY ---\n")
print(summary_message.content)