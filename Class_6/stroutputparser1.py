from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os


# Set Hugging Face settings before loading the model.
os.environ["HF_HOME"] = r"D:/My Projects/Langchain_models/.cache/huggingface"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


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

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template="Write a detailed report on the following topic:\n\n{topic}",
    input_variables=["topic"],
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template="Write a summary in exactly 5 lines of the following text:\n\n{text}",
    input_variables=["text"],
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic": "The impact of climate change on global agriculture."})

print("\n--- REPORT ---\n")
print(result)