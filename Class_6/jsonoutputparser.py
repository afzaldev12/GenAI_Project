from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
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


parser = JsonOutputParser()

prompt = PromptTemplate(
    template="""
Answer ONLY in JSON.

{format_instructions}

Question:
{query}
""",
    input_variables=["query"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)


chain = prompt | model | parser

result = chain.invoke({"query": "Generate a student with name, age, and city."})

print("\n--- JSON OUTPUT ---\n")
print(result)

