from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os


os.environ["HF_HOME"] = r"D:/My Projects/Langchain_models/.cache/huggingface"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


def load_llm():
    hf_pipeline = HuggingFacePipeline.from_model_id(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 150,
            "do_sample": False,
            "repetition_penalty": 1.1,
            "return_full_text": False,
        },
    )
    return ChatHuggingFace(llm=hf_pipeline)


model = load_llm()

schemas = [
    ResponseSchema(name="name", description="The student's name"),
    ResponseSchema(name="age", description="The student's age"),
    ResponseSchema(name="city", description="The student's city"),
]
parser = StructuredOutputParser.from_response_schemas(schemas)


template = ChatPromptTemplate.from_messages([
    ("system", "You always respond in English only, and follow formatting instructions exactly."),
    ("human", "Invent a fictional student and provide their name, age, and city.\n{format_instructions}"),
])

template = template.partial(format_instructions=parser.get_format_instructions())

response = model.invoke(template.invoke({}))

input_variables=[],
partial_variables={
    "format_instructions": parser.get_format_instructions()
},

chain = template | model | parser
# response = model.invoke(template.invoke({}))

try:
    result = chain.invoke({'topic': 'fictional student'})
    print("\n--- STRUCTURED OUTPUT ---\n")
    print(result)
except Exception as exc:
    print(f"Could not parse model output: {exc}")
    print("Raw response:", response.content)