import json

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


# ==================================================
# 1. Define JSON Schema
# ==================================================

student_schema = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Student's full name"
        },
        "age": {
            "type": "integer",
            "description": "Student's age"
        },
        "course": {
            "type": "string",
            "description": "Student's course"
        },
        "university": {
            "type": "string",
            "description": "Student's university"
        }
    },
    "required": [
        "name",
        "age",
        "course",
        "university"
    ]
}

schema = json.dumps(student_schema, indent=4)

parser = JsonOutputParser()


# ==================================================
# 2. Load Local Model
# ==================================================

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0,
    do_sample=False,
    return_full_text=False,
)

llm = HuggingFacePipeline(pipeline=pipe)
chat_model = ChatHuggingFace(llm=llm)


# ==================================================
# 3. Prompt
# ==================================================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an information extraction assistant.

Return ONLY valid JSON.

Follow this JSON Schema exactly.

{schema}
"""
        ),
        (
            "human",
            "{text}"
        )
    ]
)


# ==================================================
# 4. Chain
# ==================================================

chain = prompt | chat_model | parser


# ==================================================
# 5. Input
# ==================================================

text = """
Ahmed is 22 years old.
He studies Artificial Intelligence.
He is enrolled at Agriculture University Peshawar.
"""

result = chain.invoke(
    {
        "text": text,
        "schema": schema
    }
)

print(result)