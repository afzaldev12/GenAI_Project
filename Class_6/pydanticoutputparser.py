from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, EmailStr, ValidationError, field_validator
from typing import Literal
import os
import re
import json

load_dotenv()



def load_llm():
    hf_pipeline = HuggingFacePipeline.from_model_id(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 256,
            "do_sample": True,
            "repetition_penalty": 1.1,
            "return_full_text": False,
        },
    )
    return ChatHuggingFace(llm=hf_pipeline)


model = load_llm()

class person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(gt=18, description="The person's age")
    city: str = Field(description="The person's city")


parser = PydanticOutputParser(pydantic_object=person)

template = PromptTemplate(
    template='Generate the name, age, and city of a fictional {place} person \n {format_instructions}',
    input_variables=['place'],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

chain = template | model | parser

final_result = chain.invoke({'place':'Pakistan'})

print(final_result)
