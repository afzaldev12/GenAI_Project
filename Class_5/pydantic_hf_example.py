from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from pydantic import BaseModel, Field, EmailStr, ValidationError, field_validator
from typing import Literal
import os
import re
import json

load_dotenv()

# Local Hugging Face cache settings
os.environ["HF_HOME"] = r"D:/My Projects/Langchain_models/.cache/huggingface"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


# ------------------------------------------------
# Part 1: Basic Pydantic concepts
# ------------------------------------------------

# Nested model
class Address(BaseModel):
    city: str
    country: str = "India"  # Default value


class Student(BaseModel):
    # Required fields
    name: str
    email: EmailStr

    # Default values and number constraints
    age: int = Field(default=18, ge=5, le=100)
    cgpa: float = Field(
        default=0.0,
        ge=0.0,
        le=4.0,
        description="CGPA must be between 0.0 and 4.0"
    )

    # List with a safe default
    hobbies: list[str] = Field(default_factory=list)

    # Optional nested object
    address: Address | None = None

    # Limited allowed values
    status: Literal["active", "inactive"] = "active"

    # Custom validation / cleanup
    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        return value.strip().title()


student_data = {
    "name": "  john doe ",
    "email": "john@example.com",
    "age": 21,
    "cgpa": 3.7,
    "hobbies": ["Python", "AI"],
    "address": {
        "city": "Delhi",
        "country": "India"
    },
    "status": "active"
}

try:
    # Validate a Python dictionary
    student = Student.model_validate(student_data)

    print("----- Basic Pydantic Example -----")
    print(student)
    print("\nName:", student.name)
    print("CGPA:", student.cgpa)

    # Convert the Pydantic model back into a dictionary
    print("\nDictionary:")
    print(student.model_dump())

    # Convert the model into JSON text
    print("\nJSON:")
    print(student.model_dump_json(indent=2))

except ValidationError as error:
    print("Student validation failed:")
    print(error)


# ------------------------------------------------
# Part 2: Pydantic + local Hugging Face model
# ------------------------------------------------

class ReviewResult(BaseModel):
    summary: str = Field(
        min_length=1,
        description="A short summary of the review"
    )

    sentiment: Literal["positive", "negative", "neutral"]

    key_topics: list[str] = Field(
        default_factory=list,
        description="Main subjects mentioned in the review"
    )

    # Converts "Positive" or "POSITIVE" to "positive"
    @field_validator("sentiment", mode="before")
    @classmethod
    def normalize_sentiment(cls, value):
        if isinstance(value, str):
            return value.strip().lower()
        return value


def load_llm():
    pipeline = HuggingFacePipeline.from_model_id(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 200,
            "do_sample": False,
            "return_full_text": False,
        },
    )

    return ChatHuggingFace(llm=pipeline)


def parse_llm_response(text: str) -> ReviewResult:
    # Removes Markdown code fences: ```json ... ```
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text.strip(),
        flags=re.IGNORECASE
    )

    # Finds JSON even if the model writes introductory text
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("No JSON object found in the model response.")

    data, _ = json.JSONDecoder().raw_decode(cleaned[start:])

    # Pydantic validates the model's JSON data
    return ReviewResult.model_validate(data)


review = """
The phone has excellent performance and great battery life.
The camera quality is impressive, especially at night.
However, the device is expensive and feels too heavy for one-handed use.
"""

prompt = f"""
Return only one valid JSON object.
Do not use Markdown or explanations.

Use exactly this format:
{{
  "summary": "short summary",
  "sentiment": "positive, negative, or neutral",
  "key_topics": ["topic 1", "topic 2"]
}}

Review:
{review}
"""

chat_model = load_llm()
response = chat_model.invoke(prompt)

print("\n----- Local Hugging Face + Pydantic Example -----")
print("Raw model response:")
print(response.content)

try:
    result = parse_llm_response(response.content)

    print("\nValidated result:")
    print("Summary:", result.summary)
    print("Sentiment:", result.sentiment)
    print("Key topics:", result.key_topics)

    print("\nValidated JSON:")
    print(result.model_dump_json(indent=2))

except (json.JSONDecodeError, ValueError, ValidationError) as error:
    print("\nModel response could not be validated:")
    print(error)