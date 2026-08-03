from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import Literal
import os
import re
import json

load_dotenv()

os.environ["HF_HOME"] = r"D:/My Projects/Langchain_models/.cache/huggingface"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


# Pydantic schema: validates the model's extracted JSON
class ReviewAnalysis(BaseModel):
    summary: str = Field(
        min_length=1,
        description="A short summary of the review"
    )
    sentiment: Literal["positive", "negative", "neutral"]

    # Converts "Positive" -> "positive" before Literal validation
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
            "max_new_tokens": 150,
            "do_sample": False,
            "return_full_text": False,
        },
    )
    return ChatHuggingFace(llm=pipeline)


def parse_model_response(text: str) -> ReviewAnalysis:
    # Remove ```json and ``` Markdown fences
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )

    # Find the first JSON object if extra text was generated
    start = cleaned.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in response:\n{text}")

    data, _ = json.JSONDecoder().raw_decode(cleaned[start:])

    # Pydantic validates dictionary fields and values
    return ReviewAnalysis.model_validate(data)


chat_model = load_llm()

prompt = """
Return only one valid JSON object.
Do not use Markdown and do not add explanations.

Required format:
{
  "summary": "short summary",
  "sentiment": "positive" | "negative" | "neutral"
}

Review:
The hardware is great, but the software feels bloated.
There are too many pre-installed apps that I can't remove.
Also, the UI looks outdated compared to other brands.
Hoping for a software update to fix this.
"""

response = chat_model.invoke(prompt)

print("Raw Response:")
print(response.content)

try:
    result = parse_model_response(response.content)

    print("\nSummary:", result.summary)
    print("Sentiment:", result.sentiment)

except (json.JSONDecodeError, ValueError, ValidationError) as error:
    print("\nCould not validate the model response.")
    print("Reason:", error)