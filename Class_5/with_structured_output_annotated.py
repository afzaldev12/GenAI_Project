from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from typing import Annotated, Literal, TypedDict, cast, Optional
import os
import re
import json

load_dotenv()

os.environ["HF_HOME"] = r"D:/My Projects/Langchain_models/.cache/huggingface"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


# Expected structure of the JSON result
class ReviewResult(TypedDict):
    summary: Annotated[str, "A short summary of the customer review"]
    sentiment: Annotated[
        Literal["positive", "negative", "neutral"],
        "Overall review sentiment"
    ]


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


def parse_review_result(text: str) -> ReviewResult:
    # Removes ```json ... ``` if the model adds Markdown fencing
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )

    # Finds the first JSON object if the model adds introductory text
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("The model response contains no JSON object.")

    data, _ = json.JSONDecoder().raw_decode(cleaned[start:])

    # Runtime validation (TypedDict alone does not validate)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object.")

    if not isinstance(data.get("summary"), str):
        raise ValueError("`summary` must be a string.")

    allowed_sentiments = {"positive", "negative", "neutral"}
    if data.get("sentiment") not in allowed_sentiments:
        raise ValueError(
            "`sentiment` must be positive, negative, or neutral."
        )

    return cast(ReviewResult, data)


chat_model = load_llm()

prompt = """
Return only one JSON object. Do not use Markdown or explanations.

Format:
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
    result = parse_review_result(response.content)

    print("\nSummary:", result["summary"])
    print("Sentiment:", result["sentiment"])

except (json.JSONDecodeError, ValueError) as error:
    print("\nCould not parse the model response.")
    print("Reason:", error)