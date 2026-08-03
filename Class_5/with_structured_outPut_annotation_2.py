from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, data
from typing import Annotated, Literal, cast
from typing_extensions import TypedDict, NotRequired
import os
import re
import json

load_dotenv()

os.environ["HF_HOME"] = r"D:/My Projects/Langchain_models/.cache/huggingface"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


# Expected JSON structure
class ReviewResult(TypedDict):
    summary: Annotated[str, "A concise summary of the review"]
    sentiment: Annotated[
        Literal["positive", "negative", "neutral"],
        "The overall sentiment"
    ]
    key_topics: Annotated[
        list[str],
        "Main topics discussed in the review"
    ]

    # These keys are optional: the model should omit them if none exist.
    pros: NotRequired[Annotated[list[str], "Positive points mentioned"]]
    cons: NotRequired[Annotated[list[str], "Negative points mentioned"]]


def load_llm():
    pipeline = HuggingFacePipeline.from_model_id(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 300,
            "do_sample": False,
            "return_full_text": False,
        },
    )
    return ChatHuggingFace(llm=pipeline)


def parse_review_result(text: str) -> ReviewResult:
    # Remove Markdown code fences if the model returns ```json ... ```
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )

    # Find and decode the first JSON object
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("No JSON object was found in the model response.")

    data, _ = json.JSONDecoder().raw_decode(cleaned[start:])

    # Runtime validation
    if not isinstance(data, dict):
        raise ValueError("The response must be a JSON object.")

    required_keys = {"summary", "sentiment", "key_topics"}
    missing_keys = required_keys - data.keys()

    if missing_keys:
        raise ValueError(f"Missing required field(s): {', '.join(missing_keys)}")

    if not isinstance(data["sentiment"], str):
        raise ValueError("`sentiment` must be a string.")

    # Convert Positive, POSITIVE, and positive into the same value
    data["sentiment"] = data["sentiment"].strip().lower()

    if data["sentiment"] not in {"positive", "negative", "neutral"}:
        raise ValueError(
            "`sentiment` must be positive, negative, or neutral."
        )

    if (
        not isinstance(data["key_topics"], list)
        or not all(isinstance(topic, str) for topic in data["key_topics"])
    ):
        raise ValueError("`key_topics` must be a list of strings.")

    # Validate optional keys only if the model includes them
    for field in ("pros", "cons"):
        if field in data:
            if (
                not isinstance(data[field], list)
                or not all(isinstance(item, str) for item in data[field])
            ):
                raise ValueError(f"`{field}` must be a list of strings.")

    return cast(ReviewResult, data)


chat_model = load_llm()

prompt = """
Return only one valid JSON object.
Do not add Markdown, explanations, or ```json code fences.

Required JSON format:
{
  "summary": "short summary",
  "sentiment": "positive, negative, or neutral",
  "key_topics": ["topic 1", "topic 2"]
}

Optional fields:
- Include "pros": ["..."] only when positive points exist.
- Include "cons": ["..."] only when negative points exist.
- Omit those keys if the review contains no pros or cons.

Review:
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say,
it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes
everything lightning fast—whether I’m gaming, multitasking, or editing
photos. The 5000mAh battery easily lasts a full day even with heavy use,
and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches,
though I don't use it often. What really blew me away is the 200MP camera—
the night mode is stunning, capturing crisp, vibrant images even in low
light. Zooming up to 100x actually works well for distant objects, but
anything beyond 30x loses quality.

However, the weight and size make it uncomfortable for one-handed use.
Samsung’s One UI still comes with bloatware, and the $1,300 price tag is
very expensive.
"""

response = chat_model.invoke(prompt)

print("Raw Response:")
print(response.content)

try:
    result = parse_review_result(response.content)

    print("\nSummary:")
    print(result["summary"])

    print("\nSentiment:")
    print(result["sentiment"])

    print("\nKey Topics:")
    for topic in result["key_topics"]:
        print("-", topic)

    # .get() returns an empty list when optional fields do not exist
    if result.get("pros"):
        print("\nPros:")
        for pro in result["pros"]:
            print("-", pro)

    if result.get("cons"):
        print("\nCons:")
        for con in result["cons"]:
            print("-", con)

except (json.JSONDecodeError, ValueError) as error:
    print("\nCould not parse the model response.")
    print("Reason:", error)