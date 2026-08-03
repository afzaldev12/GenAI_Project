from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
import re
import json

load_dotenv()

os.environ["HF_HOME"] = r"D:/My Projects/Langchain_models/.cache/huggingface"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


def load_llm():
    pipeline = HuggingFacePipeline.from_model_id(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 150,
            "do_sample": False,        # deterministic output
            "return_full_text": False  # do not include the prompt in output
        },
    )
    return ChatHuggingFace(llm=pipeline)


def parse_model_json(text):
    # Remove Markdown code fences such as ```json ... ```
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(),
                     flags=re.IGNORECASE)

    # Extract the first JSON object even if the model added a short explanation
    start = cleaned.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in response:\n{text}")

    decoder = json.JSONDecoder()
    data, _ = decoder.raw_decode(cleaned[start:])
    return data


chat_model = load_llm()

prompt = """
Return only one JSON object. Do not use Markdown. Do not add explanations.

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



try:
    data = parse_model_json(response.content)

    print("\nSummary:", data["summary"])
    print("Sentiment:", data["sentiment"])

except (json.JSONDecodeError, ValueError, KeyError) as error:
    print("\nCould not parse the model response as expected.")
    print("Reason:", error)