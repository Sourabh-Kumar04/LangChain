from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field, ValidationError
import json, re

load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
#     task="text-generation"
# )

# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation"
# )

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# schema
class Review(BaseModel):

    key_themes: list[str] = Field(description="Write a list of key themes or topics in the review")
    summary: str = Field(description="A brief summary of the review")
    sentiment: str = Field(description="Return the overall sentiment of the review as postive, negative, or neutral")
    pros: Optional[list[str]] = Field(default=None, description="List of pros mentioned in the revied")
    cons: Optional[list[str]] = Field(default=None, description="List of pros mentioned in the revied")
    name: Optional[str] = Field(default=None, description="rite the name of the reviewer")

# structured_model = model.with_structured_output(Review)  # Works with OpenAI, and others aid models which support JSON output
# with_structured_output(Review, validate=True)  # Pydantic validation not supported by HuggingFace models

## prompt explanation
# - LangChain sends your TypedDict schema to the LLM and expects the LLM to respond in exact JSON that matches it.
# - Models like GPT-4, Claude-3, or Mistral-Large with JSON mode will comply.
# - But Mistral-7B-Instruct (and most Hugging Face chat models) just return free-form text → LangChain parser fails → returns None.
# - If you want this to work with Mistral-7B on Hugging Face, you’d need to manually enforce JSON output and parse it yourself, like

# --- helper to fix common JSON issues ---
def safe_json_loads(text: str):
    # replace single quotes → double quotes
    text = text.replace("'", '"')
    # remove trailing commas
    text = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print("Could not parse JSON, returning empty dict")
        return {}

prompt = """
Analyze the following review text and return ONLY valid JSON, nothing else.

Review:
The HP Spectre x360 14 is a luxurious, ultra-versatile 2-in-1 convertible that offers a breathtaking 14-inch 2.8K OLED touchscreen with vibrant 100% DCI-P3 color and adaptive 48–120 Hz refresh rate, powered by Intel’s Core Ultra processors and integrated Arc graphics—making it a smooth performer for productivity, creative tasks, and light gaming 
. Its CNC-machined aluminum chassis delivers premium build quality and four flexible use modes (laptop, tablet, tent, stand), while thoughtful extras like a high-res IR webcam with AI features, stylus input, and quad Bang & Olufsen-tuned speakers enhance its appeal 
. Battery life hits 7–9 hours under real-world usage, though some benchmarks show shorter endurance in heavy tasks 
. Despite a higher price point, its stylish design, robust performance, and creative flexibility make it a compelling choice for professionals, students, and creators seeking a premium blend of form and function.

Return the following fields in the JSON:
- key_themes: Write a list of key themes or topics in the review
- summary: A brief summary of the review
- sentiment: Return the overall sentiment of the review as postive, negative, or neutral
- pros: List of pros mentioned in the revied
- cons: List of cons mentioned in the revied
- name: Write the name of the reviewer
"""

# --- run model ---
raw_output = model.invoke(prompt)

# HuggingFace returns text → extract string
if hasattr(raw_output, "content"):
    raw_text = raw_output.content
else:
    raw_text = str(raw_output)

# --- clean & parse JSON ---
parsed_json = safe_json_loads(raw_text)

# --- validate with Pydantic ---
try:
    validated_result = Review.model_validate(parsed_json)
except ValidationError as e:
    print("Validation failed:", e)
    validated_result = None

print(validated_result, "\n")
print(type(validated_result))
# print("\nKeys: ", validated_result.keys())
print("\nKey Themes: ", validated_result.key_themes)
print("\nSummary: ", validated_result.summary)
print("\nSentiment: ", validated_result.sentiment)
print("\nPros: ", validated_result.pros)
print("\nCons: ", validated_result.cons)
print("\nName: ", validated_result.name)