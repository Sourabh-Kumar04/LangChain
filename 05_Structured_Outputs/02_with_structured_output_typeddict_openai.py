from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI(model="gpt-4", temperature=0.7, max_completion_tokens=1000)

# schema
class Review(TypedDict):
    
    key_themes: list[str] = Field(description="Write a list of key themes or topics in the review")
    summary: str = Field(description="A brief summary of the review")
    sentiment: str = Field(description="Return the overall sentiment of the review as postive, negative, or neutral")
    pros: Optional[list[str]] = Field(default=None, description="List of pros mentioned in the revied")
    cons: Optional[list[str]] = Field(default=None, description="List of pros mentioned in the revied")
    name: Optional[str] = Field(default=None, description="rite the name of the reviewer")


structured_model = model.with_structured_output(Review)

## prompt explanation
# - LangChain sends your TypedDict schema to the LLM and expects the LLM to respond in exact JSON that matches it.
# - Models like GPT-4, Claude-3, or Mistral-Large with JSON mode will comply.
# - But Mistral-7B-Instruct (and most Hugging Face chat models) just return free-form text → LangChain parser fails → returns None.
# - If you want this to work with Mistral-7B on Hugging Face, you’d need to manually enforce JSON output and parse it yourself, like

prompt = """
The HP Spectre x360 14 is a luxurious, ultra-versatile 2-in-1 convertible that offers a breathtaking 14-inch 2.8K OLED touchscreen with vibrant 100% DCI-P3 color and adaptive 48–120 Hz refresh rate, powered by Intel’s Core Ultra processors and integrated Arc graphics—making it a smooth performer for productivity, creative tasks, and light gaming 
. Its CNC-machined aluminum chassis delivers premium build quality and four flexible use modes (laptop, tablet, tent, stand), while thoughtful extras like a high-res IR webcam with AI features, stylus input, and quad Bang & Olufsen-tuned speakers enhance its appeal 
. Battery life hits 7–9 hours under real-world usage, though some benchmarks show shorter endurance in heavy tasks 
. Despite a higher price point, its stylish design, robust performance, and creative flexibility make it a compelling choice for professionals, students, and creators seeking a premium blend of form and function.
"""

result = structured_model.invoke(prompt)

print(result, "\n")
print(type(result))
# print("\nKeys: ", result.keys())
print("\nKey Themes: ", result.key_themes)
print("\nSummary: ", result.summary)
print("\nSentiment: ", result.sentiment)
print("\nPros: ", result.pros)
print("\nCons: ", result.cons)
print("\nName: ", result.name)