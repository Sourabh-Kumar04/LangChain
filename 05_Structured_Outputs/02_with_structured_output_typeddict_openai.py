from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated

load_dotenv()

model = ChatOpenAI(model="gpt-4", temperature=0.7, max_completion_tokens=1000)

# schema
class Review(TypedDict):
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[str, "Return sentiment of the review as positive, negative, or neutral"]

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

print(result)
print(type(result))
print("\nSummary: ", result['summary'])
print("\nSentiment: ", result['sentiment'])