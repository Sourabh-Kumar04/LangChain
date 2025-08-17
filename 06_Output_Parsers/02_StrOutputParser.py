from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os, re

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# 1st pormpt -> detailed report
from langchain.prompts import PromptTemplate

template1 = PromptTemplate(
    template=(
        "You are given a raw transcript of a video on the topic: {topic}.\n\n"
        "Your task is to transform this transcript into a well-structured, detailed, and comprehensive text document. Perform the following steps:\n\n"
        "1. **Transcript Refinement**: Clean and improve the transcript for clarity, grammar, and readability.\n"
        "2. **Concept Expansion**: Add missing explanations, background context, and prerequisite knowledge "
        "to ensure a beginner can understand the material.\n"
        "3. **Depth and Examples**: Insert detailed examples, case studies, and real-world applications.\n"
        "4. **Research Integration**: Reference relevant research papers, articles, or authoritative sources (summarize them, do not copy).\n"
        "5. **Structure**: Organize the content into sections with headings such as Introduction, Prerequisites, Key Concepts, Applications, Advanced Insights, and References.\n"
        "6. **Final Output**: Produce a polished, engaging, and academically reliable document that goes beyond the raw transcript.\n\n"
        "Transcript:\n{transcript}\n\n"
        "Now generate the improved and enriched version."
    ),
    input_variables=["topic", "transcript"]
)

# 2nd prompt -> summary 
from langchain_core.prompts import PromptTemplate

template2 = PromptTemplate(
    template="""
You are a professional research writer and blogger.

Take the following text and **expand it into a long-form professional blog post** 
with the depth and rigor of a research paper.  

Formatting rules:
- Use **clear headings and subheadings**.
- Add **bullet points, numbered lists, and callouts** when helpful.
- Add **emojis (🔥📖💡🔬 etc.)** to make it visually engaging like a Notion blog.
- Include **citations and references** (real research papers, books, blogs).
- Maintain a professional, yet blog-style readable tone.
- Expand with: definitions, context, historical background, key concepts, case studies, real-world applications, challenges, and future directions.
- Ensure smooth flow and logical connections.

Input text:  
{text}

Now write the expanded blog article.
""",
    input_variables=["text"]
)

topic = input(f"Enter the topic: ")

with open("./LangChain/06_output_parsers/02_transcript.txt", "r", encoding="utf-8") as f:
    video_transcript = f.read()

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({
    "topic": topic,
    "transcript": video_transcript
})

# print(result)

# --------------save the result--------

# Clean topic name for safe filenames
def clean_filename(name: str) -> str:
    # Replace spaces with underscores and remove unsafe characters
    return re.sub(r'[^a-zA-Z0-9_\-]', '', name.replace(" ", "_"))

# Generate folder and file path
output_dir = "./LangChain/06_output_parsers/02_StrOutputParser_outputs"
os.makedirs(output_dir, exist_ok=True)  # Create folder if not exists

topic_name = clean_filename(topic)
file_path = os.path.join(output_dir, f"{topic_name}.md")

# Save result to file
with open(file_path, "w", encoding="utf-8") as f:
    f.write(result)

print(f"Result saved automatically to: {file_path}")
