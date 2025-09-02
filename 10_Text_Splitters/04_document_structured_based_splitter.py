from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

text = """from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# 1st pormpt -> detailed report
template1 = PromptTemplate(
    template="Write a comprehensive 1000-word report about {topic}. Include definitions, types, examples, and their significance in astrophysics. Do not ask questions back.",
    input_variables=["topic"]
)

# 2nd prompt -> summary 
template2 = PromptTemplate(
    template="Write a 5 line summary on the following text. \n{text}",
    input_variables=["text"]
)

prompt1 = template1.invoke({'topic': "Black holes"})

result = model.invoke(prompt1)

# print(result.content)
# print("\n\n")

prompt2 = template2.invoke({'text': result.content})

result = model.invoke(prompt2)

print(result.content)
"""

text_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    # language=Language.MARKDOWN,  # use for markdown text/files
    chunk_size=200,
    chunk_overlap=0
)

chunks = text_splitter.split_text(text)

# print("Chunks: ", chunks)
print(chunks[1])
print(f"Number of chunks: {len(chunks)}")