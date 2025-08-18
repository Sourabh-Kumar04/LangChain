from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
    template="Write a detailed summary on the following text. \n{text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': "https://in.linkedin.com/in/sourabh-kumar04"})

print(result)

