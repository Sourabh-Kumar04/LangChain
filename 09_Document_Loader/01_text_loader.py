from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate(
    template="Write a detailed report on {topic}.",
    input_variables=["topic"]
)

# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation"
# )

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

loader = TextLoader("LangChain/09_Document_Loader/01_LLM_architecture_behind_chatgpt.md", encoding="utf8")

docs = loader.load()

# print(docs)
# print(type(docs))
# print(docs[0])
# print(type(docs[0]))
print(docs[0].page_content)
print("\n\n-----------------------\n\n")
# print(docs[0].metadata)

chain = prompt | model | parser

result = chain.invoke({"content": docs[0].page_content})

print(result)

# document loader in langchain always returns a list of documents i.e. list[Document]