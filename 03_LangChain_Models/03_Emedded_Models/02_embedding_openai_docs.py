from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

documents = [
    "Delhi is the capital of India",
    "Lucknow is the capital of Uttar Pradesh",
    "Mumbai is the capital of Maharashtra"
]

result = embedding.embed_documents(documents)

print(result)
print("-----------------------")
print(str(result))