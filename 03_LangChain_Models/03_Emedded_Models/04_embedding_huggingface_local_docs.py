from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    "Delhi is the capital of India",
    "Lucknow is the capital of Uttar Pradesh",
    "Mumbai is the capital of Maharashtra"
]

vector = embedding.embed_documents(documents)

print(vector)
print("-----------------------")
print(str(vector))