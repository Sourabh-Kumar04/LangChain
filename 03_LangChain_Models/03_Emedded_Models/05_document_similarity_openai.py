from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting style.",
    "Sachin Tendulkar is considered one of the greatest batsmen in cricket history.",
    "Rohit Sharma is the captain of the Indian cricket team and holds the record for the highest individual score in ODIs.",
    "MS Dhoni is a former captain of the Indian cricket team and is known for his calm demeanor.",
    "Ravindra Jadeja is an all-rounder known for his exceptional fielding and bowling skills.",
    "Jasprit Bumrah is a fast bowler known for his unique bowling action and death-over skills.",
    "Hardik Pandya is an all-rounder known for his explosive batting and fast bowling.",
    "KL Rahul is a versatile batsman who can play in various formats of the game.",
    "Shikhar Dhawan is known for his aggressive opening batting style in limited-overs cricket.",
    "Yuzvendra Chahal is a leg-spinner known for his wicket-taking ability in T20 cricket."
]

query = "Tell me about Virat Kohli"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = (cosine_similarity([query_embedding], [doc_embeddings]))[0]

index, score = sorted(list(enumerate(scores)), keys=lambda x:x[1])[-1]

print(f"query: {query}")
print(documents[index])
print("Similarity Score: ", score)