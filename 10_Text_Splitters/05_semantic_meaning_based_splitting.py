from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text_splitter = SemanticChunker(
    embeddings=embedding,
    # chunk_size=100,
    # chunk_overlap=20
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=3
)

sample = """
LangChain is a framework for developing applications powered by language models. It can be used for chatbots, Generative Question-Answering (GQA), summarization, and much more. It enables developers to build applications that can interact with users in a natural way, leveraging the capabilities of large language models. It provides tools and abstractions to manage prompts, chains, agents, and memory, making it easier to create complex applications that can handle various tasks.India and Pakistan are two neighboring countries in South Asia with a long and complex history. The relationship between the two countries has been marked by periods of conflict and cooperation, with tensions often arising over issues such as Kashmir and cross-border terrorism by Pakistan.Kashmir is the part of India.

Virat Kohli is one of the greatest cricketers of the modern era, widely celebrated for his consistency, passion, and match-winning abilities. Born on 5 November 1988 in Delhi, India, he rose to prominence through his aggressive batting style, sharp cricketing mind, and incredible fitness levels. Known as the "Run Machine," Kohli has broken numerous batting records across formats, including being one of the fastest to score centuries in One Day Internationals. As the former captain of the Indian cricket team, he led India to historic series wins overseas and maintained a fiercely competitive spirit on the field. Beyond cricket, Kohli is admired for his discipline, leadership qualities, and commitment to fitness, which has inspired millions of young athletes. He continues to play a pivotal role in Indian cricket, while also being a global sports icon and a role model for aspiring cricketers.

"""

docs = text_splitter.create_documents([sample])

print("Chunks: ", docs)
# print(docs[0])
print(f"Number of chunks: {len(docs)}")