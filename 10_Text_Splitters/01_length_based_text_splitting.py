from langchain.text_splitter import CharacterTextSplitter

text = """LangChain is a framework for developing applications powered by language models. It can be used for chatbots, Generative Question-Answering (GQA), summarization, and much more. It enables developers to build applications that can interact with users in a natural way, leveraging the capabilities of large language models. It provides tools and abstractions to manage prompts, chains, agents, and memory, making it easier to create complex applications that can handle various tasks."""

text_splitter = CharacterTextSplitter(
    separator="",
    chunk_size=100,
    chunk_overlap=20
)

chunks = text_splitter.split_text(text)

print("Chunks: ", chunks)
print(f"Number of chunks: {len(chunks)}")