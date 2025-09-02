from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """LangChain is a framework for developing applications powered by language models. It can be used for chatbots, Generative Question-Answering (GQA), summarization, and much more. 

It enables developers to build applications that can interact with users in a natural way, leveraging the capabilities of large language models. It provides tools and abstractions to manage prompts, chains, agents, and memory, making it easier to create complex applications that can handle various tasks.

It is designed to simplify the process of integrating language models into applications, allowing developers to focus on building features rather than dealing with the intricacies of language model APIs.
"""

text_splitter = RecursiveCharacterTextSplitter(
    separators="",
    chunk_size=100,
    chunk_overlap=0
)

chunks = text_splitter.split_text(text)

print("Chunks: ", chunks)
print(f"Number of chunks: {len(chunks)}")

# separators = ["\n\n", "\n", " ", ""]
# separators used in RecursiveCharacterTextSplitter to split text hierarchically based on these separators  
# It tries to split by the first separator, if the chunk is too big, it tries the next separator, and so on.
# This helps in maintaining the context and meaning of the text better than just splitting by character count.
# Here, it will first try to split by double newlines (paragraphs), then by single newlines (lines), then by spaces (words), and finally by characters if needed.
# This hierarchical splitting is particularly useful for documents with structured text, such as articles, books, or reports, where maintaining the integrity of sentences and paragraphs is important.
# If you want to see how the splitting works with different separators, you can modify the separators list and observe the changes in the output chunks.