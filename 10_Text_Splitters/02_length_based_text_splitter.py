from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("LangChain/09_Document_Loader/03_Books/Blockchain-Engineering-Playbook.pdf")

docs = loader.load()

text_splitter = CharacterTextSplitter(
    separator="",
    chunk_size=100,
    chunk_overlap=20
)

chunks = text_splitter.split_documents(docs)

# print("Chunks: ", chunks)
print(chunks[0])
print(f"Number of chunks: {len(chunks)}")