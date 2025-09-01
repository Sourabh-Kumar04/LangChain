from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="LangChain/09_Document_Loader/06_tasks.csv", encoding="utf-8")

docs = loader.load()

print(f"Number of documents: {len(docs)}")
print(docs[0])
print("\n\n-------------------\n\n")
print(docs[0].metadata)