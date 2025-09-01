from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path="LangChain/09_Document_Loader/03_Books/", 
    glob="*.pdf",
    loader_cls=PyPDFLoader    
)

# docs = loader.load()
docs = loader.lazy_load()

for doc in docs:
    print(doc.metadata)

# print(f"Number of documents: {len(docs)}")
# print(docs[0].page_content)
# print("\n\n-------------------\n\n")
# print(docs[215].metadata)


# glob patterns in python
# | Pattern        | Matches                                                     |
# | -------------- | ----------------------------------------------------------- |
# | `*.pdf`        | All files ending with `.pdf` in the current folder          |
# | `*.txt`        | All `.txt` files                                            |
# | `*.md`         | All Markdown files                                          |
# | `*.*`          | All files with an extension                                 |
# | `**/*.pdf`     | All `.pdf` files in **this folder and all subfolders**      |
# | `chapter?.pdf` | `chapter1.pdf`, `chapter2.pdf` … but not `chapter10.pdf`    |
# | `[abc]*.txt`   | Files starting with `a`, `b`, or `c` and ending with `.txt` |

# load() method loads all the documents at once and returns a list of documents.
# lazy_load() method returns an iterator that loads documents one by one, which is more memory efficient for large datasets. (generators in Python)
