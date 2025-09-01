from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("LangChain/09_Document_Loader/02_Cryptography_Privacy_Security_Handbook.pdf")

docs = loader.load()

# print(docs)
# print(f"Number of pages: {len(docs)}")
print(docs[0].page_content)
print("\n\n-------------------\n\n")
print(docs[0].metadata)

# for scanned documents, PyPDF will not work. You can use other libraries like PyMuPDF, pdfplumber, etc.
# Simple, clean PDF -> PyPDFLoader
# PDF with tables/columns -> PDFPlumberLoader
# Scanned/Image PDF -> UnstructuredPDFLoader or AmazonTextractLoader
# Need layout and image data -> PyMuPDFLoader
# Want best structured extraction -> UnstructuredPDFLoader