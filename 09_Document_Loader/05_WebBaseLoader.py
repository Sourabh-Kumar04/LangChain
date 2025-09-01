# WebBasedLoader uses requests and BeautifulSoup to scrape web pages.
# Limitations: only static pages, no JS rendering, etc.
# For more complex web scraping, consider using tools like Scrapy or Selenium.

from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation"
# )

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="Answer the following question \n{question} from the following text \n{text}.",
    input_variables=["question", "text"]
)

parser = StrOutputParser()

url1="https://www.linkedin.com/in/sourabh-kumar04/"
url2="https://github.com/Sourabh-Kumar04"

urls=[
    "https://github.com/Sourabh-Kumar04",
    "https://www.linkedin.com/in/sourabh-kumar04/"
]

# loader = WebBaseLoader(urls)
loader = WebBaseLoader(url2)

docs = loader.load()

# print(f"Number of documents: {len(docs)}")
# print(docs)
# print(docs[0].page_content)
# print("\n\n-------------------\n\n")
# print(docs[0].metadata)

chain = prompt | model | parser

result = chain.invoke({"question": "Who is Sourabh Kumar?", "text": docs[0].page_content})

print(result)

# We can make a chrome plugin to scrape the current page and send the content to langchain for processing. 
# We can use selenium to automate the browser and scrape the content.
# User in real-time ask questions about the current pagem or chat with the content of the page.