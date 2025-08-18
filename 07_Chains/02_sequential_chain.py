from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)

prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Genearate a 5 pointer summary from the following text \n{text}",
    input_variables=['text']
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = prompt1| model | parser | prompt2 | model | parser

result = chain.invoke({'topic': '@Sourabh-Kumar04 github'})

print(result)

# Visualise the chain
chain.get_graph().print_ascii()