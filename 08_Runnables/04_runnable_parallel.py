from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel

load_dotenv()

prompt1 = PromptTemplate(
    template="Generate a tweet about {topic} for X(formely Twitter).",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a Linkedin post about {topic}.",
    input_variables=["topic"]
)

# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation"
# )

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = RunnableParallel({
    "tweet": RunnableSequence(prompt1, model, parser),
    "linkedin_post": RunnableSequence(prompt2, model, parser)
})

result = chain.invoke({"topic": "AI"})

# print(result)
print("Tweet:\n", result["tweet"])
print("\nLinkedin Post:\n", result["linkedin_post"])