from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch

load_dotenv()

prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}.",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Summarize the following text: {text} in a concise manner.",
    input_variables=["text"]
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

report_gen_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda text: len(text.split()) > 100, RunnableSequence(prompt2, model, parser)),  # If text length > 500, summarize
    RunnablePassthrough() # Else, pass through the original text
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

result = final_chain.invoke({"topic": "AI"})

print(result)
