from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv()

def word_count(text: str) -> int:
    return len(text.split())

prompt = PromptTemplate(
    template="Write a joke about {topic}.",
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

joke_gen_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    # "word_count": RunnableLambda(word_count)
    "word_count": RunnableLambda(lambda text: len(text.split()))
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({"topic": "AI"})

final_result = """Joke: \n{} \n\nWord Count: \n{}""".format(result["joke"], result["word_count"])

# print(result)
print(final_result)