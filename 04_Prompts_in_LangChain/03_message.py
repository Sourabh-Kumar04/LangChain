# Types of messages in LangChain:
# 1. HumanMessage: Represents a message from the user.
# 2. AIMessage: Represents a message from the AI.
# 3. SystemMessage: Represents a message from the system, which can be used to set context or instructions.

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation"
# )

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation"
)

# llm = HuggingFaceEndpoint(
#     repo_id="mistralai/Mistral-7B-Instruct-v0.2",
#     task="text-generation"
# )

model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me about the LangChain framework.")
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)