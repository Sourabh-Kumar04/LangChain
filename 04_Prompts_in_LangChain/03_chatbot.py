from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation"
#     # repo_id="HuggingFaceH4/zephyr-7b-beta", 
#     # task="text-generation"
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

chat_history = [
    SystemMessage(content="You are a helpful AI assistant.")
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() == "exit":
        print("Exiting the chatbot. Goodbye!")
        break
    try: 
        response = model.invoke(chat_history)
        chat_history.append(AIMessage(content=response.content))
        print(f"AI: {response.content}")
    except Exception as e:
        print(f"An error occurred: {e}. Please try again later.")

print(chat_history)
