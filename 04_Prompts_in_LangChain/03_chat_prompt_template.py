from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

chat_template = ChatPromptTemplate([
    ("system", "You are a helpful {domain} expert."),
    ("human", "Explain {topic} in a {style} style with {length} length.")
])

prompt = chat_template.invoke({
    "domain": "machine learning",
    "topic": "neural networks",
    "style": "technical",
    "length": "medium"
})

print(prompt)
