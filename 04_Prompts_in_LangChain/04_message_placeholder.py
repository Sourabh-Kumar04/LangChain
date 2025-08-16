from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# chat template
chat_template = ChatPromptTemplate([
    ('system', "You are a helpful customer support agent."),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

chat_history = []

# load chat history
with open('LangChain/04_Prompts_in_LangChain/04_chat_history.txt') as f:
    chat_history.extend(f.readlines())

# create a prompt with chat history
prompt = chat_template.invoke({
    'chat_history': chat_history,
    'query': "Where is my refund for order #8734"
})

print(prompt)