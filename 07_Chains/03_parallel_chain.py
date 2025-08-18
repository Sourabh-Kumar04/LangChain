from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

llm1= HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)

llm2= HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation"
)

model1 = ChatHuggingFace(llm=llm1)

model2 = ChatHuggingFace(llm=llm2)

prompt1 = PromptTemplate(
    template="Generate a detailed, professional and research paper based notes on the following text \n{text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template="Genearate a 5 question answer from the following text \n{text}",
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and question answer into a single document \nnotes -> {notes} and QA -> {question_answer}",
    input_variables=["notes", "question_answer"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "question_answer": prompt2 | model2 | parser
})

merge_chain = prompt3 | model2 | parser

chain = parallel_chain | merge_chain

text = '''
# 🤖 Transformers – The Deep‑Learning Engine Behind Large‑Language Models  
*A Self‑Contained, Beginner‑Friendly Guide (Expanded & Research‑Ready)*  

---

### 🚀 Why this Post?  
From the first paper on *Attention Is All You Need* to today’s GPT‑4, Transformers have reshaped every AI field that cares about sequence data. Yet, the terminology and math still feel like a black box for many. This article turns the murk into a **well‑structured, research‑grade tour** that you can read, take notes, and apply.  

We’ll cover:  
1️⃣ History & core ideas  
2️⃣ Building blocks: embeddings, attention, feed‑forward  
3️⃣ Training dynamics and scaling laws  
4️⃣ Real‑world use cases  
5️⃣ Advanced concepts (prefix tuning, bias, explainability)  
6️⃣ Future research directions  
'''

result = chain.invoke({'text': text})

print(result)

# Visualise the chain
chain.get_graph().print_ascii()