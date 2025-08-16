from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt
import time
from requests.exceptions import HTTPError
from huggingface_hub.errors import HfHubHTTPError

load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation"
# )

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

st.header("Research Tool")

paper_input = st.selectbox("Select Research Paper Name", ["Attention All You Need", "BERT: Pre-training of Deep BiDirectional Transformers", "GPT: Language Models are Few-Shot Learners", "Diffusion Models Best GANs om Image Synthesis"])

style_input = st.selectbox("Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"])

length_input = st.selectbox("Select Explanation Length", ["Short (1-2 Paragraph)", "Medium (3-4 Paragraphs)", "Long (Detailed Explanation)"])

template = load_prompt("./LangChain/04_Prompts_in_LangChain/02_template.json")

# fill the placholder
prompt = template.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input
})

if st.button('Summarize'):
    # try:
    #     result = model.invoke(prompt)
    #     st.write(result)
    # except HfHubHTTPError:
    #     st.error("The Hugging Face service is currently unavailable. Please try again later.")
    for i in range(5):
        try:
            result = model.invoke(prompt)
            st.write(result.content)
            break
        except (HTTPError, HfHubHTTPError) as e:
            st.write(f"{e}. \nRetrying in {2**i} seconds...")
            time.sleep(2**i)

