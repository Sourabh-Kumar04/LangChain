from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import streamlit as st
from langchain_core.prompts import PromptTemplate

# # for downloading in D drive
# import os
# os.environ["HF_HOME"] = 'D:/huggingface_cache'

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",

    pipeline_kwargs=dict(
        temperature=0.7,
        max_new_tokens=100,
    )
)

model = ChatHuggingFace(llm=llm)

st.header("Research Tool")

paper_input = st.selectbox("Select Research Paper Name", ["Attention All You Need", "BERT: Pre-training of Deep BiDirectional Transformers", "GPT: Language Models are Few-Shot Learners", "Diffusion Models Best GANs om Image Synthesis"])

style_input = st.selectbox("Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"])

length_input = st.selectbox("Select Explanation Length", ["Short (1-2 Paragraph)", "Medium (3-4 Paragraphs)", "Long (Detailed Explanation)"])

# template
template = PromptTemplate(
    template="""
Please summarize the research paper titled "{paper_input}" with the following specifications:
Explanation Style: {style_input}
Explanation Length: {length_input}
1. Mathematical Details:
    - Include relevant mathematical equations if present in the paper
    - Explain the mathematical concepts using simple, intuitive code snippets where applicable
2. Analogies:
    - Use the relevant analogies to simplify complex ideas.
If certain information is not available in the paper, repond with : "Insufficient information available" instead of guessing.
Ensure the summary is clear, accurate, and aligned with the provided style and length.
""",
input_variables=["paper_input", "style_input", "length_input"]
)

# fill the placholder
prompt = template.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input
})

if st.button('Summarize'):
    result = model.invoke(prompt)
    st.write(result.content)
