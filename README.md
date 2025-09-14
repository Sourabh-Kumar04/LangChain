# 🦜⛓️ LangChain Learning Hub

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0-green.svg)](https://python.langchain.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![uv Package Manager](https://img.shields.io/badge/uv-package%20manager-orange)](https://github.com/astral-sh/uv)

> **A comprehensive collection of LangChain tutorials, examples, and implementations** — from basic concepts to advanced AI agent workflows.✨ Curated and built by Sourabh Kumar with ❤️ as part of my hands-on journey exploring the real-world power of LangChain and open-source LLMs.

---

## 📖 About This Repository

This repository is a **learning and experimentation hub** for [LangChain](https://www.langchain.com/), designed to guide you from the basics to production-ready AI systems.

It covers:

* 🧱 **Core components** of LangChain
* 🤖 **LLM integrations** (OpenAI, Hugging Face, Anthropic, Google Gemini, etc.)
* 🔗 **Chains, Agents, and Tools**
* 🔍 **RAG systems** (Retrieval-Augmented Generation)
* 🛠️ **Production practices** (error handling, monitoring, optimization)

If you want to **learn LangChain hands-on** or **bootstrap your own AI projects**, this repo is for you.

---

## 🏗️ Repository Structure

```bash
LangChain/
├── 📚 Foundation
│   ├── 01_Introduction_to_LangChain.md
│   └── 02_LangChain_Components.md
│
├── 🤖 Models & Integration
│   ├── 03_LangChain_Models/
│   ├── 04_Prompts_in_LangChain/
│   ├── 05_Structured_Outputs/
│   └── 06_Output_Parsers/
│
├── ⛓️ Chains & Workflows
│   ├── 07_Chains/
│   └── 08_Runnables/
│
├── 🔍 RAG Implementation
│   ├── 09_Document_Loader/
│   ├── 10_Text_Splitters/
│   ├── 11_Vector_Store/
│   ├── 12_Retrievers/
│   └── 13_RAG/
│
├── 🛠️ Advanced Features
│   ├── 14_Tools_in_LangChain/
│   └── 15_Agents/
│       └── 01_web_search_agent.ipynb
│
└── 📋 Configuration
    ├── requirements.txt
    ├── pyproject.toml   # Managed by uv
    ├── .env.example
    └── README.md
```

---

## 🚀 Quick Start

### 🔑 Prerequisites

* Python **3.10+**
* [uv](https://github.com/astral-sh/uv) package manager (recommended over `pip`)
* API keys for providers:

  * `OPENAI_API_KEY` (OpenAI & GPT-OSS-20B)
  * `HUGGINGFACEHUB_API_TOKEN`
  * `ANTHROPIC_API_KEY`
  * `GOOGLE_API_KEY`
  * (Optional) `SERPAPI_API_KEY` or `DUCKDUCKGO` for search tools

---

### ⚡ Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Sourabh-Kumar04/LangChain.git
   cd LangChain
   ```

2. **Install dependencies using uv**

   ```bash
   uv venv
   source .venv/bin/activate   # Linux/Mac
   .venv\Scripts\activate      # Windows

   uv pip install -r requirements.txt
   ```

3. **Set environment variables**

   ```bash
   cp .env.example .env
   # Add your API keys in .env
   ```

4. **Run Jupyter**

   ```bash
   uv pip install jupyter
   jupyter notebook
   ```

---

## 🤖 LLM Integrations

This repo demonstrates **multiple LLM providers**:

* **OpenAI** → `gpt-4`, `gpt-3.5`
* **Hugging Face** → `openai/gpt-oss-20b`
* **Anthropic Claude** → `claude-3.5-sonnet`
* **Google Gemini** → `gemini-2.5-flash`
* **Local Models** via `transformers` + `langchain`

---

## 📚 Learning Path

* 🌱 **Beginner** → Foundations, prompts, basic chains
* 🔧 **Intermediate** → RAG, structured outputs, multi-step workflows
* 🎯 **Advanced** → Agents, multi-LLM orchestration, production best practices

---

## 🎮 Example: Hugging Face `gemini-2.5-flash` Agent

```python
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage
from langchain import hub

from dotenv import load_dotenv
import requests

load_dotenv()

# Hugging Face model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",              # or "gemini-1.5-pro" if you have access
    temperature=0,
    convert_system_message_to_human=True
)

# Web search tool
search = DuckDuckGoSearchAPIWrapper()
tools = [Tool(name="WebSearch", func=search.run, description="Search the internet")]

# Initialize agent
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
print(agent.run("3 ways to reach Goa from Delhi"))
```

---

## 📋 Requirements

### Core

```
langchain>=0.1.0
python-dotenv
jupyter
```

### LLMs

```
openai
huggingface_hub
anthropic
google-generativeai
```

### Tools

```
duckduckgo-search
google-search-results
faiss-cpu
chromadb
pypdf
```

---

## 🛠️ Development

```bash
uv pip install -r requirements-dev.txt
pre-commit install
pytest
black .
isort .
flake8 .
```

---

## 📈 Roadmap

* ✅ LangChain basics
* ✅ Hugging Face OSS LLMs (`openai/gpt-oss-20b`)
* ✅ Agent workflows
* 🚧 LangGraph orchestration
* 🚧 Advanced RAG (re-ranking, citations)
* 🚧 Production deployment (Docker, Kubernetes, cloud)

---

## 👤 Author

Hey 👋, I’m **Sourabh Kumar** — a CS undergrad at the University of Delhi.
I’m passionate about **AI, LangChain, and open-source LLMs** and this repo is a way of documenting + sharing my journey with the community.

* 🐙 [GitHub/Sourabh-Kumar04](https://github.com/Sourabh-Kumar04)
* 💼 [LinkedIn/sourabh-kumar04](https://linkedin.com/in/sourabh-kumar04)
* 💌 Always open to collaboration, feedback, or just a good tech chat!  


---

## ⭐ Support

If this repo resonates with you or saves you time:

* ⭐ **Star it** (makes my day 🌟)
* 🍴 **Fork it** (use it as your learning base)
* 🐛 **Contribute** (PRs welcome)

---

<div align="center">

💡 *“Learning is best when shared — this repo is my small contribution to make LangChain approachable for everyone.”*

</div>  

---