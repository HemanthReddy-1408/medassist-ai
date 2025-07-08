# medassist-ai
MedAssist AI – Agentic AI Medical Assistant using LangGraph + RAG + FastAPI + Streamlit
# 🩺 MedAssist AI

**MedAssist AI** is an intelligent, multi-tool medical assistant built using LangGraph, LangChain, and Groq.  
It leverages RAG, memory, and human-in-the-loop feedback for safe and insightful responses to user medical queries.

> 🧠 Designed for personalized, contextual, and explainable healthcare assistance.

---

## 🧭 System Architecture

![MedAssist Agent Graph](https://github.com/HemanthReddy-1408/medassist-ai/blob/main/outputs/graph/medassist_graph.png)

---

## 🚀 Features

- 🧠 Agentic LangGraph flow with planning, tools, finalizer
- 🔍 Multi-source **RAG** (PubMed, Wikipedia, Tavily)
- 🧵 Session-based memory using `thread_id` and MongoDB
- 💬 Feedback system with like/dislike + optional comments
- 🔐 JWT-based login and access restriction
- 📊 Streamlit UI for real-time querying and feedback
- 🗂️ API built with FastAPI

---

## 🧱 Project Structure

