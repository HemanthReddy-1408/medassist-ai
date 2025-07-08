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

medassist-ai/
│
├── app/ # LangGraph agent logic
│ ├── agent/
│ │ ├── graph.py # LangGraph builder
│ │ ├── state.py # State schema
│ │ └── nodes/ # Planner, tools, finalizer, etc.
│ └── llm.py # Groq / LLM setup
│
├── backend/ # FastAPI backend
│ ├── api/
│ │ ├── routes.py # Query, feedback, history
│ │ └── auth.py # JWT auth
│ └── db/
│ ├── models.py # DB models
│ └── mongo_client.py # Mongo connector
│
├── streamlit_ui/ # Streamlit frontend
│ ├── app.py # Main UI
│ ├── chat.py # Chat + feedback
│ ├── auth.py # Login/signup
│ └── feedback.py # General feedback form
│
├── outputs/ # Generated graph image
│
├── data/ # Sample queries or logs
├── whisper_stt/ # Whisper input (planned)
├── .env.example # Env config template
├── requirements.txt # Python dependencies
└── README.md


---

## 📦 Tech Stack

| Layer       | Toolset                                      |
|-------------|----------------------------------------------|
| **Agent**   | LangGraph + LangChain                        |
| **LLM**     | `Gemma-2B It` via **Groq API**               |
| **RAG**     | PubMed, Wikipedia, Tavily Search             |
| **Frontend**| Streamlit                                    |
| **Backend** | FastAPI + JWT                                |
| **Database**| MongoDB (for memory + feedback)              |

---

## ⚙️ Quick Setup

```bash
# 1. Clone the repo
git clone https://github.com/HemanthReddy-1408/medassist-ai.git
cd medassist-ai

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add environment variables
cp .env.example .env
# Then open `.env` and fill:
# - GROQ_API_KEY=
# - MONGODB_URI=
# - JWT_SECRET=
# - TAVILY_API_KEY=

# 5. Run FastAPI backend
uvicorn backend.main:app --reload

# 6. In a new terminal, run Streamlit frontend
streamlit run streamlit_ui/app.py


🔐 Authentication
=>JWT-based login & token verification
=>Access is required for querying, feedback, and history routes

🧪 Sample Prompt
User: I've been feeling tired even after sleeping, get headaches often, look pale, and feel breathless climbing stairs. What could be the reason?

MedAssist AI: It might indicate anemia or iron deficiency. A blood test (CBC) is advised. Please consult a healthcare professional.
