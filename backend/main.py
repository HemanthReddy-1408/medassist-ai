# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import router as api_router
from backend.api.auth import router as auth_router

app = FastAPI(
    title="MedAssist API",
    description="Agentic AI Medical Assistant with feedback and memory",
    version="1.0.0"
)

# Register Auth Routes at /auth
app.include_router(auth_router, prefix="/auth")

# Register all API (RAG, feedback, history) at /api
app.include_router(api_router, prefix="/api")

# CORS for frontend (e.g., Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Welcome to MedAssist AI API 🚀"}
