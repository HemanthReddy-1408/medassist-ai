from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from enum import Enum
from app.agent.graph import build_medassist_graph
from backend.db.models import save_interaction, get_thread_history, save_feedback
from backend.api.auth import get_current_user
from langchain_core.messages import HumanMessage

router = APIRouter()
graph = build_medassist_graph()

# ------------------------- Query Handling -------------------------

class QueryRequest(BaseModel):
    message: str
    thread_id: str

class QueryResponse(BaseModel):
    response: str

@router.get("/health")
def health_check():
    return {"status": "ok", "message": "MedAssist API is running!"}

@router.post("/query", response_model=QueryResponse)
def query_agent(
    payload: QueryRequest,
    current_user: dict = Depends(get_current_user)
):
    try:
        user_id = current_user["email"]
        config = {"configurable": {"thread_id": payload.thread_id}}

        state = {
            "messages": [HumanMessage(content=payload.message)],
            "final_response": None,
            "__tool__": None,
            "risk_flag": False
        }

        result = graph.invoke(state, config=config)

        save_interaction(
            thread_id=payload.thread_id,
            user_input=payload.message,
            agent_response=result["final_response"],
            risk_flag=result.get("risk_flag", False),
            user_id=user_id
        )

        return {"response": result["final_response"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

# ------------------------- History Retrieval -------------------------

@router.get("/history/{thread_id}")
def get_history(
    thread_id: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        user_id = current_user["email"]
        history = get_thread_history(user_id=user_id, thread_id=thread_id)
        for h in history:
            h["_id"] = str(h["_id"])  # Convert ObjectId to string
        return {"thread_id": thread_id, "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"History error: {e}")

# ------------------------- Feedback Submission -------------------------

class Rating(str, Enum):
    like = "like"
    dislike = "dislike"

class FeedbackRequest(BaseModel):
    thread_id: str
    message: str
    rating: Rating
    comment: Optional[str] = None

class FeedbackResponse(BaseModel):
    status: str
    message: str

@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(
    feedback: FeedbackRequest,
    current_user: dict = Depends(get_current_user)
):
    try:
        user_id = current_user["email"]
        save_feedback(
            thread_id=feedback.thread_id,
            message=feedback.message,
            rating=feedback.rating,
            comment=feedback.comment,
            user_id=user_id  # 🔥 Add this to properly associate feedback
        )
        return {"status": "success", "message": "Feedback recorded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback error: {e}")
