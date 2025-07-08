from backend.db.mongo_client import db
from datetime import datetime
from typing import Optional

def save_interaction(user_id, thread_id, user_input, agent_response, risk_flag=False):
    db.interactions.insert_one({
        "user_id": user_id,
        "thread_id": thread_id,
        "user_input": user_input,
        "agent_response": agent_response,
        "risk_flag": risk_flag,
        "timestamp": datetime.utcnow()
    })

def get_thread_history(user_id, thread_id):
    return list(db.interactions.find({
        "thread_id": thread_id,
        "user_id": user_id
    }).sort("timestamp", 1))

def save_feedback(thread_id: str, message: str, rating: str, comment: Optional[str] = None, user_id: str = None):
    db.feedback.insert_one({
        "user_id": user_id,   # ✅ include user_id
        "thread_id": thread_id,
        "message": message,
        "rating": rating,      # "like" or "dislike"
        "comment": comment,
        "timestamp": datetime.utcnow()
    })
