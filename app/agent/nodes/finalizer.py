from typing import Dict
from app.agent.state import MedAssistState
from app.llm import llm
from langchain_core.messages import SystemMessage, HumanMessage

def final_response_node(state: MedAssistState) -> Dict:
    query = get_user_query(state)
    answer = state.get("tool_response", "")
    report_context = state.get("report_context", "")

    final_prompt = f"""
You are a medical assistant. Given the following:

User Query:
{query}

Generated Answer:
{answer}

Uploaded Report Context (if any):
{report_context}

Format a polished final response that:
- Is clear and medically accurate
- Does not include unsafe advice
- Optionally cite source if present (e.g., PubMed IDs)
Return only the final response text.
"""

    response = llm.invoke([SystemMessage(content=final_prompt)]).content.strip()

    new_state = state.copy()
    new_state["final_response"] = response
    return new_state


def get_user_query(state: MedAssistState) -> str:
    return next((msg.content for msg in reversed(state["messages"])
                 if msg.type == "human"), "")
