from typing import Dict
from app.agent.state import MedAssistState

def hitl_node(state: MedAssistState) -> Dict:
    # Extract the response and query
    query = get_user_query(state)
    response = state.get("tool_response", "")

    # Simulate human review approval
    print("\n⚠️ [HITL TRIGGERED] Risky response flagged.")
    print(f"User Query: {query}")
    print(f"LLM Response: {response}")
    print("🧑‍⚕️ [Simulated Approval]: OK by human reviewer.\n")

    # Update the final_response field
    new_state = state.copy()
    new_state["final_response"] = "[HITL Approved] " + response
    return new_state


def get_user_query(state: MedAssistState) -> str:
    return next((msg.content for msg in reversed(state["messages"])
                 if msg.type == "human"), "")
