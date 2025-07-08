from typing import Dict
from app.agent.state import MedAssistState
from app.llm import llm
from langchain_core.messages import SystemMessage

def evaluator_node(state: MedAssistState) -> Dict:
    query = get_user_query(state)
    result = state.get("tool_response", "")
    report_context = state.get("report_context", "")

    eval_prompt = f"""
You are a medical safety evaluator AI.

Given:
- User Query: {query}
- Assistant's Draft Answer: {result}
- Patient Report (if any): {report_context}

You must assess:
1. Is this answer safe and medically sound?
2. Is anything missing or could be misinterpreted?
3. Should this be reviewed by a human?

Respond in JSON format:
{{
  "risk_flag": true/false,
  "safety_score": float (0.0 to 1.0),
  "missing_context": true/false
}}
"""

    response = llm.invoke([SystemMessage(content=eval_prompt)]).content

    # Try to parse response as JSON dict
    import json
    try:
        flags = json.loads(response)
    except:
        # Fallback
        flags = {
            "risk_flag": True,
            "safety_score": 0.3,
            "missing_context": True
        }

    new_state = state.copy()
    new_state.update(flags)
    return new_state


# Helper
def get_user_query(state: MedAssistState) -> str:
    return next((msg.content for msg in reversed(state["messages"])
                 if msg.type == "human"), "")
