from typing import Dict
from app.agent.state import MedAssistState
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from app.llm import llm


def react_planner_node(state: MedAssistState) -> Dict:
    topic = state.get("topic", "").strip()
    user_message = next((msg.content for msg in reversed(state["messages"])
                         if msg.type == "human"), "")

    planner_prompt = f"""
You are a medical assistant tool planner. Based on the topic and user query, decide which tool to use:
- "pubmed" → for drug information, treatment, or clinical references
- "wikipedia" → for general medical knowledge or symptom explanations
- "tavily" → for summarizing uploaded reports, finding latest info

ONLY return one of these tool names: pubmed, wikipedia, tavily

Topic: {topic}
User Query: {user_message}
"""

    tool_decision = llm.invoke([
        SystemMessage(content=planner_prompt)
    ]).content.strip().lower()

    # Put decision in state for routing
    new_state = state.copy()
    new_state["tool_response"] = tool_decision  # could be used for logging

    # We'll return a special key for routing in LangGraph
    return {
        "__tool__": tool_decision,
        **new_state
    }
