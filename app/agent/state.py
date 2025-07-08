from typing import List, Optional
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class MedAssistState(TypedDict):
    """
    LangGraph state for MedAssist AI.
    This state will be updated at each node as the conversation progresses.
    """
    # The topic or intent detected (e.g., 'symptoms', 'drug_info', etc.)
    topic: Optional[str]

    # List of all messages (chat history) handled by the agent
    messages: Annotated[List[BaseMessage], add_messages]

    # Final response to return to user
    final_response: Optional[str]

    # Optional flags to handle safety
    risk_flag: Optional[bool]          # Set True by evaluator if risky
    safety_score: Optional[float]      # Confidence score from evaluator
    missing_context: Optional[bool]    # If important info is missing

    # tool_response can be added if you want to pass intermediate outputs
    tool_response: Optional[str]
    # Optional uploaded report content
    report_context: Optional[str]

