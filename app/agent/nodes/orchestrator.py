from typing import Dict
from app.agent.state import MedAssistState
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

# ⬇️ Use your local LLM wrapped in LangChain — Gemma, Mistral, etc.
from app.llm import llm  # We’ll add this file later

# Prompt template to classify intent
intent_prompt = """
You are an intent classifier for a medical assistant.
Classify the user's query into one of the following intents:
- drug_info
- symptom_check
- report_analysis
- general_medical

ONLY respond with the intent name.

Context from uploaded report (if any):
{report_context}

User's message:
{message}
"""

def orchestrator_node(state: MedAssistState) -> Dict:
    # Get latest human message
    last_user_msg = next((msg.content for msg in reversed(state["messages"])
                         if msg.type == "human"), "")

    prompt = intent_prompt.format(
        message=last_user_msg,
        report_context=state.get("report_context", "None")
    )

    # Run prompt through LLM (Gemma / Mistral)
    intent = llm.invoke([SystemMessage(content=prompt)]).content.strip()

    # Add it to state
    new_state = state.copy()
    new_state["topic"] = intent

    return new_state
