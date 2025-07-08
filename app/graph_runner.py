from app.agent.graph import build_medassist_graph
from app.agent.state import MedAssistState
from langchain_core.messages import HumanMessage
import uuid

# Create the graph
graph = build_medassist_graph()

# Optional: thread ID to persist memory per user
thread_id = str(uuid.uuid4())

print("🩺 MedAssist CLI – Ask your medical question (type 'exit' to quit)")

while True:
    user_input = input("\n👤 You: ")
    if user_input.lower() in {"exit", "quit"}:
        break

    # Initial state for LangGraph run
    input_state: MedAssistState = {
        "messages": [HumanMessage(content=user_input)],
        "topic": None,
        "final_response": None,
        "risk_flag": None,
        "safety_score": None,
        "missing_context": None,
        "report_context": None,  # Can pass patient report here if needed
        "tool_response": None
    }

    # Run the graph
    config={"configurable":{"thread_id":thread_id}}
    output = graph.invoke(input=input_state, config=config)
    print(f"\n🤖 MedAssist: {output['final_response']}")
