from langgraph.graph import StateGraph, END
from app.agent.state import MedAssistState
from app.agent.memory import memory

# Node functions
from app.agent.nodes.orchestrator import orchestrator_node
from app.agent.nodes.planner import react_planner_node
from app.agent.nodes.tools import pubmed_node, wikipedia_node, tavily_node
from app.agent.nodes.evaluator import evaluator_node
from app.agent.nodes.hitl import hitl_node
from app.agent.nodes.finalizer import final_response_node
from IPython.display import Image, display

def build_medassist_graph():
    # Create graph with state schema
    builder = StateGraph(MedAssistState)

    # Register nodes
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("planner", react_planner_node)
    builder.add_node("pubmed_tool", pubmed_node)
    builder.add_node("wikipedia_tool", wikipedia_node)
    builder.add_node("tavily_tool", tavily_node)
    builder.add_node("evaluator", evaluator_node)
    builder.add_node("hitl", hitl_node)
    builder.add_node("final_response", final_response_node)

    # Entry → Orchestrator
    builder.set_entry_point("orchestrator")
    builder.set_finish_point("final_response")

    # Static edges
    builder.add_edge("orchestrator", "planner")

    # 🧠 Planner conditional tool routing
    builder.add_conditional_edges(
        "planner",
        lambda state: state.get("__tool__", "pubmed"),
        {
            "pubmed": "pubmed_tool",
            "wikipedia": "wikipedia_tool",
            "tavily": "tavily_tool"
        }
    )

    # Tool nodes → Evaluator
    builder.add_edge("pubmed_tool", "evaluator")
    builder.add_edge("wikipedia_tool", "evaluator")
    builder.add_edge("tavily_tool", "evaluator")

    # Evaluator → HITL or Final
    builder.add_conditional_edges(
        "evaluator",
        lambda state: "hitl" if state.get("risk_flag") else "final_response",
        {
            "hitl": "hitl",
            "final_response": "final_response"
        }
    )

    # HITL → Final
    builder.add_edge("hitl", "final_response")

    # Compile with memory for thread persistence
    graph = builder.compile(checkpointer=memory)
    display(Image(graph.get_graph().draw_mermaid_png()))
    return graph
