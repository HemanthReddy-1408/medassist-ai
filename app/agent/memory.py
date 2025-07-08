from langgraph.checkpoint.memory import MemorySaver

# Create a single memory object (can be shared across LangGraph runs)
memory = MemorySaver()
