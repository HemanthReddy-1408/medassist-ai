from typing import Dict
from app.agent.state import MedAssistState
from langchain_core.messages import HumanMessage
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
import os
from langchain_community.tools.tavily_search import TavilySearchResults

# ✅ Use env variable for Tavily key
tavily_tool = TavilySearchResults(
    k=3,
    tavily_api_key=os.getenv("TAVILY_API_KEY")
)

# === Setup Wikipedia Tool ===
wiki_search = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

def wikipedia_node(state: MedAssistState) -> Dict:
    query = get_user_query(state)
    response = wiki_search.run(query)

    new_state = state.copy()
    new_state["tool_response"] = response
    return new_state


# === Setup Tavily Tool ===
tavily_tool = TavilySearchResults(k=3)  # You can control # of results

def tavily_node(state: MedAssistState) -> Dict:
    query = get_user_query(state)
    response = tavily_tool.run(query)

    new_state = state.copy()
    new_state["tool_response"] = response
    return new_state


# === Setup PubMed Tool (for now: simple search placeholder) ===
# Later we’ll build RAG pipeline using datasets or scraping
def pubmed_node(state: MedAssistState) -> Dict:
    query = get_user_query(state)

    # Placeholder response
    response = f"(PubMed Placeholder) You asked about: {query}. Clinical data lookup not implemented yet."

    new_state = state.copy()
    new_state["tool_response"] = response
    return new_state


# === Helper to get latest user query ===
def get_user_query(state: MedAssistState) -> str:
    return next((msg.content for msg in reversed(state["messages"])
                 if msg.type == "human"), "")
