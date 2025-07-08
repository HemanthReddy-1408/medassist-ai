# streamlit_ui/history.py
import streamlit as st
import requests

BASE_URL = "http://localhost:8000/api"

def get_auth_headers():
    token = st.session_state.get("access_token")
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}

def view_history():
    st.title("📜 Your Chat History")

    thread_id = st.session_state.get("thread_id")
    if not thread_id:
        st.warning("⚠️ You are not logged in or no thread found.")
        return

    headers = get_auth_headers()
    try:
        res = requests.get(f"{BASE_URL}/history/{thread_id}", headers=headers)
        res.raise_for_status()
        history = res.json().get("history", [])
        if not history:
            st.info("No past interactions found.")
            return

        for item in history:
            st.markdown("---")
            st.markdown(f"🧑‍💬 **You**: {item['user_input']}")
            st.markdown(f"🤖 **MedAssist**: {item['agent_response']}")
            if item.get("risk_flag"):
                st.markdown("⚠️ _Risk Detected in Response_")
    except Exception as e:
        st.error(f"Failed to load history: {e}")