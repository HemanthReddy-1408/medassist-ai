# streamlit_ui/app.py

import streamlit as st
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from auth import show_login, show_signup
from chat import chat_interface
from history import view_history
from feedback import submit_feedback
from utils import is_authenticated, logout

# Page configuration
st.set_page_config(
    page_title="MedAssist AI",
    page_icon="🩺",
    layout="centered"
)

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "login"

# Sidebar Navigation
with st.sidebar:
    st.title("🩺 MedAssist AI")
    st.markdown("""
        Your intelligent medical assistant.
        Powered by Agentic AI with memory, feedback, and personalized chat history.
    """)

    if is_authenticated():
        st.success(f"✅ Logged in as `{st.session_state.thread_id}`")

        if st.button("💬 Chat with AI"):
            st.session_state.page = "chat"

        if st.button("📜 View History"):
            st.session_state.page = "history"

        if st.button("❤️ Give Feedback"):
            st.session_state.page = "feedback"

        if st.button("🚪 Logout"):
            logout()
            st.session_state.page = "login"
            st.rerun()
    else:
        st.info("🔒 Please login or signup to continue.")

        if st.button("🔐 Login"):
            st.session_state.page = "login"

        if st.button("🆕 Signup"):
            st.session_state.page = "signup"

# Main Content Area
page = st.session_state.page

if page == "login":
    show_login()

elif page == "signup":
    show_signup()

elif page == "chat":
    if is_authenticated():
        chat_interface()
    else:
        st.warning("⚠️ Please login to start chatting with MedAssist.")

elif page == "history":
    if is_authenticated():
        view_history()
    else:
        st.warning("⚠️ Please login to view your conversation history.")

elif page == "feedback":
    if is_authenticated():
        submit_feedback()
    else:
        st.warning("⚠️ Please login to submit feedback.")

else:
    st.error("🚫 404 - Page not found.")