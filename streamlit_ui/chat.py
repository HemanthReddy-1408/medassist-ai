# streamlit_ui/chat.py

import streamlit as st
import requests

def chat_interface():
    st.title("🩺 MedAssist AI - Chat")

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Input for user prompt
    user_input = st.text_input("Your medical query", key="user_input")

    if st.button("Send", key="send_button") and user_input.strip():
        token = st.session_state.get("access_token")
        thread_id = st.session_state.get("thread_id")

        if not token or not thread_id:
            st.error("⚠️ You are not authenticated. Please log in again.")
            st.stop()

        headers = {"Authorization": f"Bearer {token}"}
        payload = {
            "message": user_input,
            "thread_id": thread_id,
        }

        try:
            res = requests.post("http://localhost:8000/api/query", json=payload, headers=headers)
            res.raise_for_status()
            assistant_response = res.json()["response"]
            st.session_state.chat_history.append((user_input, assistant_response))
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error: {e}")

    # ---------- Display chat and feedback ----------
    for i, (user_msg, assistant_msg) in enumerate(st.session_state.chat_history):
        st.markdown(f"**🧑 You:** {user_msg}")
        st.markdown(f"**🤖 MedAssist:** {assistant_msg}")

        with st.expander("📝 Feedback on this response", expanded=False):
            col1, col2 = st.columns([1, 3])

            with col1:
                feedback_rating = st.radio(
                    "Was this helpful?",
                    ["like", "dislike"],
                    key=f"rating_{i}"
                )

            with col2:
                feedback_comment = st.text_input(
                    "Any comments?",
                    key=f"comment_{i}"
                )

            if st.button("Submit Feedback", key=f"submit_feedback_{i}"):
                token = st.session_state.get("access_token")
                thread_id = st.session_state.get("thread_id")

                if not token or not thread_id:
                    st.error("⚠️ You are not logged in.")
                    st.stop()

                headers = {"Authorization": f"Bearer {token}"}
                feedback_payload = {
                    "thread_id": thread_id,
                    "message": assistant_msg,
                    "rating": feedback_rating,
                    "comment": feedback_comment or ""
                }

                try:
                    st.json(feedback_payload)  # Debug: Show payload
                    fb_res = requests.post("http://localhost:8000/api/feedback", json=feedback_payload, headers=headers)
                    fb_res.raise_for_status()
                    st.success("✅ Feedback submitted successfully.")
                except Exception as e:
                    st.error(f"❌ Error submitting feedback: {e}")
