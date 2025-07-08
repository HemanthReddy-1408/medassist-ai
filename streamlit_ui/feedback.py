import requests
import streamlit as st

def submit_feedback():
    st.title("❤️ Submit Feedback")
    
    st.markdown("""
    We value your feedback! Please let us know about your experience with MedAssist AI.
    """)

    # General feedback form
    rating = st.radio(
        "How would you rate your overall experience?",
        ["Excellent", "Good", "Average", "Poor"],
        key="overall_rating"
    )
    
    feedback_type = st.selectbox(
        "What type of feedback would you like to provide?",
        ["General Feedback", "Bug Report", "Feature Request", "Other"]
    )
    
    comment = st.text_area(
        "Please share your detailed feedback:",
        height=150
    )
    
    if st.button("Submit Feedback"):
        if comment.strip():
            # ✅ Map to backend-compatible values
            mapped_rating = "like" if rating in ["Excellent", "Good"] else "dislike"

            feedback_payload = {
                "thread_id": st.session_state.thread_id,
                "message": f"General feedback: {feedback_type}",
                "rating": mapped_rating,
                "comment": comment,
            }

            headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
            
            try:
                fb_res = requests.post("http://localhost:8000/api/feedback", json=feedback_payload, headers=headers)
                fb_res.raise_for_status()
                st.success("✅ Thank you for your feedback!")
            except Exception as e:
                st.error(f"❌ Error submitting feedback: {e}")
        else:
            st.warning("Please enter some feedback before submitting.")

# Legacy version, used only in unit testing
def submit_feedback_old(api_url: str, token: str, thread_id: str, message: str, rating: str, comment: str = None):
    try:
        response = requests.post(
            f"{api_url}/api/feedback",
            json={
                "thread_id": thread_id,
                "message": message,
                "rating": rating,
                "comment": comment
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        if response.status_code == 200:
            st.success("✅ Feedback submitted!")
        else:
            st.error(f"❌ Feedback failed: {response.json().get('detail')}")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
