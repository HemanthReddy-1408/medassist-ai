# streamlit_ui/utils.py
import streamlit as st
import requests

BASE_URL = "http://localhost:8000/api"

def save_token(token_data):
    st.session_state["access_token"] = token_data["access_token"]
    st.session_state["token_type"] = token_data["token_type"]

def get_auth_headers():
    token = st.session_state.get("access_token")
    token_type = st.session_state.get("token_type", "bearer")
    if token:
        return {"Authorization": f"{token_type} {token}"}
    return {}

def is_authenticated():
    return "access_token" in st.session_state and "thread_id" in st.session_state

def logout():
    for key in ["access_token", "token_type", "thread_id"]:
        st.session_state.pop(key, None)

def post_with_auth(endpoint: str, json_data: dict):
    headers = get_auth_headers()
    return requests.post(f"{BASE_URL}{endpoint}", json=json_data, headers=headers)
