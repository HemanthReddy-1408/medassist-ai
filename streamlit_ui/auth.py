# streamlit_ui/auth.py
import streamlit as st
import requests

API_URL = "http://localhost:8000/auth"  # ⬅️ Changed from /api to /auth

def show_signup():
    st.title("👩‍⚕️ MedAssist – Signup")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    name = st.text_input("Full Name")

    if st.button("Create Account"):
        response = requests.post(f"{API_URL}/register", json={
            "email": email, "password": password, "name": name
        })
        if response.status_code == 200:
            st.success("✅ Account created! Please login.")
            st.session_state.page = "login"
        else:
            st.error("❌ Signup failed.")

def show_login():
    st.title("🔐 MedAssist – Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        response = requests.post(f"{API_URL}/token", data={
            "username": email,
            "password": password,
            "grant_type": "password"
        }, headers={"Content-Type": "application/x-www-form-urlencoded"})

        if response.status_code == 200:
            token = response.json()["access_token"]
            st.session_state["access_token"] = token
            st.session_state["thread_id"] = email
            st.success("✅ Logged in!")
            st.session_state.page = "chat"
            st.rerun()
        else:
            st.error("❌ Invalid credentials.")
