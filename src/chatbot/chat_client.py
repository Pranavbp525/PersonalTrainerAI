# chatbot_client.py

import os
import json
import requests
import streamlit as st

# ─── Configuration ──────────────────────────────────────────────────────────────

BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")

# ─── API Helpers ────────────────────────────────────────────────────────────────

def handle_api_error(resp, action="API call"):
    """Show one‐line error and swallow the exception."""
    if not resp.ok:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        st.error(f"{action} failed: {resp.status_code} – {detail}")
        return False
    return True

def api_get_user(username: str):
    resp = requests.get(f"{BASE_URL}/users/username/{username}")
    if not handle_api_error(resp, "Get user"):
        return None
    return resp.json()

def api_signup(username: str, password: str):
    resp = requests.post(f"{BASE_URL}/signup/", json={"username": username, "password": password})
    if not handle_api_error(resp, "Sign up"):
        return None
    return resp.json()

def api_login(username: str, password: str):
    resp = requests.post(f"{BASE_URL}/login/", json={"username": username, "password": password})
    if resp.status_code == 401:
        st.warning("Invalid username or password.")
        return None
    if not handle_api_error(resp, "Log in"):
        return None
    return resp.json()

def api_create_session(user_id: int):
    resp = requests.post(f"{BASE_URL}/sessions/", json={"user_id": user_id})
    if not handle_api_error(resp, "Create session"):
        return None
    return resp.json()

def api_get_messages(session_id: str):
    resp = requests.get(f"{BASE_URL}/messages/{session_id}")
    if resp.status_code == 404:
        return []
    if not handle_api_error(resp, "Load messages"):
        return []
    return resp.json()

def api_send_message(session_id: str, content: str):
    resp = requests.post(
        f"{BASE_URL}/messages/",
        json={"session_id": session_id, "role": "user", "content": content}
    )
    if not handle_api_error(resp, "Send message"):
        return None
    return resp.json().get("content")

def api_feedback(session_id: str, thumbs_up: bool):
    resp = requests.post(
        f"{BASE_URL}/feedback/",
        json={"session_id": session_id, "thumbs_up": thumbs_up}
    )
    return handle_api_error(resp, "Submit feedback")

def api_logout(session_id: str):
    resp = requests.post(f"{BASE_URL}/logout/", json={"session_id": session_id})
    return handle_api_error(resp, "Log out")

# ─── Streamlit App ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Fitness Chatbot", layout="centered")
st.title("🦾 FIT BOT 💪")

# ── Session State Initialization ────────────────────────────────────────────────

for k in ("username","password","user_id","session_id","chat_history","logging_out"):
    if k not in st.session_state:
        st.session_state[k] = None

if st.session_state.chat_history is None:
    st.session_state.chat_history = []

# ── Authentication (Signup / Login) ─────────────────────────────────────────────

if not st.session_state.session_id:
    st.subheader("Welcome! Log in or Sign up to start chatting.")
    username = st.text_input("Username", key="ui_username", label_visibility="collapsed")
    password = st.text_input("Password", type="password", key="ui_password", label_visibility="collapsed")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Sign Up"):
            if username and password:
                if api_signup(username, password):
                    st.success("Account created! Please log in.")
            else:
                st.warning("Enter a username and password.")
    with col2:
        if st.button("Log In"):
            if not (username and password):
                st.warning("Enter a username and password.")
            else:
                sess = api_login(username, password)
                if sess:
                    st.session_state.username   = username
                    st.session_state.user_id    = sess["user_id"]
                    st.session_state.session_id = sess["id"]
                    st.session_state.chat_history = [
                        (m["role"], m["content"]) for m in api_get_messages(sess["id"])
                    ]
                    st.rerun()

# ── Chat Interface ──────────────────────────────────────────────────────────────

else:
    st.markdown(f"👤 **{st.session_state.username}**  |  Session `{st.session_state.session_id[:8]}…`")

    # Display history
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(msg)

    # New message input
    user_input = st.chat_input("Type your message…")
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        with st.chat_message("user"):
            st.write(user_input)

        reply = api_send_message(st.session_state.session_id, user_input)
        if reply is not None:
            st.session_state.chat_history.append(("assistant", reply))
            with st.chat_message("assistant"):
                st.write(reply)

    # — Logout & Feedback Flow —
    if not st.session_state.logging_out:
        if st.button("Logout"):
            usr = api_get_user(st.session_state.username)
            if usr and usr.get("thumbs_up") is not None:
                # already given feedback
                if api_logout(st.session_state.session_id):
                    for k in ("username","password","user_id","session_id","chat_history","logging_out"):
                        st.session_state[k] = None
                    st.success("Logged out.")
            else:
                st.session_state.logging_out = True
            st.rerun()
    else:
        st.info("Before you go, how was your experience?")
        choice = st.radio("", ("👍","👎"))
        if st.button("Submit & Logout"):
            thumbs = (choice == "👍")
            if api_feedback(st.session_state.session_id, thumbs) and \
               api_logout(st.session_state.session_id):
                for k in ("username","password","user_id","session_id","chat_history","logging_out"):
                    st.session_state[k] = None
                st.success("Thanks for your feedback! You’ve been logged out.")
            st.rerun()
