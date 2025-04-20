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

def handle_delete_session(session_id):
    resp = requests.delete(f"{BASE_URL}/sessions/{session_id}")
    if resp.status_code == 204:
        if st.session_state.session_id == session_id:
            # Deleted the currently active session
            # Fetch all sessions again to find another one to switch to
            all_sessions = requests.get(f"{BASE_URL}/sessions/{st.session_state.user_id}")
            if all_sessions.ok and all_sessions.json():
                latest = all_sessions.json()[0]  # Switch to the latest remaining session
                st.session_state.session_id = latest["id"]
                messages = api_get_messages(latest["id"])
                st.session_state.chat_history = [(m["role"], m["content"], m["id"]) for m in messages]
            else:
                # No sessions left, start a new one instead of logging out
                new_session = api_create_session(st.session_state.user_id)
                if new_session:
                    st.session_state.session_id = new_session["id"]
                    st.session_state.chat_history = []
                    st.session_state.feedback_given = {}
        st.rerun()
    else:
        st.error("Failed to delete session.")

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

def api_feedback( message_id: int, thumbs_up: bool):
    resp = requests.post(
        f"{BASE_URL}/feedback/",
        json={"message_id": message_id, "thumbs_up": thumbs_up}
    )
    return handle_api_error(resp, "Submit feedback")

def api_logout(session_id: str):
    resp = requests.post(f"{BASE_URL}/logout/", json={"session_id": session_id})
    return handle_api_error(resp, "Log out")


# ─── Streamlit App ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Fitness Chatbot", layout="centered")
st.markdown("""
<div style="display: flex; align-items: center; gap: 10px;">
  <h1 style="margin: 0;">AI-thlete.</h1>
  <span style="font-size: 18px; color: gray; margin-top: 10px; margin-left: -20px;">Prompting progress, one rep at a time.</span>
</div>
""", unsafe_allow_html=True)

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
                        (m["role"], m["content"], m["id"]) for m in api_get_messages(sess["id"])
                    ]
                    st.rerun()

# ── Chat Interface ──────────────────────────────────────────────────────────────

else:
    st.markdown(f"👤 **{st.session_state.username}**  |  Session `{st.session_state.session_id[:8]}…`")

    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = {}  # message_id -> True

    for role, msg, msg_id in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)
            if role == "assistant":
                if msg_id in st.session_state.feedback_given:
                    st.caption("✅ Feedback submitted")
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("👍", key=f"thumb_up_{msg_id}"):
                            api_feedback(st.session_state.session_id, msg_id, True)
                            st.session_state.feedback_given[msg_id] = True
                            st.rerun()
                    with col2:
                        if st.button("👎", key=f"thumb_down_{msg_id}"):
                            api_feedback(st.session_state.session_id, msg_id, False)
                            st.session_state.feedback_given[msg_id] = True
                            st.rerun()

    with st.sidebar:
        st.markdown("### 💬 Your Chats")
        if st.button("➕ New Chat"):
            new_session = api_create_session(st.session_state.user_id)
            if new_session:
                st.session_state.session_id = new_session["id"]
                st.session_state.chat_history = []
                st.session_state.feedback_given = {}
                st.rerun()

        try:
            sessions = requests.get(f"{BASE_URL}/sessions/{st.session_state.user_id}")
            if sessions.ok:
                sessions = sessions.json()
                for s in sessions:
                    messages = api_get_messages(s["id"])
                    first_user_msg = next((m["content"] for m in messages if m["role"] == "user"), "New Chat")
                    preview = " ".join(first_user_msg.split()[:5]) or "New Chat"
                    is_selected = s["id"] == st.session_state.session_id
                    button_style = "background-color:#444;border:none;padding:6px 12px;width:100%;text-align:left;border-radius:5px;"
                    delete_style = "color:red;background:none;border:none;padding:6px;margin-left:6px;"

                    cols = st.columns([0.85, 0.15])
                    with cols[0]:
                        if st.button(f"👉 {preview}" if is_selected else preview, key=f"chat_{s['id']}"):
                            st.session_state.session_id = s["id"]
                            st.session_state.chat_history = [
                                (m["role"], m["content"], m["id"]) for m in messages
                            ]
                            st.session_state.feedback_given = {}
                            st.rerun()
                    with cols[1]:
                        if st.button("🗑️", key=f"del_{s['id']}"):
                            handle_delete_session(s["id"])
                            st.rerun()
        except Exception as e:
            st.sidebar.error("⚠️ Couldn't load your chats.")

    user_input = st.chat_input("Type your message…")
    if user_input:
        st.session_state.chat_history.append(("user", user_input, None))
        with st.chat_message("user"):
            st.write(user_input)

        reply = api_send_message(st.session_state.session_id, user_input)
        if reply is not None:
            messages = api_get_messages(st.session_state.session_id)
            latest = next((m for m in reversed(messages) if m["role"] == "assistant" and m["content"] == reply), None)
            msg_id = latest["id"] if latest else None

            st.session_state.chat_history.append(("assistant", reply, msg_id))
            st.rerun()
            with st.chat_message("assistant"):
                st.write(reply)

    # — Logout & Feedback Flow —
    if st.button("Logout"):
        if api_logout(st.session_state.session_id):
            for k in ("username", "password", "user_id", "session_id", "chat_history", "feedback_given"):
                st.session_state[k] = None
            st.success("Logged out.")
            st.rerun()
