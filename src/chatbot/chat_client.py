# chatbot_client.py
import requests
import json
import streamlit as st
import os
import time # Import time for potential delays/retries if needed

# Use environment variable or fallback for BASE_URL
BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")

# --- API Call Functions ---

def handle_api_error(response, operation_name="API Call"):
    """Helper to raise exceptions for bad responses."""
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        status_code = err.response.status_code
        try:
            # Try to get detail from JSON response
            detail = err.response.json().get("detail", err.response.text)
        except json.JSONDecodeError:
            detail = err.response.text # Fallback to raw text
        st.error(f"Error during {operation_name}: {status_code} - {detail}")
        # Re-raise the original exception to stop the flow if needed
        raise err


def get_user_by_username(username: str):
    """Gets a user by username. Returns user data dict or None if not found."""
    print(f"Attempting to get user: {username}")
    try:
        response = requests.get(f"{BASE_URL}/users/username/{username}")
        response.raise_for_status() # Raise exception for non-2xx status codes
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"User '{username}' not found (404).")
            return None # Explicitly return None for 404
        else:
            # Handle other HTTP errors (like 500)
            handle_api_error(e.response, f"getting user '{username}'")
            return None # Should not be reached if handle_api_error raises
    except requests.exceptions.RequestException as e:
        st.error(f"Network error checking user: {e}")
        raise e # Re-raise network errors


def create_user(username: str):
    """Creates a new user."""
    data = {"username": username}
    print(f"Sending data to create user: {data}")
    response = requests.post(f"{BASE_URL}/users/", json=data)
    # Use handle_api_error which checks for non-2xx responses
    handle_api_error(response, "creating user")
    return response.json()


def get_latest_session(user_id: int):
    """Gets the latest session for a user. Returns session data dict or None."""
    print(f"Attempting to get latest session for user_id: {user_id}")
    try:
        response = requests.get(f"{BASE_URL}/users/{user_id}/sessions/latest")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"No sessions found for user_id {user_id} (404).")
            return None # Explicitly return None for 404
        else:
            handle_api_error(e.response, f"getting latest session for user {user_id}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network error getting latest session: {e}")
        raise e


def create_session(user_id: int):
    """Creates a new session for the given user."""
    print(f"Attempting to create session for user_id: {user_id}")
    response = requests.post(f"{BASE_URL}/sessions/", json={"user_id": user_id})
    handle_api_error(response, "creating session")
    return response.json()


def send_message(session_id: str, message: str):
    """Sends a message to the chatbot and returns the assistant's response content."""
    print(f"Sending message to session: {session_id[:8]}...") # Log truncated session ID
    response = requests.post(
        f"{BASE_URL}/messages/",
        json={"session_id": session_id, "role": "user", "content": message},
    )
    handle_api_error(response, "sending message")
    return response.json()["content"]


def get_messages(session_id: str):
    """Gets all messages for a given session. Returns list of message dicts or empty list."""
    print(f"Attempting to get messages for session: {session_id[:8]}...")
    try:
        response = requests.get(f"{BASE_URL}/messages/{session_id}")
        response.raise_for_status()
        return response.json() # Returns list of messages like [{'role': 'user', 'content': 'hi', ...}]
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"No messages found for session {session_id[:8]} (404).")
            return [] # Return empty list if no messages found
        else:
            handle_api_error(e.response, f"getting messages for session {session_id[:8]}")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"Network error getting messages: {e}")
        raise e


# --- Streamlit App ---

st.set_page_config(page_title="Fitness chatbot", layout="centered")
st.title("🦾 FIT BOT 💪")

# Initialize session state variables
if "username" not in st.session_state:
    st.session_state.username = None
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] # Stores tuples: (role, content)

# --- Login/User Identification Flow ---
if not st.session_state.username:
    username_input = st.text_input("Enter your username to start or continue:", key="username_input")
    if st.button("Start / Load Chat", key="start_button"):
        if username_input:
            try:
                # 1. Check if user exists
                user_data = get_user_by_username(username_input)

                if user_data:
                    # --- Returning User ---
                    st.session_state.username = user_data["username"]
                    st.session_state.user_id = user_data["id"]
                    st.info(f"Welcome back, {st.session_state.username}!")
                    print(f"User {st.session_state.username} (ID: {st.session_state.user_id}) found.")

                    # 2. Get latest session
                    latest_session = get_latest_session(st.session_state.user_id)

                    if latest_session:
                        # --- Load Existing Session ---
                        st.session_state.session_id = latest_session["id"]
                        print(f"Found latest session: {st.session_state.session_id[:8]}")
                        # 3. Fetch history for this session
                        message_history = get_messages(st.session_state.session_id)
                        # Convert API response format to Streamlit state format
                        st.session_state.chat_history = [(msg["role"], msg["content"]) for msg in message_history]
                        st.success(f"Loaded latest chat session ({len(st.session_state.chat_history)} messages).")
                        st.rerun() # Rerun to update the display immediately

                    else:
                        # --- Returning User, No Sessions (Create New) ---
                        print(f"No existing sessions found for user {st.session_state.user_id}. Creating a new one.")
                        st.info("No previous sessions found. Starting a new one.")
                        new_session = create_session(st.session_state.user_id)
                        st.session_state.session_id = new_session["id"]
                        st.session_state.chat_history = []
                        st.success("Started a new chat session.")
                        st.rerun()

                else:
                    # --- New User ---
                    print(f"User '{username_input}' not found. Creating new user.")
                    st.info(f"Creating a new profile for {username_input}...")
                    # 2. Create user
                    new_user = create_user(username_input)
                    st.session_state.username = new_user["username"]
                    st.session_state.user_id = new_user["id"]
                    print(f"New user created: {st.session_state.username} (ID: {st.session_state.user_id})")

                    # 3. Create first session
                    new_session = create_session(st.session_state.user_id)
                    st.session_state.session_id = new_session["id"]
                    st.session_state.chat_history = []
                    print(f"Created first session: {st.session_state.session_id[:8]}")
                    st.success(f"Welcome, {st.session_state.username}! Your chat is ready.")
                    st.rerun()

            except requests.exceptions.RequestException as e:
                # Error already shown by helper functions, maybe add context
                st.error(f"Failed to start/load chat due to a network or API error.")
                # Clear potentially partially set state
                st.session_state.username = None
                st.session_state.user_id = None
                st.session_state.session_id = None
                st.session_state.chat_history = []
            except Exception as e:
                # Catch other unexpected errors
                st.error(f"An unexpected error occurred: {e}")
                log.exception("Unexpected error during chat start/load.") # Use backend logger if available
                # Clear potentially partially set state
                st.session_state.username = None
                st.session_state.user_id = None
                st.session_state.session_id = None
                st.session_state.chat_history = []

        else:
            st.warning("Please enter a username.")

# --- Chat Interface (Displayed only if username/session are set) ---
else:
    st.markdown(f"👤 **{st.session_state.username}** | Session ID: `{st.session_state.session_id[:8]}...`") # Show partial session ID

    # --- Display Chat History ---
    # Ensure chat history is displayed from the state
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    # --- User Input Area ---
    user_input = st.chat_input("Type a message...")

    if user_input:
        # Add user message to state and display it immediately
        st.session_state.chat_history.append(("user", user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        # Send message to backend and get reply
        try:
            with st.spinner("Fit Bot is thinking..."):
                 # Make sure session_id is valid before sending
                 if not st.session_state.session_id:
                     st.error("Session ID is missing. Cannot send message.")
                     # Optionally try to re-establish session or force login
                 else:
                    reply = send_message(st.session_state.session_id, user_input)
                    # Add assistant reply to state and display it
                    st.session_state.chat_history.append(("assistant", reply))
                    with st.chat_message("assistant"):
                        st.markdown(reply)
                    # No need for st.rerun() here, Streamlit handles chat updates automatically

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to send message. Check connection or API status.")
            # Don't add an error message to the chat history, just show the error banner
        except Exception as e:
            st.error(f"An unexpected error occurred while sending the message: {e}")
            # Log this error properly on the client/server side if possible

# --- Optional: Add a way to start a new session ---
if st.session_state.session_id:
    if st.button("Start New Chat Session"):
        try:
            new_session = create_session(st.session_state.user_id)
            st.session_state.session_id = new_session["id"]
            st.session_state.chat_history = []
            st.success("Started a new chat session.")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to start new session: {e}")