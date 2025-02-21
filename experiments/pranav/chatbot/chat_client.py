# chatbot_client.py
import requests
import json

BASE_URL = "http://localhost:8000"  # Or your deployment URL


def create_user(username: str):
    """Creates a new user."""
    data = {"username": username}
    print(f"Sending data: {data}")  # Debugging print
    response = requests.post(f"{BASE_URL}/users/", json=data)  # Use 'json' instead of 'data'
    response.raise_for_status()
    return response.json()


def create_session(user_id: int):
    """Creates a new session for the given user."""
    response = requests.post(f"{BASE_URL}/sessions/", json={"user_id": user_id})
    response.raise_for_status()
    return response.json()


def send_message(session_id: str, message: str):
    """Sends a message to the chatbot and returns the assistant's response."""
    response = requests.post(
        f"{BASE_URL}/messages/",
        json={"session_id": session_id, "role": "user", "content": message},
    )
    response.raise_for_status()
    return response.json()["content"]  # Extract the 'content' from the response


def run_chat(username: str):
    """Runs the interactive chat loop."""
    print("Creating user...")
    user_data = create_user(username)
    user_id = user_data["user_id"]
    print(f"User created: {user_data}")

    print("Creating session...")
    session_data = create_session(user_id)
    session_id = session_data["session_id"]
    print(f"Session created: {session_data}")

    print("\n--- Chat started! Type 'quit' to exit. ---")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        try:
            assistant_response = send_message(session_id, user_input)
            print(f"Chatbot: {assistant_response}")
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with the chatbot API: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

def get_messages(session_id: str):
    response = requests.get(
        f"{BASE_URL}/messages/{session_id}",
    )
    response.raise_for_status()
    return response.json()

if __name__ == "__main__":
    username = input("Enter a username: ")
    run_chat(username)
    # print(get_messages("937e8569-1d64-4f6a-8915-568554f2336a")) # Example to test messages. You will need to change session ID