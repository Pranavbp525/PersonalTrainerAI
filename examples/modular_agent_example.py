"""
Modular Agent Example

This script demonstrates how to use the modular agent architecture
with different LLM providers.
"""

import os
import asyncio
from dotenv import load_dotenv

from src.chatbot.agent import process_message

# Load environment variables from .env file
load_dotenv()


async def main():
    """Run the example."""
    print("Personal Trainer AI - Modular Agent Example")
    print("-------------------------------------------")
    
    # Choose a provider
    provider = input("Choose an LLM provider (openai, gemini, ollama, deepseek, grow) [default: openai]: ").strip().lower() or "openai"
    
    # Set up API keys based on provider
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            api_key = input("Enter your OpenAI API key: ").strip()
            os.environ["OPENAI_API_KEY"] = api_key
    elif provider == "gemini":
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            api_key = input("Enter your Google API key: ").strip()
            os.environ["GOOGLE_API_KEY"] = api_key
    elif provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            api_key = input("Enter your DeepSeek API key: ").strip()
            os.environ["DEEPSEEK_API_KEY"] = api_key
    elif provider == "grow":
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            api_key = input("Enter your Groq API key: ").strip()
            os.environ["GROQ_API_KEY"] = api_key
    elif provider == "ollama":
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        print(f"Using Ollama at {base_url}")
    
    # Initialize state
    state = None
    
    # Main conversation loop
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        # Exit if user types 'exit' or 'quit'
        if user_input.lower() in ["exit", "quit"]:
            break
        
        # Process the message
        try:
            state = await process_message(user_input, state, provider_name=provider)
            
            # Extract and print the assistant's response
            messages = state.get("messages", [])
            if messages:
                assistant_message = messages[-1].content
                print(f"\nAssistant: {assistant_message}")
        except Exception as e:
            print(f"\nError: {e}")
    
    print("\nThank you for using Personal Trainer AI!")


if __name__ == "__main__":
    asyncio.run(main())