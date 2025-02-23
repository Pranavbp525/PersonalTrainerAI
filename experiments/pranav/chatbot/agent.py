from typing import Annotated
import logging
from langchain_openai import ChatOpenAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.tools import tool
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
from workout_models import WorkoutRoutine
import os
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class AgentState(TypedDict):
    messages: Annotated[list[dict], add_messages]
    session_id: str
    tool_results: dict

    
pc = Pinecone(api_key = os.environ.get("PINECONE_API_KEY"))
index_name = "fitness-chatbot"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,  # Matches your embedding dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

@tool
def retrieve_exercise_info(query: str) -> str:
    """Retrieves science-based exercise information from Pinecone vector store."""
    log.info(f"RAG query: {query}")
    query_embedding = embeddings.embed_query(query)
    log.info(f"Generated query embedding: {query_embedding[:5]}... (truncated)")
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    log.info(f"Pinecone query results: {results}")
    retrieved_docs = [match["metadata"].get("text", "No text available") for match in results["matches"]]
    log.info(f"Retrieved documents: {retrieved_docs}")
    return "\n".join(retrieved_docs)

tools = [retrieve_exercise_info]
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY"))
llm_with_tools = llm.bind_tools(tools)

workout_llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY"))
workout_system_prompt = """
You are a fitness expert tasked with creating a workout routine based on provided exercise data.
Given the user's goal and retrieved exercise information, generate a workout routine based on the data provided.
Use the retrieved data to inform your choices. Ensure the routine is practical and science-based.
If no relevant data is provided, use general fitness knowledge.
"""

workout_llm_structured = workout_llm.with_structured_output(WorkoutRoutine, method="json_mode")

def chatbot(state: AgentState) -> dict:
    log.info(f"Entering chatbot node with state: {state}")
    response = llm_with_tools.invoke(state["messages"])
    log.info(f"Chatbot LLM response: {response}")
    return {"messages": [response]}

def workout_node(state: AgentState) -> dict:
    log.info(f"Entering workout_node with state: {state}")
    try:
        tool_result = next(
            (msg.content for msg in reversed(state["messages"]) if hasattr(msg, "type") and msg.type == "tool"),
            "No exercise data available."
        )
        log.info(f"Extracted tool result: {tool_result}")
        last_user_message = state["messages"][-1]["content"]
        log.info(f"Last user message: {last_user_message}")

        prompt = f"{workout_system_prompt}\n\nUser goal: {last_user_message}\nRetrieved exercise data:\n{tool_result}"
        log.info(f"Workout LLM prompt: {prompt}")
        workout = workout_llm_structured.invoke([{"role": "system", "content": prompt}, {"role": "user", "content": "Generate the workout routine."}])
        log.info(f"Workout LLM response: {workout}")
        
        workout_json = workout.json()
        log.info(f"Workout JSON: {workout_json}")
        return {"messages": [{"role": "assistant", "content": workout_json}]}
    except Exception as e:
        log.error(f"Error in workout_node: {type(e).__name__}: {str(e)}")
        return {"messages": [{"role": "assistant", "content": f"Error generating workout: {str(e)}"}]}
    
graph_builder = StateGraph(AgentState)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.add_node("workout", workout_node)

graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "workout")
graph_builder.add_edge("workout", "chatbot")
graph_builder.set_entry_point("chatbot")

agent_app = graph_builder.compile()