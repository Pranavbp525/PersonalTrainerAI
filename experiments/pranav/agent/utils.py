from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from models import (TrainingPrinciples,
                    TrainingApproaches,
                    Citations,
                    BasicRoutine,
                    AdherenceRate,
                    ProgressMetrics,
                    IssuesList,
                    AdjustmentsList,
                    RoutineCreate,
                    RoutineExtract)
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from typing import List, Dict
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "fitness-chatbot"

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,  # Matches embedding dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


llm = ChatOpenAI(model="gpt-4o", api_key=os.environ.get("OPENAI_API_KEY"))


def extract_principles(text: str) -> List[str]:
    """Extract scientific principles from text."""
    extraction_chain = llm.with_structured_output(TrainingPrinciples)
    
    result = extraction_chain.invoke(
        "Extract the scientific fitness principles from the following text. Return a list of specific principles mentioned:\n\n" + text
    )
    
    return result.principles

def extract_approaches(text: str) -> List[Dict]:
    """Extract training approaches from text."""
    extraction_chain = llm.with_structured_output(TrainingApproaches)
    
    result = extraction_chain.invoke(
        "Extract the training approaches with their names and descriptions from the following text:\n\n" + text
    )
    
    return [{"name": approach.name, "description": approach.description} 
            for approach in result.approaches]

def extract_citations(text: str) -> List[Dict]:
    """Extract citations from text."""
    extraction_chain = llm.with_structured_output(Citations)
    
    result = extraction_chain.invoke(
        "Extract the citations and their sources from the following text. For each citation, identify the source (author, influencer, publication) and the main content or claim:\n\n" + text
    )
    
    return [{"source": citation.source, "content": citation.content} 
            for citation in result.citations]

def extract_routine_data(text: str) -> Dict:
    """Extract structured routine data for Hevy API."""
    extraction_chain = llm.with_structured_output(BasicRoutine)
    
    result = extraction_chain.invoke(
        "Extract the basic workout routine information from the following text. Identify the name, description, and list of workout days or sessions:\n\n" + text
    )
    
    return {
        "name": result.name,
        "description": result.description,
        "workouts": result.workouts
    }

def extract_adherence_rate(text: str) -> float:
    """Extract adherence rate from analysis text."""
    extraction_chain = llm.with_structured_output(AdherenceRate)
    
    result = extraction_chain.invoke(
        "Extract the workout adherence rate as a decimal between 0 and 1 from the following analysis text. This should represent the percentage of planned workouts that were completed:\n\n" + text
    )
    
    return result.rate

def extract_progress_metrics(text: str) -> Dict:
    """Extract progress metrics from analysis text."""
    extraction_chain = llm.with_structured_output(ProgressMetrics)
    
    result = extraction_chain.invoke(
        "Extract the progress metrics with their values from the following analysis text. Identify metrics like strength gains, endurance improvements, weight changes, etc. and their numeric values:\n\n" + text
    )
    
    return result.metrics

def extract_issues(text: str) -> List[str]:
    """Extract identified issues from analysis text."""
    extraction_chain = llm.with_structured_output(IssuesList)
    
    result = extraction_chain.invoke(
        "Extract the identified issues or problems from the following workout analysis text. List each distinct issue that needs attention:\n\n" + text
    )
    
    return result.issues

def extract_adjustments(text: str) -> List[Dict]:
    """Extract suggested adjustments from analysis text."""
    extraction_chain = llm.with_structured_output(AdjustmentsList)
    
    result = extraction_chain.invoke(
        "Extract the suggested workout adjustments from the following analysis text. For each adjustment, identify the target (exercise, schedule, etc.) and the specific change recommended:\n\n" + text
    )
    
    return [{"target": adj.target, "change": adj.change} 
            for adj in result.adjustments]


def extract_routine_structure(text: str) -> Dict:
    """Extract detailed routine structure from text for Hevy API."""
    extraction_chain = llm.with_structured_output(RoutineCreate)
    
    result = extraction_chain.invoke("""
    Extract a detailed workout routine structure from the following text, suitable for the Hevy API:
    
    """ + text + """
    
    Create a structured workout routine with:
    - A title for the routine
    - Overall notes or description
    - A list of exercises, each with:
      - Exercise name
      - Exercise ID (use a placeholder if not specified)
      - Exercise type (strength, cardio, etc)
      - Sets with reps, weight, and type
      - Any specific notes for the exercise
    """)

    
    return result.model_dump()

def extract_routine_updates(text: str) -> Dict:
    """Extract routine updates from text for Hevy API."""
    extraction_chain = llm.with_structured_output(RoutineExtract)
    
    result = extraction_chain.invoke("""
    Extract updates to a workout routine from the following text, suitable for the Hevy API:
    
    """ + text + """
    
    Create a structured representation of the updated workout routine with:
    - The updated title for the routine
    - Updated overall notes
    - The list of exercises with their updated details, each with:
      - Exercise name
      - Exercise ID (use a placeholder if not specified)
      - Exercise type (strength, cardio, etc)
      - Updated sets with reps, weight, and type
      - Any updated notes for the exercise
    """)
    
    return {
        "title": result.title,
        "notes": result.notes,
        "exercises": [
            {
                "exercise_name": ex.exercise_name,
                "exercise_id": ex.exercise_id,
                "exercise_type": ex.exercise_type,
                "sets": [
                    {
                        "type": s.type, 
                        "weight": s.weight,
                        "reps": s.reps,
                        "duration_seconds": s.duration_seconds,
                        "distance_meters": s.distance_meters
                    } for s in ex.sets
                ],
                "notes": ex.notes
            } for ex in result.exercises
        ]
    }


async def retrieve_data(query: str) -> str:
    
    """Retrieves science-based exercise information from Pinecone vector store."""
    logger.info(f"RAG query: {query}")
    query_embedding = embeddings.embed_query(query)
    logger.info(f"Generated query embedding: {query_embedding[:5]}... (truncated)")
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    logger.info(f"Pinecone query results: {results}")
    retrieved_docs = [match["metadata"].get("text", "No text available") for match in results["matches"]]
    logger.info(f"Retrieved documents: {retrieved_docs}")
    return "\n".join(retrieved_docs)