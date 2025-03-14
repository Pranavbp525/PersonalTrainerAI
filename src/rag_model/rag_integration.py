"""
Integration module for connecting RAG components with existing data pipeline

This module provides functionality to integrate the RAG implementation
with the existing data pipeline, Pinecone vector database, and Airflow orchestration.
"""

import os
import logging
import sys
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import pinecone

# Add parent directory to path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import from existing data pipeline
from src.data_pipeline.vector_db import chunk_to_db

# Import RAG implementations
from rag_model.naive_rag import NaiveRAG
from rag_model.advanced_rag import AdvancedRAG
from rag_model.modular_rag import ModularRAG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "personal-trainer-ai")

class RAGIntegration:
    """
    Integration class for connecting RAG components with the existing data pipeline.
    """
    
    def __init__(self, rag_implementation: str = "modular"):
        """
        Initialize the RAG integration.
        
        Args:
            rag_implementation: Name of the RAG implementation to use (naive, advanced, or modular)
        """
        self.rag_implementation = rag_implementation
        
        # Initialize Pinecone
        logger.info("Connecting to Pinecone")
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        
        # Check if index exists
        if INDEX_NAME not in pinecone.list_indexes().names():
            logger.warning(f"Index {INDEX_NAME} not found in Pinecone")
            logger.info("Please run the data pipeline first to create the index")
        
        # Initialize RAG implementation
        if rag_implementation == "naive":
            self.rag = NaiveRAG()
        elif rag_implementation == "advanced":
            self.rag = AdvancedRAG()
        elif rag_implementation == "modular":
            self.rag = ModularRAG()
        else:
            logger.error(f"Unknown RAG implementation: {rag_implementation}")
            raise ValueError(f"Unknown RAG implementation: {rag_implementation}")
        
        logger.info(f"Initialized {rag_implementation} RAG implementation")
    
    def update_vector_database(self):
        """
        Update the vector database with the latest data from the data pipeline.
        """
        logger.info("Updating vector database")
        
        try:
            # Run the data pipeline to update the vector database
            chunk_to_db()
            logger.info("Vector database updated successfully")
        except Exception as e:
            logger.error(f"Error updating vector database: {e}")
            raise
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query using the selected RAG implementation.
        
        Args:
            query: User query
            
        Returns:
            Dictionary containing the query results
        """
        logger.info(f"Processing query with {self.rag_implementation} RAG: {query}")
        
        try:
            # Process query with RAG implementation
            result = self.rag.query(query)
            return result
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "query": query,
                "error": str(e)
            }
    
    def get_implementation_info(self) -> Dict[str, Any]:
        """
        Get information about the current RAG implementation.
        
        Returns:
            Dictionary containing implementation information
        """
        return {
            "implementation": self.rag_implementation,
            "index_name": INDEX_NAME,
            "pinecone_environment": PINECONE_ENVIRONMENT
        }


# Example usage
if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG Integration for PersonalTrainerAI')
    parser.add_argument('--implementation', type=str, default='modular', 
                        choices=['naive', 'advanced', 'modular'],
                        help='RAG implementation to use')
    parser.add_argument('--update-db', action='store_true',
                        help='Update vector database before processing queries')
    parser.add_argument('--query', type=str,
                        help='Query to process (if not provided, interactive mode is used)')
    
    args = parser.parse_args()
    
    # Initialize RAG integration
    integration = RAGIntegration(rag_implementation=args.implementation)
    
    # Update vector database if requested
    if args.update_db:
        integration.update_vector_database()
    
    # Process query if provided
    if args.query:
        result = integration.process_query(args.query)
        print(f"\nQuery: {result['query']}")
        print(f"\nResponse: {result['response']}")
    else:
        # Interactive mode
        print(f"\nPersonalTrainerAI RAG ({args.implementation.capitalize()} Implementation)")
        print("Type 'exit' to quit")
        
        while True:
            query = input("\nEnter your fitness question: ")
            
            if query.lower() in ['exit', 'quit', 'q']:
                break
            
            result = integration.process_query(query)
            
            if 'error' in result:
                print(f"\nError: {result['error']}")
            else:
                print(f"\nResponse: {result['response']}")
