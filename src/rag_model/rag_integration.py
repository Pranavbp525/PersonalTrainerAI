"""
RAG Integration Module for PersonalTrainerAI

This module provides a unified interface for using different RAG implementations.
"""

import os
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Import RAG implementations
from .advanced_rag import AdvancedRAG
from .modular_rag import ModularRAG
from .raptor_rag import RaptorRAG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class RAGIntegration:
    """
    Integration class for using different RAG implementations with a unified interface.
    """
    
    def __init__(
        self,
        implementation: str = "advanced",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the RAG integration.
        
        Args:
            implementation: Name of the RAG implementation to use
            config: Configuration parameters for the RAG implementation
        """
        logger.info(f"Initializing RAG integration with {implementation} implementation")
        
        # Set default config if not provided
        if config is None:
            config = {}
        
        # Initialize the specified RAG implementation
        if implementation.lower() == "advanced":
            self.rag = AdvancedRAG(**config)
        elif implementation.lower() == "modular":
            self.rag = ModularRAG(**config)
        elif implementation.lower() == "raptor":
            self.rag = RaptorRAG(**config)
        else:
            logger.warning(f"Unknown implementation: {implementation}, defaulting to advanced")
            self.rag = AdvancedRAG(**config)
        
        self.implementation = implementation
        logger.info(f"RAG integration initialized with {implementation} implementation")
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query using the selected RAG implementation.
        
        Args:
            query: The query string
            
        Returns:
            Dictionary containing the response and metadata
        """
        logger.info(f"Processing query with {self.implementation} implementation: {query[:50]}...")
        
        # Get response from RAG implementation
        response = self.rag.answer_question(query)
        
        # Return response with metadata
        return {
            "query": query,
            "response": response,
            "implementation": self.implementation
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG integration for fitness knowledge")
    parser.add_argument("--implementation", type=str, default="advanced", 
                        choices=["advanced", "modular", "raptor"],
                        help="RAG implementation to use")
    parser.add_argument("--query", type=str, help="Query to process")
    args = parser.parse_args()
    
    # Initialize RAG integration
    integration = RAGIntegration(implementation=args.implementation)
    
    # Process query
    if args.query:
        result = integration.process_query(args.query)
        print(f"Query: {result['query']}")
        print(f"Implementation: {result['implementation']}")
        print(f"Response: {result['response']}")
    else:
        # Interactive mode
        print(f"RAG integration with {args.implementation} implementation")
        print("Enter 'quit' to exit")
        
        while True:
            query = input("\nEnter your fitness question: ")
            if query.lower() in ["quit", "exit", "q"]:
                break
                
            result = integration.process_query(query)
            print(f"\nResponse: {result['response']}")
