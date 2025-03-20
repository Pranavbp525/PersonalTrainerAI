"""
RAG Integration Module for PersonalTrainerAI

This module provides a unified interface for using different RAG implementations.
"""

import os
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Import RAG implementations
from .naive_rag import NaiveRAG
from .advanced_rag import AdvancedRAG
from .modular_rag import ModularRAG
from .graph_rag import GraphRAG
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
            config: Optional configuration parameters for the RAG implementation
        """
        logger.info(f"Initializing RAG integration with {implementation} implementation")
        
        self.implementation = implementation
        self.config = config or {}
        
        # Initialize the selected RAG implementation
        if implementation == "naive":
            self.rag = NaiveRAG(**self.config)
        elif implementation == "advanced":
            self.rag = AdvancedRAG(**self.config)
        elif implementation == "modular":
            self.rag = ModularRAG(**self.config)
        elif implementation == "graph":
            self.rag = GraphRAG(**self.config)
        elif implementation == "raptor":
            self.rag = RaptorRAG(**self.config)
        else:
            logger.warning(f"Unknown implementation: {implementation}, defaulting to advanced")
            self.implementation = "advanced"
            self.rag = AdvancedRAG(**self.config)
    
    def process_query(self, query: str) -> str:
        """
        Process a query using the selected RAG implementation.
        
        Args:
            query: The query string
            
        Returns:
            Generated answer
        """
        logger.info(f"Processing query with {self.implementation} RAG: {query[:50]}...")
        return self.rag.answer_question(query)
    
    def get_implementation_info(self) -> Dict[str, Any]:
        """
        Get information about the current RAG implementation.
        
        Returns:
            Dictionary with implementation details
        """
        info = {
            "name": self.implementation,
            "config": self.config
        }
        
        # Add implementation-specific information
        if self.implementation == "naive":
            info["description"] = "Basic RAG with simple vector similarity search"
        elif self.implementation == "advanced":
            info["description"] = "Advanced RAG with reranking and query expansion"
        elif self.implementation == "modular":
            info["description"] = "Modular RAG with query classification and specialized retrievers"
        elif self.implementation == "graph":
            info["description"] = "Graph RAG using knowledge graph structure"
        elif self.implementation == "raptor":
            info["description"] = "RAPTOR RAG with multi-step reasoning and self-reflection"
        
        return info


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Integration for PersonalTrainerAI")
    parser.add_argument("--implementation", type=str, default="advanced", 
                        choices=["naive", "advanced", "modular", "graph", "raptor"],
                        help="RAG implementation to use")
    parser.add_argument("--query", type=str, help="Query to process")
    args = parser.parse_args()
    
    # Initialize integration
    integration = RAGIntegration(implementation=args.implementation)
    
    # Process query if provided
    if args.query:
        answer = integration.process_query(args.query)
        print(f"Query: {args.query}")
        print(f"Answer: {answer}")
    else:
        # Interactive mode
        print(f"Using {integration.implementation} RAG implementation")
        print("Enter 'quit' to exit")
        
        while True:
            query = input("\nEnter your fitness question: ")
            if query.lower() in ["quit", "exit", "q"]:
                break
                
            answer = integration.process_query(query)
            print(f"\nAnswer: {answer}")
