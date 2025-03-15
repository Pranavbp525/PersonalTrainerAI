"""
RAG Integration Module for PersonalTrainerAI

This module provides integration between the different RAG implementations
and the existing PersonalTrainerAI pipeline, now including Graph RAG and RAPTOR RAG.
"""

import os
import logging
import json
import argparse
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Import RAG implementations
from rag_model.naive_rag import NaiveRAG
from rag_model.advanced_rag import AdvancedRAG
from rag_model.modular_rag import ModularRAG
from rag_model.graph_rag import GraphRAG
from rag_model.raptor_rag import RAPTORRAG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class RAGIntegration:
    """
    Integration class for different RAG implementations with PersonalTrainerAI.
    """
    
    def __init__(
        self,
        implementation: str = "modular",
        graph_path: Optional[str] = "fitness_knowledge_graph.json"
    ):
        """
        Initialize the RAG integration.
        
        Args:
            implementation: RAG implementation to use (naive, advanced, modular, graph, raptor)
            graph_path: Path to knowledge graph for Graph RAG
        """
        logger.info(f"Initializing RAG integration with {implementation} implementation")
        
        # Initialize the specified RAG implementation
        if implementation == "naive":
            self.rag = NaiveRAG()
        elif implementation == "advanced":
            self.rag = AdvancedRAG()
        elif implementation == "modular":
            self.rag = ModularRAG()
        elif implementation == "graph":
            self.rag = GraphRAG(graph_path=graph_path, build_graph=not os.path.exists(graph_path))
        elif implementation == "raptor":
            self.rag = RAPTORRAG()
        else:
            logger.warning(f"Unknown implementation: {implementation}, defaulting to modular")
            self.rag = ModularRAG()
        
        self.implementation = implementation
        logger.info(f"{implementation.capitalize()} RAG initialized successfully")
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query using the selected RAG implementation.
        
        Args:
            query: User query
            
        Returns:
            Dictionary containing the query results
        """
        logger.info(f"Processing query with {self.implementation} RAG: {query}")
        return self.rag.query(query)
    
    def process_batch_queries(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Process a batch of user queries.
        
        Args:
            queries: List of user queries
            
        Returns:
            List of dictionaries containing query results
        """
        logger.info(f"Processing batch of {len(queries)} queries with {self.implementation} RAG")
        results = []
        
        for query in queries:
            result = self.process_query(query)
            results.append(result)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_file: str) -> None:
        """
        Save query results to a JSON file.
        
        Args:
            results: List of query results
            output_file: Path to save the results
        """
        logger.info(f"Saving {len(results)} query results to {output_file}")
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved query results to {output_file}")
        except Exception as e:
            logger.error(f"Error saving query results: {e}")
    
    def interactive_mode(self) -> None:
        """
        Run the RAG integration in interactive mode.
        """
        logger.info(f"Starting interactive mode with {self.implementation} RAG")
        print(f"\nPersonalTrainerAI {self.implementation.capitalize()} RAG")
        print("Type 'exit' to quit")
        
        while True:
            query = input("\nEnter your fitness question: ")
            
            if query.lower() in ['exit', 'quit', 'q']:
                break
            
            result = self.process_query(query)
            
            if 'error' in result:
                print(f"\nError: {result['error']}")
            else:
                print(f"\nResponse: {result['response']}")


# Example usage
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='RAG Integration for PersonalTrainerAI')
    parser.add_argument('--implementation', type=str, default='modular',
                        choices=['naive', 'advanced', 'modular', 'graph', 'raptor'],
                        help='RAG implementation to use')
    parser.add_argument('--graph-path', type=str, default='fitness_knowledge_graph.json',
                        help='Path to knowledge graph for Graph RAG')
    parser.add_argument('--query', type=str, help='Query to process')
    parser.add_argument('--batch-file', type=str, help='Path to JSON file containing batch queries')
    parser.add_argument('--output-file', type=str, help='Path to save query results')
    
    args = parser.parse_args()
    
    # Initialize RAG integration
    rag_integration = RAGIntegration(
        implementation=args.implementation,
        graph_path=args.graph_path
    )
    
    # Process batch queries if specified
    if args.batch_file:
        try:
            with open(args.batch_file, 'r') as f:
                queries = json.load(f)
            
            results = rag_integration.process_batch_queries(queries)
            
            if args.output_file:
                rag_integration.save_results(results, args.output_file)
            else:
                print(json.dumps(results, indent=2))
        except Exception as e:
            logger.error(f"Error processing batch queries: {e}")
    
    # Process single query if specified
    elif args.query:
        result = rag_integration.process_query(args.query)
        
        if args.output_file:
            rag_integration.save_results([result], args.output_file)
        else:
            print(json.dumps(result, indent=2))
    
    # Run in interactive mode if no query specified
    else:
        rag_integration.interactive_mode()
