"""
Graph RAG Implementation for PersonalTrainerAI

This module implements a Graph RAG approach that uses a knowledge graph structure
to represent relationships between fitness concepts, allowing for more contextual
retrieval by traversing related nodes.

Key features:
1. Knowledge graph construction from fitness documents
2. Graph-based retrieval using node relationships
3. Path-aware context augmentation
4. Relationship-enhanced prompting
5. Multi-hop reasoning for complex queries
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict
import networkx as nx
import numpy as np
from dotenv import load_dotenv
import pinecone

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "personal-trainer-ai")

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class GraphRAG:
    """
    A Graph RAG implementation that uses a knowledge graph structure for enhanced retrieval
    and reasoning over fitness knowledge.
    """
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        llm_model_name: str = "gpt-3.5-turbo",
        top_k: int = 10,
        graph_path: Optional[str] = None,
        build_graph: bool = True
    ):
        """
        Initialize the GraphRAG system.
        
        Args:
            embedding_model_name: Name of the HuggingFace embedding model
            llm_model_name: Name of the LLM model
            top_k: Number of documents to retrieve initially
            graph_path: Path to saved knowledge graph (if None, will build from scratch)
            build_graph: Whether to build the graph on initialization
        """
        self.top_k = top_k
        
        # Initialize embedding model
        logger.info(f"Initializing embedding model: {embedding_model_name}")
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        # Initialize Pinecone
        logger.info("Connecting to Pinecone")
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        self.index = pinecone.Index(INDEX_NAME)
        
        # Initialize LLM
        logger.info(f"Initializing LLM: {llm_model_name}")
        self.llm = OpenAI(model_name=llm_model_name, temperature=0.7, api_key=OPENAI_API_KEY)
        
        # Initialize knowledge graph
        self.graph = nx.DiGraph()
        
        # Load or build knowledge graph
        if graph_path and os.path.exists(graph_path):
            self.load_graph(graph_path)
        elif build_graph:
            self.build_knowledge_graph()
        
        # Define entity extraction prompt
        self.entity_extraction_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Extract fitness-related entities from the following text. 
            For each entity, identify its type (exercise, muscle_group, equipment, nutrient, training_principle).
            Format your response as a JSON array of objects, each with 'entity', 'type', and 'importance' fields.
            Importance should be a number from 1-10 indicating how central the entity is to the text.
            
            Text: {text}
            
            Entities:
            """
        )
        
        # Define relationship extraction prompt
        self.relationship_extraction_prompt = PromptTemplate(
            input_variables=["entities", "text"],
            template="""
            Given the following fitness-related entities and text, identify relationships between the entities.
            Format your response as a JSON array of objects, each with 'source', 'target', 'relationship', and 'strength' fields.
            Relationship should be one of: 'targets', 'requires', 'enhances', 'prevents', 'contains', 'is_type_of'.
            Strength should be a number from 1-10 indicating how strong the relationship is.
            
            Entities: {entities}
            
            Text: {text}
            
            Relationships:
            """
        )
        
        # Define query analysis prompt
        self.query_analysis_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            Analyze this fitness-related query and identify:
            1. The main entities (exercises, muscle groups, equipment, nutrients, training principles)
            2. The relationships being asked about
            3. The type of information being requested (how-to, comparison, explanation, recommendation)
            
            Format your response as a JSON object with 'entities', 'relationships', and 'query_type' fields.
            
            Query: {query}
            
            Analysis:
            """
        )
        
        # Define response generation prompt
        self.response_prompt = PromptTemplate(
            input_variables=["query", "context", "graph_paths"],
            template="""
            You are a knowledgeable personal fitness trainer assistant. Use the following retrieved information and graph relationships to answer the user's question.
            
            Guidelines:
            - Provide a comprehensive and detailed answer
            - Include specific exercises, techniques, or training principles when relevant
            - Explain the scientific reasoning behind your recommendations when possible
            - Use the graph path information to explain relationships between concepts
            - Format your response in a clear, structured way with appropriate headings and bullet points
            
            Retrieved information:
            {context}
            
            Graph relationships:
            {graph_paths}
            
            User question: {query}
            
            Your answer:
            """
        )
        
        # Create chains
        self.entity_extraction_chain = LLMChain(llm=self.llm, prompt=self.entity_extraction_prompt)
        self.relationship_extraction_chain = LLMChain(llm=self.llm, prompt=self.relationship_extraction_prompt)
        self.query_analysis_chain = LLMChain(llm=self.llm, prompt=self.query_analysis_prompt)
        self.response_chain = LLMChain(llm=self.llm, prompt=self.response_prompt)
        
        logger.info("GraphRAG initialized successfully")
    
    def build_knowledge_graph(self) -> None:
        """
        Build a knowledge graph from the documents in the vector database.
        """
        logger.info("Building knowledge graph from vector database")
        
        # Get a sample of documents from the vector database
        # In a real implementation, you would process all documents or use a streaming approach
        sample_query = np.random.rand(768).tolist()  # Random vector for sampling
        results = self.index.query(
            vector=sample_query,
            top_k=100,  # Get a larger sample
            include_metadata=True
        )
        
        # Process each document to extract entities and relationships
        for i, match in enumerate(results['matches']):
            if i % 10 == 0:
                logger.info(f"Processing document {i+1}/100")
            
            metadata = match.get('metadata', {})
            text = metadata.get('text', '')
            
            if not text:
                continue
            
            # Extract entities
            try:
                entity_result = self.entity_extraction_chain.run(text=text)
                entities = json.loads(entity_result)
                
                # Add entities to graph
                for entity in entities:
                    entity_name = entity['entity']
                    entity_type = entity['type']
                    importance = entity['importance']
                    
                    # Add node if it doesn't exist
                    if not self.graph.has_node(entity_name):
                        self.graph.add_node(entity_name, type=entity_type, importance=importance)
                    
                # Extract relationships between entities
                if len(entities) > 1:
                    entity_names = [e['entity'] for e in entities]
                    relationship_result = self.relationship_extraction_chain.run(
                        entities=str(entity_names),
                        text=text
                    )
                    relationships = json.loads(relationship_result)
                    
                    # Add relationships to graph
                    for rel in relationships:
                        source = rel['source']
                        target = rel['target']
                        relationship = rel['relationship']
                        strength = rel['strength']
                        
                        # Add edge if both nodes exist
                        if self.graph.has_node(source) and self.graph.has_node(target):
                            self.graph.add_edge(
                                source, target,
                                relationship=relationship,
                                strength=strength
                            )
            except Exception as e:
                logger.error(f"Error processing document: {e}")
        
        logger.info(f"Knowledge graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    def save_graph(self, path: str) -> None:
        """
        Save the knowledge graph to a file.
        
        Args:
            path: Path to save the graph
        """
        logger.info(f"Saving knowledge graph to {path}")
        
        # Convert graph to dictionary
        graph_data = {
            'nodes': [],
            'edges': []
        }
        
        # Add nodes
        for node, attrs in self.graph.nodes(data=True):
            node_data = {'id': node}
            node_data.update(attrs)
            graph_data['nodes'].append(node_data)
        
        # Add edges
        for source, target, attrs in self.graph.edges(data=True):
            edge_data = {'source': source, 'target': target}
            edge_data.update(attrs)
            graph_data['edges'].append(edge_data)
        
        # Save to file
        with open(path, 'w') as f:
            json.dump(graph_data, f, indent=2)
    
    def load_graph(self, path: str) -> None:
        """
        Load the knowledge graph from a file.
        
        Args:
            path: Path to the saved graph
        """
        logger.info(f"Loading knowledge graph from {path}")
        
        try:
            with open(path, 'r') as f:
                graph_data = json.load(f)
            
            # Create new graph
            self.graph = nx.DiGraph()
            
            # Add nodes
            for node in graph_data['nodes']:
                node_id = node.pop('id')
                self.graph.add_node(node_id, **node)
            
            # Add edges
            for edge in graph_data['edges']:
                source = edge.pop('source')
                target = edge.pop('target')
                self.graph.add_edge(source, target, **edge)
            
            logger.info(f"Loaded knowledge graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        except Exception as e:
            logger.error(f"Error loading knowledge graph: {e}")
            # Initialize empty graph
            self.graph = nx.DiGraph()
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze the query to identify entities, relationships, and query type.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with query analysis
        """
        logger.info(f"Analyzing query: {query}")
        
        try:
            analysis_result = self.query_analysis_chain.run(query=query)
            analysis = json.loads(analysis_result)
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            return {
                'entities': [],
                'relationships': [],
                'query_type': 'unknown'
            }
    
    def retrieve_documents(self, query: str, entities: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve documents based on the query and identified entities.
        
        Args:
            query: User query
            entities: List of entities identified in the query
            
        Returns:
            List of retrieved documents
        """
        logger.info(f"Retrieving documents for query: {query}")
        
        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Search Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=self.top_k,
            include_metadata=True
        )
        
        return results['matches']
    
    def find_graph_paths(self, entities: List[str], max_hops: int = 2) -> List[Dict[str, Any]]:
        """
        Find paths in the knowledge graph connecting the identified entities.
        
        Args:
            entities: List of entities identified in the query
            max_hops: Maximum number of hops between entities
            
        Returns:
            List of paths between entities
        """
        logger.info(f"Finding graph paths for entities: {entities}")
        
        paths = []
        
        # Find entities that exist in the graph
        existing_entities = [e for e in entities if self.graph.has_node(e)]
        
        if len(existing_entities) < 2:
            return paths
        
        # Find paths between all pairs of entities
        for i, source in enumerate(existing_entities):
            for target in existing_entities[i+1:]:
                # Find all simple paths with limited length
                try:
                    simple_paths = list(nx.all_simple_paths(
                        self.graph, source, target, cutoff=max_hops
                    ))
                    
                    # Add reverse paths
                    reverse_paths = list(nx.all_simple_paths(
                        self.graph, target, source, cutoff=max_hops
                    ))
                    
                    # Process and add paths
                    for path in simple_paths + reverse_paths:
                        if len(path) > 1:
                            path_info = {
                                'path': path,
                                'relationships': []
                            }
                            
                            # Add relationship information
                            for j in range(len(path) - 1):
                                source_node = path[j]
                                target_node = path[j + 1]
                                edge_data = self.graph.get_edge_data(source_node, target_node)
                                
                                if edge_data:
                                    relationship = edge_data.get('relationship', 'related_to')
                                    strength = edge_data.get('strength', 5)
                                    
                                  <response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>