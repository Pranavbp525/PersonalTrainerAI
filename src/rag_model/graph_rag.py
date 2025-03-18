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

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphRAG:
    """
    Graph-based Retrieval Augmented Generation for fitness knowledge.
    
    This implementation builds a knowledge graph from fitness documents and uses
    graph traversal algorithms to enhance context retrieval for more accurate
    and contextually relevant responses.
    """
    
    def __init__(
        self,
        vector_db_name: str = "fitness-knowledge",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.2,
        max_tokens: int = 500,
        graph_path: Optional[str] = None
    ):
        """
        Initialize the Graph RAG system.
        
        Args:
            vector_db_name: Name of the Pinecone vector database
            embedding_model_name: Name of the embedding model to use
            llm_model_name: Name of the language model to use
            temperature: Temperature parameter for the LLM
            max_tokens: Maximum tokens for LLM response
            graph_path: Path to save/load the knowledge graph
        """
        # Initialize Pinecone
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_env = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
        
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
        
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
        
        # Check if the index exists, if not create it
        if vector_db_name not in pinecone.list_indexes():
            logger.info(f"Creating new Pinecone index: {vector_db_name}")
            pinecone.create_index(
                name=vector_db_name,
                dimension=384,  # Dimension for all-MiniLM-L6-v2
                metric="cosine"
            )
        
        self.index = pinecone.Index(vector_db_name)
        
        # Initialize embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        # Initialize LLM
        self.llm = OpenAI(
            model_name=llm_model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Initialize knowledge graph
        self.graph = nx.DiGraph()
        self.graph_path = graph_path or "fitness_knowledge_graph.gpickle"
        
        # Load existing graph if available
        if os.path.exists(self.graph_path):
            self._load_graph()
        
        # Define entity types and relationships for the fitness domain
        self.entity_types = {
            "exercise": ["name", "muscle_group", "equipment", "difficulty"],
            "workout": ["name", "type", "duration", "intensity", "goal"],
            "nutrition": ["name", "category", "macros", "benefits"],
            "muscle_group": ["name", "location", "function"],
            "equipment": ["name", "type", "difficulty"],
            "fitness_goal": ["name", "timeframe", "metrics"]
        }
        
        self.relationships = [
            ("exercise", "targets", "muscle_group"),
            ("exercise", "requires", "equipment"),
            ("exercise", "contributes_to", "fitness_goal"),
            ("workout", "includes", "exercise"),
            ("workout", "aims_for", "fitness_goal"),
            ("nutrition", "supports", "fitness_goal"),
            ("nutrition", "benefits", "muscle_group")
        ]
        
        # Templates for graph-based prompting
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question", "graph_paths"],
            template="""
            You are a knowledgeable fitness assistant with expertise in workouts, nutrition, and exercise science.
            
            Use the following retrieved information to answer the question. If you don't know the answer, just say that you don't know.
            
            Retrieved information:
            {context}
            
            The following relationships between fitness concepts are relevant:
            {graph_paths}
            
            Question: {question}
            
            Answer:
            """
        )

    def _load_graph(self):
        """Load the knowledge graph from disk."""
        try:
            self.graph = nx.read_gpickle(self.graph_path)
            logger.info(f"Loaded knowledge graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        except Exception as e:
            logger.warning(f"Could not load knowledge graph: {e}")
            self.graph = nx.DiGraph()

    def _save_graph(self):
        """Save the knowledge graph to disk."""
        try:
            nx.write_gpickle(self.graph, self.graph_path)
            logger.info(f"Saved knowledge graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        except Exception as e:
            logger.error(f"Could not save knowledge graph: {e}")

    def build_graph_from_documents(self, documents: List[Dict[str, Any]]):
        """
        Build a knowledge graph from fitness documents.
        
        Args:
            documents: List of document dictionaries with text and metadata
        """
        logger.info(f"Building knowledge graph from {len(documents)} documents")
        
        # Extract entities and relationships from documents
        for doc in documents:
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            
            # Extract entities based on metadata
            doc_type = metadata.get("type", "unknown")
            
            if doc_type in self.entity_types:
                # Create node for the document
                node_id = metadata.get("id", f"{doc_type}_{len(self.graph.nodes)}")
                
                # Add node to graph with document properties
                self.graph.add_node(
                    node_id,
                    type=doc_type,
                    text=text,
                    **{k: metadata.get(k) for k in self.entity_types[doc_type] if k in metadata}
                )
                
                # Connect to related entities based on metadata
                for rel_type, rel_name, target_type in self.relationships:
                    if rel_type == doc_type and f"{rel_name}_ids" in metadata:
                        for target_id in metadata[f"{rel_name}_ids"]:
                            self.graph.add_edge(
                                node_id,
                                target_id,
                                type=rel_name
                            )
        
        # Save the updated graph
        self._save_graph()
        logger.info(f"Knowledge graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")

    def extract_entities_from_text(self, text: str) -> Dict[str, List[str]]:
        """
        Extract fitness-related entities from text using LLM.
        
        Args:
            text: Input text to extract entities from
            
        Returns:
            Dictionary of entity types and their instances
        """
        # Use LLM to extract entities
        prompt = f"""
        Extract fitness-related entities from the following text. 
        Return a JSON object with the following entity types:
        - exercises: List of exercise names
        - muscle_groups: List of muscle groups
        - equipment: List of equipment mentioned
        - workout_types: List of workout types
        - nutrition: List of nutrition concepts
        - fitness_goals: List of fitness goals
        
        Text: {text}
        
        JSON:
        """
        
        try:
            response = self.llm(prompt)
            entities = json.loads(response)
            return entities
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return {
                "exercises": [],
                "muscle_groups": [],
                "equipment": [],
                "workout_types": [],
                "nutrition": [],
                "fitness_goals": []
            }

    def add_document_to_graph(self, document: Dict[str, Any]):
        """
        Add a single document to the knowledge graph.
        
        Args:
            document: Document dictionary with text and metadata
        """
        text = document.get("text", "")
        metadata = document.get("metadata", {})
        doc_id = metadata.get("id", f"doc_{len(self.graph.nodes)}")
        
        # Add document node
        self.graph.add_node(
            doc_id,
            type="document",
            text=text,
            **metadata
        )
        
        # Extract entities
        entities = self.extract_entities_from_text(text)
        
        # Add entity nodes and connect to document
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                # Create a normalized entity ID
                entity_id = f"{entity_type}_{entity.lower().replace(' ', '_')}"
                
                # Add entity node if it doesn't exist
                if not self.graph.has_node(entity_id):
                    self.graph.add_node(
                        entity_id,
                        type=entity_type,
                        name=entity
                    )
                
                # Connect document to entity
                self.graph.add_edge(
                    doc_id,
                    entity_id,
                    type="mentions"
                )
                
                # Connect entities to each other based on domain knowledge
                self._connect_related_entities(entity_id, entity_type, entities)
        
        # Save the updated graph
        self._save_graph()

    def _connect_related_entities(self, entity_id: str, entity_type: str, entities: Dict[str, List[str]]):
        """
        Connect related entities based on domain knowledge.
        
        Args:
            entity_id: ID of the entity to connect
            entity_type: Type of the entity
            entities: Dictionary of extracted entities
        """
        # Map entity types to relationship types
        type_to_rel = {
            "exercises": {
                "muscle_groups": "targets",
                "equipment": "requires",
                "fitness_goals": "contributes_to"
            },
            "workout_types": {
                "exercises": "includes",
                "fitness_goals": "aims_for"
            },
            "nutrition": {
                "fitness_goals": "supports",
                "muscle_groups": "benefits"
            }
        }
        
        # Connect entity to related entities
        if entity_type in type_to_rel:
            for target_type, rel_type in type_to_rel[entity_type].items():
                for target_entity in entities.get(target_type, []):
                    target_id = f"{target_type}_{target_entity.lower().replace(' ', '_')}"
                    
                    # Add target entity if it doesn't exist
                    if not self.graph.has_node(target_id):
                        self.graph.add_node(
                            target_id,
                            type=target_type,
                            name=target_entity
                        )
                    
                    # Connect entities
                    self.graph.add_edge(
                        entity_id,
                        target_id,
                        type=rel_type
                    )

    def find_paths_between_entities(self, source_entities: List[str], target_entities: List[str], max_length: int = 3) -> List[List[str]]:
        """
        Find paths between source and target entities in the knowledge graph.
        
        Args:
            source_entities: List of source entity IDs
            target_entities: List of target entity IDs
            max_length: Maximum path length
            
        Returns:
            List of paths (each path is a list of node IDs)
        """
        all_paths = []
        
        for source in source_entities:
            for target in target_entities:
                try:
                    # Find all simple paths between source and target
                    paths = list(nx.all_simple_paths(
                        self.graph, 
                        source=source, 
                        target=target, 
                        cutoff=max_length
                    ))
                    all_paths.extend(paths)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
        
        return all_paths

    def format_paths_for_prompt(self, paths: List[List[str]]) -> str:
        """
        Format graph paths for inclusion in the prompt.
        
        Args:
            paths: List of paths (each path is a list of node IDs)
            
        Returns:
            Formatted string describing the paths
        """
        if not paths:
            return "No relevant relationships found."
        
        formatted_paths = []
        
        for path in paths:
            path_description = []
            
            for i in range(len(path) - 1):
                source = path[i]
                target = path[i + 1]
                
                # Get node names
                source_name = self.graph.nodes[source].get("name", source)
                target_name = self.graph.nodes[target].get("name", target)
                
                # Get edge type
                edge_data = self.graph.get_edge_data(source, target)
                relation = edge_data.get("type", "related_to") if edge_data else "related_to"
                
                # Format relationship
                path_description.append(f"{source_name} {relation} {target_name}")
            
            formatted_paths.append(" -> ".join(path_description))
        
        return "\n".join(formatted_paths)

    def retrieve_relevant_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from the vector database.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Extract documents
        documents = []
        for match in results.matches:
            doc = {
                "id": match.id,
                "score": match.score,
                "text": match.metadata.get("text", ""),
                "metadata": {k: v for k, v in match.metadata.items() if k != "text"}
            }
            documents.append(doc)
        
        return documents

    def extract_query_entities(self, query: str) -> Dict[str, List[str]]:
        """
        Extract entities from the user query.
        
        Args:
            query: User query
            
        Returns:
            Dictionary of entity types and their instances
        """
        return self.extract_entities_from_text(query)

    def get_entity_ids_from_names(self, entities: Dict[str, List[str]]) -> List[str]:
        """
        Convert entity names to entity IDs.
        
        Args:
            entities: Dictionary of entity types and their instances
            
        Returns:
            List of entity IDs
        """
        entity_ids = []
        
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                entity_id = f"{entity_type}_{entity.lower().replace(' ', '_')}"
                if self.graph.has_node(entity_id):
                    entity_ids.append(entity_id)
        
        return entity_ids

    def answer_question(self, question: str) -> str:
        """
        Answer a fitness-related question using Graph RAG.
        
        Args:
            question: User question
            
        Returns:
            Answer to the question
        """
        # Step 1: Retrieve relevant documents
        documents = self.retrieve_relevant_documents(question, top_k=5)
        context = "\n\n".join([doc["text"] for doc in documents])
        
        # Step 2: Extract entities from question and documents
        question_entities = self.extract_query_entities(question)
        question_entity_ids = self.get_entity_ids_from_names(question_entities)
        
        document_entities = {}
        for doc in documents:
            doc_entities = self.extract_entities_from_text(doc["text"])
            for entity_type, entity_list in doc_entities.items():
                if entity_type not in document_entities:
                    document_entities[entity_type] = []
                document_entities[entity_type].extend(entity_list)
        
        document_entity_ids = self.get_entity_ids_from_names(document_entities)
        
        # Step 3: Find paths between question entities and document entities
        paths = self.find_paths_between_entities(question_entity_ids, document_entity_ids)
        formatted_paths = self.format_paths_for_prompt(paths)
        
        # Step 4: Generate answer using LLM with enhanced context
        prompt = self.prompt_template.format(
            context=context,
            question=question,
            graph_paths=formatted_paths
        )
        
        answer = self.llm(prompt)
        return answer

    def get_related_concepts(self, concept: str, relationship_type: Optional[str] = None) -> List[str]:
        """
        Get concepts related to a given concept in the knowledge graph.
        
        Args:
            concept: Concept name
            relationship_type: Type of relationship to filter by
            
        Returns:
            List of related concept names
        """
        # Find the concept node
        concept_id = None
        for node, data in self.graph.nodes(data=True):
            if data.get("name", "").lower() == concept.lower():
                concept_id = node
                break
        
        if not concept_id:
            return []
        
        # Get neighbors
        related = []
        for neighbor in self.graph.successors(concept_id):
            edge_data = self.graph.get_edge_data(concept_id, neighbor)
            edge_type = edge_data.get("type", "")
            
            if relationship_type is None or edge_type == relationship_type:
                neighbor_data = self.graph.nodes[neighbor]
                related.append(neighbor_data.get("name", neighbor))
        
        return related

    def visualize_graph(self, output_file: str = "fitness_knowledge_graph.html"):
        """
        Visualize the knowledge graph using NetworkX and Pyvis.
        
        Args:
            output_file: Path to save the visualization
        """
        try:
            from pyvis.network import Network
            
            # Create network
            net = Network(notebook=False, height="750px", width="100%")
            
            # Add nodes
            for node, data in self.graph.nodes(data=True):
                node_type = data.get("type", "unknown")
                label = data.get("name", node)
                
                # Set node color based on type
                color_map = {
                    "exercise": "#ff7f0e",
                    "workout": "#1f77b4",
                    "nutrition": "#2ca02c",
                    "muscle_group": "#d62728",
                    "equipment": "#9467bd",
                    "fitness_goal": "#8c564b",
                    "document": "#e377c2"
                }
                
                color = color_map.get(node_type, "#7f7f7f")
                
                net.add_node(node, label=label, title=str(data), color=color)
            
            # Add edges
            for source, target, data in self.graph.edges(data=True):
                edge_type = data.get("type", "related_to")
                net.add_edge(source, target, title=edge_type)
            
            # Save visualization
            net.save_graph(output_file)
            logger.info(f"Graph visualization saved to {output_file}")
            
            return output_file
        except ImportError:
            logger.warning("Pyvis not installed. Cannot visualize graph.")
            return None
