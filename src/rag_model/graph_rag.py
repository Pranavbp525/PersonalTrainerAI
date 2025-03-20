"""
Graph RAG Implementation for PersonalTrainerAI

This module implements a Graph RAG approach using a knowledge graph structure.
"""

import os
import logging
import pickle
from typing import List, Dict, Any, Optional, Set, Tuple
import networkx as nx
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class GraphRAG:
    """
    A graph-based implementation of Retrieval-Augmented Generation (RAG) for fitness knowledge.
    
    This implementation uses a knowledge graph structure to represent relationships
    between fitness concepts, enhancing retrieval with graph-based context.
    """
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        llm_model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        top_k: int = 5,
        graph_path: str = "fitness_knowledge_graph.gpickle"
    ):
        """
        Initialize the GraphRAG system.
        
        Args:
            embedding_model_name: Name of the embedding model to use
            llm_model_name: Name of the language model to use
            temperature: Temperature parameter for the LLM
            top_k: Number of documents to retrieve
            graph_path: Path to save/load the knowledge graph
        """
        # Load environment variables
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        self.PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "fitness-chatbot")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        
        if not self.PINECONE_API_KEY or not self.OPENAI_API_KEY:
            raise ValueError("Missing required environment variables. Please check your .env file.")
        
        self.top_k = top_k
        self.graph_path = graph_path
        
        # Initialize embedding model
        logger.info(f"Initializing embedding model: {embedding_model_name}")
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        # Initialize Pinecone
        logger.info("Connecting to Pinecone")
        self.pc = Pinecone(api_key=self.PINECONE_API_KEY)
        self.index = self.pc.Index(self.PINECONE_INDEX_NAME)
        
        # Initialize LLM
        logger.info(f"Initializing LLM: {llm_model_name}")
        self.llm = OpenAI(model_name=llm_model_name, temperature=temperature, openai_api_key=self.OPENAI_API_KEY)
        self.graph_builder_llm = OpenAI(model_name=llm_model_name, temperature=0.2, openai_api_key=self.OPENAI_API_KEY)
        
        # Define prompt templates
        self.entity_extraction_template = PromptTemplate(
            input_variables=["text"],
            template="""
            Extract all fitness-related entities from the following text. 
            Entities can include exercises, muscles, equipment, nutrition concepts, training methods, etc.
            
            Text: {text}
            
            Return only a comma-separated list of entities, nothing else.
            """
        )
        
        self.relation_extraction_template = PromptTemplate(
            input_variables=["entity1", "entity2"],
            template="""
            Determine the relationship between these two fitness concepts:
            Entity 1: {entity1}
            Entity 2: {entity2}
            
            Examples of relationships:
            - works (muscle works exercise)
            - targets (exercise targets muscle)
            - requires (exercise requires equipment)
            - complements (exercise complements another exercise)
            - precedes (exercise precedes another in a routine)
            - provides (food provides nutrient)
            - prevents (technique prevents injury)
            
            Return only the relationship as a single word or short phrase, or "none" if no clear relationship exists.
            """
        )
        
        self.answer_template = PromptTemplate(
            input_variables=["context", "graph_context", "question"],
            template="""
            You are a knowledgeable fitness trainer assistant with expertise in how different fitness concepts relate to each other.
            Use the following retrieved information and knowledge graph relationships to answer the question.
            
            Retrieved documents:
            {context}
            
            Knowledge graph relationships:
            {graph_context}
            
            Question: {question}
            
            Provide a comprehensive and accurate answer based on both the retrieved information and the knowledge graph relationships.
            Explain connections between concepts when relevant to the question.
            If the information doesn't contain the answer, say "I don't have enough information to answer this question."
            
            Answer:
            """
        )
        
        # Create LLM chains
        self.entity_extraction_chain = LLMChain(llm=self.graph_builder_llm, prompt=self.entity_extraction_template)
        self.relation_extraction_chain = LLMChain(llm=self.graph_builder_llm, prompt=self.relation_extraction_template)
        self.answer_chain = LLMChain(llm=self.llm, prompt=self.answer_template)
        
        # Initialize or load knowledge graph
        self.graph = self._load_graph() if os.path.exists(self.graph_path) else nx.DiGraph()
        
        logger.info("GraphRAG initialized successfully")
    
    def _load_graph(self) -> nx.DiGraph:
        """
        Load the knowledge graph from disk.
        
        Returns:
            The loaded knowledge graph
        """
        logger.info(f"Loading knowledge graph from {self.graph_path}")
        try:
            return nx.read_gpickle(self.graph_path)
        except Exception as e:
            logger.error(f"Error loading graph: {e}")
            return nx.DiGraph()
    
    def _save_graph(self) -> None:
        """
        Save the knowledge graph to disk.
        """
        logger.info(f"Saving knowledge graph to {self.graph_path}")
        try:
            nx.write_gpickle(self.graph, self.graph_path)
            logger.info(f"Graph saved with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        except Exception as e:
            logger.error(f"Error saving graph: {e}")
    
    def build_knowledge_graph(self, rebuild: bool = False) -> None:
        """
        Build or update the knowledge graph from the vector database.
        
        Args:
            rebuild: Whether to rebuild the graph from scratch
        """
        logger.info("Building knowledge graph from vector database")
        
        if rebuild:
            self.graph = nx.DiGraph()
        
        # Get all documents from Pinecone
        # This is a simplified approach - in a real implementation, you would need to paginate
        # through all vectors in the index
        
        # For demonstration, we'll use a sample query to get some documents
        sample_queries = [
            "workout routine", 
            "nutrition", 
            "muscle groups", 
            "exercise technique", 
            "fitness equipment"
        ]
        
        all_docs = []
        for query in sample_queries:
            query_embedding = self.embedding_model.embed_query(query)
            results = self.index.query(
                vector=query_embedding,
                top_k=100,  # Get more documents for graph building
                include_metadata=True
            )
            
            for match in results.matches:
                if hasattr(match, 'metadata') and match.metadata:
                    all_docs.append({
                        "id": match.id,
                        "text": match.metadata.get("text", ""),
                        "source": match.metadata.get("source", "Unknown")
                    })
        
        # Remove duplicates
        unique_docs = {doc["id"]: doc for doc in all_docs}.values()
        logger.info(f"Retrieved {len(unique_docs)} unique documents for graph building")
        
        # Extract entities and build graph
        all_entities = set()
        for doc in unique_docs:
            # Extract entities
            entity_text = self.entity_extraction_chain.run(text=doc["text"])
            entities = [e.strip() for e in entity_text.split(",") if e.strip()]
            
            # Add entities to graph
            for entity in entities:
                if entity not in self.graph:
                    self.graph.add_node(entity, documents=[doc["id"]])
                else:
                    if "documents" not in self.graph.nodes[entity]:
                        self.graph.nodes[entity]["documents"] = []
                    self.graph.nodes[entity]["documents"].append(doc["id"])
                
                all_entities.add(entity)
        
        logger.info(f"Extracted {len(all_entities)} entities")
        
        # Extract relationships between entities
        entities_list = list(all_entities)
        # Limit the number of entity pairs to avoid too many API calls
        max_pairs = min(1000, len(entities_list) * (len(entities_list) - 1) // 2)
        
        # Process a subset of entity pairs
        processed = 0
        for i in range(len(entities_list)):
            if processed >= max_pairs:
                break
                
            for j in range(i + 1, len(entities_list)):
                if processed >= max_pairs:
                    break
                    
                entity1 = entities_list[i]
                entity2 = entities_list[j]
                
                # Skip if relationship already exists
                if self.graph.has_edge(entity1, entity2) or self.graph.has_edge(entity2, entity1):
                    continue
                
                # Extract relationship
                relation = self.relation_extraction_chain.run(entity1=entity1, entity2=entity2)
                relation = relation.strip().lower()
                
                # Add edge if relationship exists
                if relation and relation != "none":
                    self.graph.add_edge(entity1, entity2, relation=relation)
                    processed += 1
                    
                    # Log progress periodically
                    if processed % 100 == 0:
                        logger.info(f"Processed {processed} entity pairs")
        
        logger.info(f"Added {self.graph.number_of_edges()} relationships to the graph")
        
        # Save the graph
        self._save_graph()
    
    def retrieve_documents(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from the vector database.
        
        Args:
            query: The query string
            
        Returns:
            A list of retrieved documents
        """
        logger.info(f"Retrieving documents for query: {query}")
        
        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=self.top_k,
            include_metadata=True
        )
        
        # Extract documents from results
        documents = []
        for match in results.matches:
            if hasattr(match, 'metadata') and match.metadata:
                documents.append({
                    "id": match.id,
                    "score": match.score,
                    "text": match.metadata.get("text", ""),
                    "source": match.metadata.get("source", "Unknown")
                })
        
        logger.info(f"Retrieved {len(documents)} documents")
        return documents
    
    def extract_entities_from_query(self, query: str) -> List[str]:
        """
        Extract fitness entities from the query.
        
        Args:
            query: The query string
            
        Returns:
            List of extracted entities
        """
        entity_text = self.entity_extraction_chain.run(text=query)
        entities = [e.strip() for e in entity_text.split(",") if e.strip()]
        logger.info(f"Extracted {len(entities)} entities from query: {entities}")
        return entities
    
    def get_graph_context(self, entities: List[str], max_hops: int = 2) -> str:
        """
        Get relevant subgraph context for the query entities.
        
        Args:
            entities: List of entities extracted from the query
            max_hops: Maximum number of hops from query entities
            
        Returns:
            Formatted graph context string
        """
        if not self.graph.nodes:
            logger.warning("Knowledge graph is empty. Run build_knowledge_graph() first.")
            return ""
        
        # Find matching nodes in the graph
        matched_entities = [e for e in entities if e in self.graph]
        if not matched_entities:
            logger.warning("No entities from query found in knowledge graph")
            return ""
        
        # Extract subgraph around matched entities
        subgraph_nodes = set(matched_entities)
        frontier = set(matched_entities)
        
        # Expand by hops
        for _ in range(max_hops):
            new_frontier = set()
            for node in frontier:
                # Add neighbors
                neighbors = set(self.graph.successors(node)) | set(self.graph.predecessors(node))
                new_frontier |= neighbors
            
            # Update sets
            subgraph_nodes |= new_frontier
            frontier = new_frontier
        
        # Create subgraph
        subgraph = self.graph.subgraph(subgraph_nodes)
        
        # Format relationships
        relationships = []
        for u, v, data in subgraph.edges(data=True):
            relation = data.get("relation", "related to")
            relationships.append(f"{u} {relation} {v}")
        
        logger.info(f"Found {len(relationships)} relationships in knowledge graph")
        return "\n".join(relationships)
    
    def format_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into a context string.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, doc in enumerate(documents):
            context_parts.append(f"Document {i+1} [Source: {doc['source']}]:\n{doc['text']}\n")
        
        return "\n".join(context_parts)
    
    def answer_question(self, question: str) -> str:
        """
        Answer a question using the graph-based RAG approach.
        
        Args:
            question: The question to answer
            
        Returns:
            Generated answer
        """
        logger.info(f"Answering question: {question}")
        
        # Check if graph exists
        if not self.graph.nodes:
            logger.warning("Knowledge graph is empty. Building graph first...")
            self.build_knowledge_graph()
        
        # Retrieve documents
        documents = self.retrieve_documents(question)
        
        if not documents:
            return "I couldn't find any relevant information to answer your question."
        
        # Extract entities from query
        entities = self.extract_entities_from_query(question)
        
        # Get graph context
        graph_context = self.get_graph_context(entities)
        
        # Format document context
        doc_context = self.format_context(documents)
        
        # Generate answer
        response = self.answer_chain.run(
            context=doc_context,
            graph_context=graph_context if graph_context else "No relevant relationships found.",
            question=question
        )
        
        return response.strip()


if __name__ == "__main__":
    # Example usage
    rag = GraphRAG()
    
    # Build knowledge graph if it doesn't exist
    if not os.path.exists(rag.graph_path):
        print("Building knowledge graph...")
        rag.build_knowledge_graph()
    
    question = "How are protein intake and muscle recovery related?"
    answer = rag.answer_question(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
