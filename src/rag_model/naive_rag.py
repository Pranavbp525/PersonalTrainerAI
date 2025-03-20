"""
Naive RAG Implementation for PersonalTrainerAI

This module implements a baseline Naive RAG approach that:
1. Embeds the user query
2. Retrieves the most similar documents from Pinecone
3. Passes the retrieved documents to an LLM to generate a response
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pinecone import Pinecone
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
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "fitness-chatbot")

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class NaiveRAG:
    """
    A simple RAG implementation that retrieves documents based on vector similarity
    and generates responses using an LLM.
    """
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        llm_model_name: str = "gpt-3.5-turbo",
        top_k: int = 5
    ):
        """
        Initialize the NaiveRAG system.
        
        Args:
            embedding_model_name: Name of the HuggingFace embedding model
            llm_model_name: Name of the LLM model
            top_k: Number of documents to retrieve
        """
        self.top_k = top_k
        
        # Initialize embedding model
        logger.info(f"Initializing embedding model: {embedding_model_name}")
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        # Initialize Pinecone
        logger.info("Connecting to Pinecone")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
                
        # Initialize LLM
        logger.info(f"Initializing LLM: {llm_model_name}")
        self.llm = OpenAI(model_name=llm_model_name, temperature=0.7, api_key=OPENAI_API_KEY)
        
        # Define prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["query", "context"],
            template="""
            You are a knowledgeable personal fitness trainer assistant. Use the following retrieved information to answer the user's question.
            If you don't know the answer based on the retrieved information, say that you don't know.
            
            Retrieved information:
            {context}
            
            User question: {query}
            
            Your answer:
            """
        )
        
        # Create LLM chain
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        
        logger.info("NaiveRAG initialized successfully")
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from Pinecone based on query similarity.
        
        Args:
            query: User query
            
        Returns:
            List of retrieved documents with metadata
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
        
        logger.info(f"Retrieved {len(results['matches'])} documents")
        return results['matches']
    
    def format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into a context string for the LLM.
        
        Args:
            retrieved_docs: List of retrieved documents from Pinecone
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs):
            metadata = doc.get('metadata', {})
            text = metadata.get('text', '')
            title = metadata.get('title', 'Unknown Title')
            source = metadata.get('source', 'Unknown Source')
            
            context_part = f"Document {i+1}:\nTitle: {title}\nSource: {source}\nContent: {text}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def generate(self, query: str, context: str) -> str:
        """
        Generate a response using the LLM based on the query and retrieved context.
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Generated response
        """
        logger.info("Generating response with LLM")
        response = self.llm_chain.run(query=query, context=context)
        return response
    
    def query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query through the RAG pipeline.
        
        Args:
            query: User query
            
        Returns:
            Dictionary containing the query, retrieved documents, and generated response
        """
        logger.info(f"Processing query: {query}")
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query)
        
        # Format context
        context = self.format_context(retrieved_docs)
        
        # Generate response
        response = self.generate(query, context)
        
        return {
            "query": query,
            "retrieved_documents": retrieved_docs,
            "response": response
        }


# Example usage
if __name__ == "__main__":
    # Initialize RAG system
    rag = NaiveRAG()
    
    # Example query
    query = "What's the best way to improve my bench press?"
    
    # Process query
    result = rag.query(query)
    
    # Print response
    print(f"Query: {result['query']}")
    print(f"Response: {result['response']}")
