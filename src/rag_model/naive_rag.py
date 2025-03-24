"""
Naive RAG Implementation for PersonalTrainerAI

This module implements a basic RAG approach with simple vector similarity search.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class NaiveRAG:
    """
    A naive implementation of Retrieval-Augmented Generation (RAG) for fitness knowledge.
    
    This implementation uses a simple vector similarity search to retrieve relevant documents
    and then generates an answer using an LLM.
    """
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        llm_model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        top_k: int = 5
    ):
        """
        Initialize the NaiveRAG system.
        
        Args:
            embedding_model_name: Name of the embedding model to use
            llm_model_name: Name of the language model to use
            temperature: Temperature parameter for the LLM
            top_k: Number of documents to retrieve
        """
        # Load environment variables
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        self.PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "fitness-chatbot")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        
        if not self.PINECONE_API_KEY or not self.OPENAI_API_KEY:
            raise ValueError("Missing required environment variables. Please check your .env file.")
        
        self.top_k = top_k
        
        # Initialize embedding model
        logger.info(f"Initializing embedding model: {embedding_model_name}")
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        # Initialize Pinecone
        logger.info("Connecting to Pinecone")
        self.pc = Pinecone(api_key=self.PINECONE_API_KEY)
        self.index = self.pc.Index(self.PINECONE_INDEX_NAME)
        
        # Initialize LLM
        logger.info(f"Initializing LLM: {llm_model_name}")
        self.llm = ChatOpenAI(model_name=llm_model_name, temperature=temperature, openai_api_key=self.OPENAI_API_KEY)
        
        # Define prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are a knowledgeable fitness trainer assistant. Use the following retrieved information to answer the question.
            
            Retrieved information:
            {context}
            
            Question: {question}
            
            Provide a comprehensive and accurate answer based on the retrieved information. If the information doesn't contain the answer, say "I don't have enough information to answer this question."
            
            Answer:
            """
        )
        
        # Create LLM chain
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        
        logger.info("NaiveRAG initialized successfully")
    
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
        Answer a question using the RAG approach.
        
        Args:
            question: The question to answer
            
        Returns:
            Generated answer
        """
        logger.info(f"Answering question: {question}")
        
        # Retrieve relevant documents
        documents = self.retrieve_documents(question)
        
        if not documents:
            return "I couldn't find any relevant information to answer your question."
        
        # Format context
        context = self.format_context(documents)
        
        # Generate answer
        response = self.llm_chain.run(context=context, question=question)
        
        return response.strip()


if __name__ == "__main__":
    # Example usage
    rag = NaiveRAG()
    question = "What is a good workout routine for beginners?"
    answer = rag.answer_question(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
