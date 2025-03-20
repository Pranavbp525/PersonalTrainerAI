"""
Advanced RAG Implementation for PersonalTrainerAI

This module implements an advanced RAG approach with reranking and query expansion.
"""

import os
import logging
from typing import List, Dict, Any, Optional
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

class AdvancedRAG:
    """
    An advanced implementation of Retrieval-Augmented Generation (RAG) for fitness knowledge.
    
    This implementation includes:
    - Query expansion
    - Reranking of retrieved documents
    - Contextual weighting
    """
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        llm_model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        top_k: int = 10,
        rerank_top_k: int = 5
    ):
        """
        Initialize the AdvancedRAG system.
        
        Args:
            embedding_model_name: Name of the embedding model to use
            llm_model_name: Name of the language model to use
            temperature: Temperature parameter for the LLM
            top_k: Number of documents to initially retrieve
            rerank_top_k: Number of documents to keep after reranking
        """
        # Load environment variables
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        self.PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "fitness-chatbot")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        
        if not self.PINECONE_API_KEY or not self.OPENAI_API_KEY:
            raise ValueError("Missing required environment variables. Please check your .env file.")
        
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        
        # Initialize embedding model
        logger.info(f"Initializing embedding model: {embedding_model_name}")
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        # Initialize Pinecone
        logger.info("Connecting to Pinecone")
        self.pc = Pinecone(api_key=self.PINECONE_API_KEY)
        self.index = self.pc.Index(self.PINECONE_INDEX_NAME)
        
        # Initialize LLMs
        logger.info(f"Initializing LLM: {llm_model_name}")
        self.llm = OpenAI(model_name=llm_model_name, temperature=temperature, openai_api_key=self.OPENAI_API_KEY)
        self.query_expansion_llm = OpenAI(model_name=llm_model_name, temperature=0.2, openai_api_key=self.OPENAI_API_KEY)
        self.reranker_llm = OpenAI(model_name=llm_model_name, temperature=0.0, openai_api_key=self.OPENAI_API_KEY)
        
        # Define prompt templates
        self.query_expansion_template = PromptTemplate(
            input_variables=["query"],
            template="""
            Generate three different versions of the following query to improve search results. 
            Each version should rephrase the query while preserving its original intent.
            
            Original query: {query}
            
            Output the three versions as a comma-separated list without numbering or additional text.
            """
        )
        
        self.reranker_template = PromptTemplate(
            input_variables=["query", "document"],
            template="""
            Rate the relevance of this document to the query on a scale of 0 to 10.
            
            Query: {query}
            
            Document: {document}
            
            Provide only a numerical score from 0 to 10, where 0 means completely irrelevant and 10 means perfectly relevant.
            """
        )
        
        self.answer_template = PromptTemplate(
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
        
        # Create LLM chains
        self.query_expansion_chain = LLMChain(llm=self.query_expansion_llm, prompt=self.query_expansion_template)
        self.reranker_chain = LLMChain(llm=self.reranker_llm, prompt=self.reranker_template)
        self.answer_chain = LLMChain(llm=self.llm, prompt=self.answer_template)
        
        logger.info("AdvancedRAG initialized successfully")
    
    def expand_query(self, query: str) -> List[str]:
        """
        Expand the original query into multiple variations.
        
        Args:
            query: The original query string
            
        Returns:
            A list of query variations
        """
        logger.info(f"Expanding query: {query}")
        
        # Generate query variations
        response = self.query_expansion_chain.run(query=query)
        
        # Parse response
        variations = [query]  # Always include the original query
        for var in response.strip().split(','):
            variations.append(var.strip())
        
        logger.info(f"Generated {len(variations)} query variations")
        return variations
    
    def retrieve_documents(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from the vector database for multiple query variations.
        
        Args:
            queries: List of query strings
            
        Returns:
            A list of retrieved documents
        """
        logger.info(f"Retrieving documents for {len(queries)} query variations")
        
        all_documents = {}  # Use dict to deduplicate by ID
        
        for query in queries:
            # Generate query embedding
            query_embedding = self.embedding_model.embed_query(query)
            
            # Query Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=self.top_k,
                include_metadata=True
            )
            
            # Extract documents from results
            for match in results.matches:
                if hasattr(match, 'metadata') and match.metadata:
                    # Use ID as key to deduplicate
                    all_documents[match.id] = {
                        "id": match.id,
                        "score": match.score,
                        "text": match.metadata.get("text", ""),
                        "source": match.metadata.get("source", "Unknown")
                    }
        
        documents = list(all_documents.values())
        logger.info(f"Retrieved {len(documents)} unique documents")
        return documents
    
    def rerank_documents(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank documents based on their relevance to the query.
        
        Args:
            query: The original query string
            documents: List of retrieved documents
            
        Returns:
            Reranked list of documents
        """
        logger.info(f"Reranking {len(documents)} documents")
        
        reranked_documents = []
        for doc in documents:
            # Get relevance score from LLM
            score_text = self.reranker_chain.run(query=query, document=doc["text"])
            
            try:
                # Parse score
                relevance_score = float(score_text.strip())
                # Ensure score is in range [0, 10]
                relevance_score = max(0, min(10, relevance_score))
            except ValueError:
                # Default score if parsing fails
                relevance_score = 5.0
            
            # Add relevance score to document
            doc["relevance_score"] = relevance_score
            reranked_documents.append(doc)
        
        # Sort by relevance score
        reranked_documents.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Keep top k
        reranked_documents = reranked_documents[:self.rerank_top_k]
        
        logger.info(f"Kept top {len(reranked_documents)} documents after reranking")
        return reranked_documents
    
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
            relevance_info = f" [Relevance: {doc.get('relevance_score', 'N/A')}/10]" if 'relevance_score' in doc else ""
            context_parts.append(f"Document {i+1} [Source: {doc['source']}]{relevance_info}:\n{doc['text']}\n")
        
        return "\n".join(context_parts)
    
    def answer_question(self, question: str) -> str:
        """
        Answer a question using the advanced RAG approach.
        
        Args:
            question: The question to answer
            
        Returns:
            Generated answer
        """
        logger.info(f"Answering question: {question}")
        
        # Expand query
        expanded_queries = self.expand_query(question)
        
        # Retrieve documents using expanded queries
        documents = self.retrieve_documents(expanded_queries)
        
        if not documents:
            return "I couldn't find any relevant information to answer your question."
        
        # Rerank documents
        reranked_documents = self.rerank_documents(question, documents)
        
        # Format context
        context = self.format_context(reranked_documents)
        
        # Generate answer
        response = self.answer_chain.run(context=context, question=question)
        
        return response.strip()


if __name__ == "__main__":
    # Example usage
    rag = AdvancedRAG()
    question = "How do I improve my squat form?"
    answer = rag.answer_question(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
