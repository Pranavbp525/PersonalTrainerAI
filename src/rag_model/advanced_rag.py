"""
Advanced RAG Implementation for PersonalTrainerAI

This module implements an advanced Retrieval-Augmented Generation (RAG) approach
for fitness knowledge with query expansion and reranking.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class AdvancedRAG:
    """
    An advanced RAG implementation for fitness knowledge with query expansion and reranking.
    """
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        llm_model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        top_k: int = 5,
        reranking_threshold: float = 0.7
    ):
        """
        Initialize the advanced RAG system.
        
        Args:
            embedding_model_name: Name of the embedding model to use
            llm_model_name: Name of the language model to use
            temperature: Temperature parameter for the LLM
            top_k: Number of documents to retrieve
            reranking_threshold: Threshold for reranking documents
        """
        # Load environment variables
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        self.PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "fitness-chatbot")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        
        if not self.PINECONE_API_KEY or not self.OPENAI_API_KEY:
            raise ValueError("Missing required environment variables. Please check your .env file.")
        
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
        
        # Set parameters
        self.top_k = top_k
        self.reranking_threshold = reranking_threshold
        
        # Define prompt templates
        self.query_expansion_template = PromptTemplate(
            input_variables=["query"],
            template="""
            You are a fitness expert. Given the following question, generate 3 alternative versions 
            that capture the same information need but with different wording or focus.
            
            Original question: {query}
            
            Alternative questions:
            1.
            2.
            3.
            """
        )
        
        self.reranking_template = PromptTemplate(
            input_variables=["query", "document"],
            template="""
            You are a fitness expert evaluating the relevance of a document to a query.
            
            Query: {query}
            
            Document: {document}
            
            On a scale of 1-10, how relevant is this document to the query?
            Provide only a numerical score, nothing else.
            """
        )
        
        self.answer_generation_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are a knowledgeable fitness trainer assistant. Use the following retrieved information to answer the question.
            
            Retrieved information:
            {context}
            
            Question: {question}
            
            Provide a comprehensive, accurate, and helpful answer based on the retrieved information.
            If the retrieved information doesn't contain the answer, acknowledge that and provide general advice.
            
            Answer:
            """
        )
        
        # Create LLM chains
        self.query_expansion_chain = LLMChain(llm=self.llm, prompt=self.query_expansion_template)
        self.reranking_chain = LLMChain(llm=self.llm, prompt=self.reranking_template)
        self.answer_generation_chain = LLMChain(llm=self.llm, prompt=self.answer_generation_template)
        
        logger.info("AdvancedRAG initialized successfully")
    
    def expand_query(self, query: str) -> List[str]:
        """
        Expand a query into multiple variations.
        
        Args:
            query: The original query
            
        Returns:
            List of expanded queries
        """
        logger.info(f"Expanding query: {query}")
        
        # Generate expanded queries
        response = self.query_expansion_chain.invoke({"query": query})
        
        # Parse response
        expanded_queries = []
        for line in response["text"].strip().split("\n"):
            line = line.strip()
            if line and line[0].isdigit() and "." in line:
                expanded_query = line.split(".", 1)[1].strip()
                if expanded_query:
                    expanded_queries.append(expanded_query)
        
        # Add original query
        if query not in expanded_queries:
            expanded_queries.append(query)
        
        logger.info(f"Generated {len(expanded_queries)} expanded queries")
        return expanded_queries
    
    def retrieve_documents(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve documents for multiple queries.
        
        Args:
            queries: List of queries
            
        Returns:
            List of retrieved documents
        """
        logger.info(f"Retrieving documents for {len(queries)} queries")
        
        all_documents = []
        seen_ids = set()
        
        for query in queries:
            # Generate query embedding
            query_embedding = self.embedding_model.embed_query(query)
            
            # Query Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=self.top_k,
                include_metadata=True
            )
            
            # Extract documents
            for match in results["matches"]:
                if match["id"] not in seen_ids and "metadata" in match and "text" in match["metadata"]:
                    all_documents.append({
                        "text": match["metadata"]["text"],
                        "score": match["score"],
                        "id": match["id"]
                    })
                    seen_ids.add(match["id"])
        
        logger.info(f"Retrieved {len(all_documents)} unique documents")
        return all_documents
    
    def rerank_documents(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to the query.
        
        Args:
            query: The original query
            documents: List of retrieved documents
            
        Returns:
            List of reranked documents
        """
        logger.info(f"Reranking {len(documents)} documents")
        
        reranked_documents = []
        
        for doc in documents:
            # Generate relevance score
            response = self.reranking_chain.invoke({"query": query, "document": doc["text"]})
            
            try:
                # Parse score
                relevance_score = float(response["text"].strip())
                
                # Add to reranked documents if above threshold
                if relevance_score >= self.reranking_threshold * 10:  # Scale to 1-10
                    reranked_documents.append({
                        "text": doc["text"],
                        "score": doc["score"],
                        "relevance_score": relevance_score,
                        "id": doc["id"]
                    })
            except ValueError:
                logger.warning(f"Failed to parse relevance score: {response}")
        
        # Sort by relevance score
        reranked_documents.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        logger.info(f"Reranked to {len(reranked_documents)} documents")
        return reranked_documents
    
    def answer_question(self, query: str, return_contexts: bool = False):
        """
        Answer a question using the advanced RAG approach.
        
        Args:
            query: The question to answer
            return_contexts: Whether to return retrieved contexts along with the answer
            
        Returns:
            If return_contexts is False: The generated answer
            If return_contexts is True: Tuple of (answer, contexts)
        """
        # Expand query
        expanded_query = self._expand_query(query)
        
        # Retrieve documents
        documents = self._retrieve_documents(expanded_query)
        
        # Re-rank documents
        reranked_documents = self._rerank_documents(documents, query)
        
        # Process documents with sentence window
        processed_documents = self._process_with_sentence_window(reranked_documents)
        
        # Generate response
        response = self._generate_response(query, processed_documents)
        
        if return_contexts:
            # Extract text from documents for evaluation
            contexts = [doc.page_content for doc in processed_documents]
            return response, contexts
        else:
            return response


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced RAG for fitness knowledge")
    parser.add_argument("--question", type=str, help="Question to answer")
    args = parser.parse_args()
    
    # Initialize RAG
    rag = AdvancedRAG()
    
    # Answer question
    if args.question:
        answer = rag.answer_question(args.question)
        print(f"Question: {args.question}")
        print(f"Answer: {answer}")
    else:
        # Interactive mode
        print("Advanced RAG for fitness knowledge")
        print("Enter 'quit' to exit")
        
        while True:
            question = input("\nEnter your fitness question: ")
            if question.lower() in ["quit", "exit", "q"]:
                break
                
            answer = rag.answer_question(question)
            print(f"\nAnswer: {answer}")
