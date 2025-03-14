"""
Advanced RAG Implementation for PersonalTrainerAI

This module implements an Advanced RAG approach thatconda deacti enhances the baseline with:
1. Query expansion using LLM
2. Sentence-window retrieval for better context
3. Re-ranking of retrieved documents
4. Dynamic context window based on relevance
5. Structured prompt engineering
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import pinecone
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

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

class AdvancedRAG:
    """
    An advanced RAG implementation with query expansion, sentence-window retrieval,
    and re-ranking for improved performance.
    """
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        llm_model_name: str = "gpt-3.5-turbo",
        top_k: int = 10,
        rerank_top_k: int = 5,
        expansion_queries: int = 3
    ):
        """
        Initialize the AdvancedRAG system.
        
        Args:
            embedding_model_name: Name of the HuggingFace embedding model
            llm_model_name: Name of the LLM model
            top_k: Number of documents to retrieve initially
            rerank_top_k: Number of documents to keep after reranking
            expansion_queries: Number of expanded queries to generate
        """
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.expansion_queries = expansion_queries
        
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
        
        # Define query expansion prompt
        self.query_expansion_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            You are a fitness expert helping to expand a user's question to improve search results.
            Generate {expansion_queries} different versions of the following question that capture the same intent but use different wording or focus on different aspects.
            Format your response as a numbered list with each question on a new line.
            
            Original question: {query}
            
            Expanded questions:
            """.format(expansion_queries=expansion_queries)
        )
        
        # Create query expansion chain
        self.query_expansion_chain = LLMChain(llm=self.llm, prompt=self.query_expansion_prompt)
        
        # Define reranking prompt
        self.reranking_prompt = PromptTemplate(
            input_variables=["query", "document"],
            template="""
            You are a fitness expert evaluating the relevance of a document to a user's question.
            On a scale of 1-10, how relevant is this document to answering the question?
            Provide only a number as your response.
            
            User question: {query}
            
            Document: {document}
            
            Relevance score (1-10):
            """
        )
        
        # Create reranking chain
        self.reranking_chain = LLMChain(llm=self.llm, prompt=self.reranking_prompt)
        
        # Define response generation prompt
        self.response_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""
            You are a knowledgeable personal fitness trainer assistant. Use the following retrieved information to answer the user's question.
            
            Guidelines:
            - Provide a comprehensive and detailed answer
            - Include specific exercises, techniques, or training principles when relevant
            - Explain the scientific reasoning behind your recommendations when possible
            - If the retrieved information doesn't fully answer the question, acknowledge the limitations
            - Format your response in a clear, structured way with appropriate headings and bullet points
            - Cite the sources of information when possible
            
            Retrieved information:
            {context}
            
            User question: {query}
            
            Your answer:
            """
        )
        
        # Create response generation chain
        self.response_chain = LLMChain(llm=self.llm, prompt=self.response_prompt)
        
        logger.info("AdvancedRAG initialized successfully")
    
    def expand_query(self, query: str) -> List[str]:
        """
        Expand the original query into multiple related queries.
        
        Args:
            query: Original user query
            
        Returns:
            List of expanded queries
        """
        logger.info(f"Expanding query: {query}")
        
        # Generate expanded queries
        expansion_result = self.query_expansion_chain.run(query=query)
        
        # Parse the result into a list of queries
        expanded_queries = []
        for line in expansion_result.strip().split('\n'):
            # Remove numbering and any leading/trailing whitespace
            if line and any(line.strip().startswith(f"{i}.") for i in range(1, 10)):
                expanded_query = line.strip().split('.', 1)[1].strip()
                expanded_queries.append(expanded_query)
        
        # Add the original query
        expanded_queries.append(query)
        
        logger.info(f"Generated {len(expanded_queries)} queries")
        return expanded_queries
    
    def retrieve_with_expanded_queries(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve documents using multiple expanded queries.
        
        Args:
            queries: List of queries to use for retrieval
            
        Returns:
            Combined list of retrieved documents
        """
        logger.info(f"Retrieving documents for {len(queries)} queries")
        
        all_results = []
        seen_ids = set()
        
        for query in queries:
            # Generate query embedding
            query_embedding = self.embedding_model.embed_query(query)
            
            # Search Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=self.top_k,
                include_metadata=True
            )
            
            # Add unique results to the combined list
            for match in results['matches']:
                if match['id'] not in seen_ids:
                    all_results.append(match)
                    seen_ids.add(match['id'])
        
        logger.info(f"Retrieved {len(all_results)} unique documents")
        return all_results
    
    def rerank_documents(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank retrieved documents based on relevance to the query.
        
        Args:
            query: User query
            documents: List of retrieved documents
            
        Returns:
            Reranked list of documents
        """
        logger.info(f"Reranking {len(documents)} documents")
        
        scored_docs = []
        
        for doc in documents:
            metadata = doc.get('metadata', {})
            text = metadata.get('text', '')
            
            # Get relevance score from LLM
            try:
                score_text = self.reranking_chain.run(query=query, document=text)
                # Extract numeric score
                score = float(score_text.strip())
            except:
                # Default score if parsing fails
                score = 5.0
            
            scored_docs.append((doc, score))
        
        # Sort by score in descending order
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Take top k documents
        reranked_docs = [doc for doc, score in scored_docs[:self.rerank_top_k]]
        
        logger.info(f"Reranked to {len(reranked_docs)} documents")
        return reranked_docs
    
    def apply_sentence_window(self, text: str, window_size: int = 3) -> str:
        """
        Apply sentence window to expand context around relevant sentences.
        
        Args:
            text: Original text
            window_size: Number of sentences to include before and after
            
        Returns:
            Expanded text with sentence window
        """
        # Simple sentence splitting (can be improved with NLP libraries)
        sentences = text.split('. ')
        
        # If text is short enough, return as is
        if len(sentences) <= (window_size * 2 + 1):
            return text
        
        # Find the most relevant sentence (middle of the text as a simple heuristic)
        middle_idx = len(sentences) // 2
        
        # Apply window
        start_idx = max(0, middle_idx - window_size)
        end_idx = min(len(sentences), middle_idx + window_size + 1)
        
        # Join sentences back together
        windowed_text = '. '.join(sentences[start_idx:end_idx])
        if not windowed_text.endswith('.'):
            windowed_text += '.'
            
        return windowed_text
    
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
            url = metadata.get('url', '')
            
            # Apply sentence window for longer texts
            if len(text.split()) > 100:
                text = self.apply_sentence_window(text)
            
            # Format with source information
            source_info = f"{source}"
            if url:
                source_info += f" ({url})"
                
            context_part = f"Document {i+1}:\nTitle: {title}\nSource: {source_info}\nContent: {text}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def generate_response(self, query: str, context: str) -> str:
        """
        Generate a response using the LLM based on the query and retrieved context.
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Generated response
        """
        logger.info("Generating response with LLM")
        response = self.response_chain.run(query=query, context=context)
        return response
    
    def query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query through the advanced RAG pipeline.
        
        Args:
            query: User query
            
        Returns:
            Dictionary containing the query, retrieved documents, and generated response
        """
        logger.info(f"Processing query: {query}")
        
        # Expand query
        expanded_queries = self.expand_query(query)
        
        # Retrieve documents with expanded queries
        retrieved_docs = self.retrieve_with_expanded_queries(expanded_queries)
        
        # Rerank documents
        reranked_docs = self.rerank_documents(query, retrieved_docs)
        
        # Format context
        context = self.format_context(reranked_docs)
        
        # Generate response
        response = self.generate_response(query, context)
        
        return {
            "query": query,
            "expanded_queries": expanded_queries,
            "retrieved_documents": retrieved_docs,
            "reranked_documents": reranked_docs,
            "response": response
        }


# Example usage
if __name__ == "__main__":
    # Initialize RAG system
    rag = AdvancedRAG()
    
    # Example query
    query = "What's the best way to improve my bench press?"
    
    # Process query
    result = rag.query(query)
    
    # Print response
    print(f"Query: {result['query']}")
    print(f"Expanded queries: {result['expanded_queries']}")
    print(f"Response: {result['response']}")
