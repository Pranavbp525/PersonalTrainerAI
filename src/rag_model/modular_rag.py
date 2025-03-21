"""
Modular RAG Implementation for PersonalTrainerAI

This module implements a modular Retrieval-Augmented Generation (RAG) approach
for fitness knowledge with query classification and specialized retrievers.
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

class ModularRAG:
    """
    A modular RAG implementation for fitness knowledge with query classification
    and specialized retrievers.
    """
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        llm_model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        top_k: int = 5
    ):
        """
        Initialize the modular RAG system.
        
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
        
        # Define query categories
        self.query_categories = [
            "workout_routines",
            "nutrition_diet",
            "exercise_technique",
            "fitness_equipment",
            "health_wellness",
            "general"
        ]
        
        # Define prompt templates
        self.query_classification_template = PromptTemplate(
            input_variables=["query", "categories"],
            template="""
            You are a fitness expert. Classify the following question into one of these categories:
            {categories}
            
            Question: {query}
            
            Respond with only the category name, nothing else.
            """
        )
        
        self.specialized_query_template = PromptTemplate(
            input_variables=["query", "category"],
            template="""
            You are a fitness expert specializing in {category}. 
            Reformulate the following question to better target information in your area of expertise.
            
            Original question: {query}
            
            Reformulated question:
            """
        )
        
        self.answer_generation_template = PromptTemplate(
            input_variables=["context", "question", "category"],
            template="""
            You are a knowledgeable fitness trainer assistant specializing in {category}.
            Use the following retrieved information to answer the question.
            
            Retrieved information:
            {context}
            
            Question: {question}
            
            Provide a comprehensive, accurate, and helpful answer based on the retrieved information.
            If the retrieved information doesn't contain the answer, acknowledge that and provide general advice.
            
            Answer:
            """
        )
        
        # Create LLM chains
        self.query_classification_chain = LLMChain(llm=self.llm, prompt=self.query_classification_template)
        self.specialized_query_chain = LLMChain(llm=self.llm, prompt=self.specialized_query_template)
        self.answer_generation_chain = LLMChain(llm=self.llm, prompt=self.answer_generation_template)
        
        logger.info("ModularRAG initialized successfully")
    
    def classify_query(self, query: str) -> str:
        """
        Classify a query into one of the predefined categories.
        
        Args:
            query: The query to classify
            
        Returns:
            Category name
        """
        logger.info(f"Classifying query: {query}")
        
        # Format categories for prompt
        categories_str = "\n".join(self.query_categories)
        
        # Classify query
        response = self.query_classification_chain.invoke({"query": query, "categories": categories_str})
        
        # Extract category
        category = response["text"].strip().lower()
        
        # Validate category
        if category not in self.query_categories:
            logger.warning(f"Invalid category: {category}, defaulting to general")
            category = "general"
        
        logger.info(f"Classified as: {category}")
        return category
    
    def specialize_query(self, query: str, category: str) -> str:
        """
        Specialize a query for a specific category.
        
        Args:
            query: The original query
            category: The query category
            
        Returns:
            Specialized query
        """
        logger.info(f"Specializing query for category: {category}")
        
        # Skip specialization for general category
        if category == "general":
            return query
        
        # Specialize query
        response = self.specialized_query_chain.invoke({"query": query, "category": category})
        
        # Extract specialized query
        specialized_query = response["text"].strip()
        
        logger.info(f"Specialized query: {specialized_query}")
        return specialized_query
    
    def retrieve_documents(self, query: str, category: str) -> List[Dict[str, Any]]:
        """
        Retrieve documents for a query with category-specific filtering.
        
        Args:
            query: The query to retrieve documents for
            category: The query category
            
        Returns:
            List of retrieved documents
        """
        logger.info(f"Retrieving documents for query in category: {category}")
        
        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Prepare filter based on category
        filter_dict = {}
        if category != "general":
            filter_dict = {"metadata": {"category": category}}
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=self.top_k,
            include_metadata=True,
            filter=filter_dict
        )
        
        # Extract documents
        documents = []
        for match in results["matches"]:
            if "metadata" in match and "text" in match["metadata"]:
                documents.append({
                    "text": match["metadata"]["text"],
                    "score": match["score"],
                    "id": match["id"],
                    "category": match["metadata"].get("category", "unknown")
                })
        
        logger.info(f"Retrieved {len(documents)} documents")
        return documents
    
    def answer_question(self, question: str) -> str:
        """
        Answer a question using the modular RAG approach.
        
        Args:
            question: The question to answer
            
        Returns:
            Generated answer
        """
        logger.info(f"Answering question: {question}")
        
        # Classify query
        category = self.classify_query(question)
        
        # Specialize query
        specialized_query = self.specialize_query(question, category)
        
        # Retrieve documents
        documents = self.retrieve_documents(specialized_query, category)
        
        # Prepare context
        if documents:
            context = "\n\n".join([f"Document {i+1}:\n{doc['text']}" for i, doc in enumerate(documents)])
        else:
            context = "No relevant documents found."
        
        # Generate answer
        response = self.answer_generation_chain.invoke({
            "context": context, 
            "question": question,
            "category": category
        })
        
        return response["text"].strip()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Modular RAG for fitness knowledge")
    parser.add_argument("--question", type=str, help="Question to answer")
    args = parser.parse_args()
    
    # Initialize RAG
    rag = ModularRAG()
    
    # Answer question
    if args.question:
        answer = rag.answer_question(args.question)
        print(f"Question: {args.question}")
        print(f"Answer: {answer}")
    else:
        # Interactive mode
        print("Modular RAG for fitness knowledge")
        print("Enter 'quit' to exit")
        
        while True:
            question = input("\nEnter your fitness question: ")
            if question.lower() in ["quit", "exit", "q"]:
                break
                
            answer = rag.answer_question(question)
            print(f"\nAnswer: {answer}")
