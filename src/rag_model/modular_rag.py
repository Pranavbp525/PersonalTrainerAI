"""
Modular RAG Implementation for PersonalTrainerAI

This module implements a modular RAG approach with query classification and specialized retrievers.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
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

class ModularRAG:
    """
    A modular implementation of Retrieval-Augmented Generation (RAG) for fitness knowledge.
    
    This implementation includes:
    - Query classification to determine the type of fitness question
    - Specialized retrievers for different types of fitness questions
    - Dynamic prompt selection based on query type
    """
    
    # Define query types
    QUERY_TYPES = [
        "workout_routine",
        "nutrition_diet",
        "exercise_technique",
        "fitness_equipment",
        "injury_prevention",
        "general_fitness"
    ]
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        llm_model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        top_k: int = 5
    ):
        """
        Initialize the ModularRAG system.
        
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
        
        # Initialize LLMs
        logger.info(f"Initializing LLM: {llm_model_name}")
        self.llm = OpenAI(model_name=llm_model_name, temperature=temperature, openai_api_key=self.OPENAI_API_KEY)
        self.classifier_llm = OpenAI(model_name=llm_model_name, temperature=0.0, openai_api_key=self.OPENAI_API_KEY)
        
        # Define prompt templates
        self.classifier_template = PromptTemplate(
            input_variables=["query", "query_types"],
            template="""
            Classify the following fitness-related query into exactly one of these categories:
            {query_types}
            
            Query: {query}
            
            Respond with only the category name, nothing else.
            """
        )
        
        # Define specialized prompt templates for each query type
        self.prompt_templates = {
            "workout_routine": PromptTemplate(
                input_variables=["context", "question"],
                template="""
                You are a certified personal trainer specializing in workout routines. Use the following retrieved information to answer the question about workout routines.
                
                Retrieved information:
                {context}
                
                Question: {question}
                
                Provide a detailed workout routine based on the retrieved information. Include sets, reps, frequency, and progression advice when applicable.
                
                Answer:
                """
            ),
            "nutrition_diet": PromptTemplate(
                input_variables=["context", "question"],
                template="""
                You are a certified nutritionist specializing in fitness nutrition. Use the following retrieved information to answer the question about nutrition and diet.
                
                Retrieved information:
                {context}
                
                Question: {question}
                
                Provide detailed nutritional advice based on the retrieved information. Include macronutrient recommendations, meal timing, and food suggestions when applicable.
                
                Answer:
                """
            ),
            "exercise_technique": PromptTemplate(
                input_variables=["context", "question"],
                template="""
                You are a fitness coach specializing in exercise technique. Use the following retrieved information to answer the question about exercise form and technique.
                
                Retrieved information:
                {context}
                
                Question: {question}
                
                Provide detailed instructions on proper exercise technique based on the retrieved information. Include common mistakes to avoid and cues for proper form.
                
                Answer:
                """
            ),
            "fitness_equipment": PromptTemplate(
                input_variables=["context", "question"],
                template="""
                You are a fitness equipment specialist. Use the following retrieved information to answer the question about fitness equipment.
                
                Retrieved information:
                {context}
                
                Question: {question}
                
                Provide detailed information about fitness equipment based on the retrieved information. Include recommendations, comparisons, and usage advice when applicable.
                
                Answer:
                """
            ),
            "injury_prevention": PromptTemplate(
                input_variables=["context", "question"],
                template="""
                You are a sports medicine specialist focusing on injury prevention. Use the following retrieved information to answer the question about injury prevention or recovery.
                
                Retrieved information:
                {context}
                
                Question: {question}
                
                Provide detailed advice on injury prevention or recovery based on the retrieved information. Include warning signs, preventive measures, and recovery strategies when applicable.
                
                Answer:
                """
            ),
            "general_fitness": PromptTemplate(
                input_variables=["context", "question"],
                template="""
                You are a knowledgeable fitness trainer assistant. Use the following retrieved information to answer the general fitness question.
                
                Retrieved information:
                {context}
                
                Question: {question}
                
                Provide a comprehensive and accurate answer based on the retrieved information. If the information doesn't contain the answer, say "I don't have enough information to answer this question."
                
                Answer:
                """
            )
        }
        
        # Create LLM chains
        self.classifier_chain = LLMChain(llm=self.classifier_llm, prompt=self.classifier_template)
        self.answer_chains = {
            query_type: LLMChain(llm=self.llm, prompt=template)
            for query_type, template in self.prompt_templates.items()
        }
        
        logger.info("ModularRAG initialized successfully")
    
    def classify_query(self, query: str) -> str:
        """
        Classify the query into one of the predefined types.
        
        Args:
            query: The query string
            
        Returns:
            Query type
        """
        logger.info(f"Classifying query: {query}")
        
        # Format query types for the prompt
        query_types_str = "\n".join(self.QUERY_TYPES)
        
        # Classify query
        response = self.classifier_chain.run(query=query, query_types=query_types_str)
        
        # Clean and validate response
        predicted_type = response.strip().lower()
        
        # Default to general_fitness if classification fails
        if predicted_type not in self.QUERY_TYPES:
            logger.warning(f"Classification failed, defaulting to general_fitness. Got: {predicted_type}")
            predicted_type = "general_fitness"
        
        logger.info(f"Query classified as: {predicted_type}")
        return predicted_type
    
    def retrieve_documents(self, query: str, query_type: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from the vector database based on query type.
        
        Args:
            query: The query string
            query_type: The type of query
            
        Returns:
            A list of retrieved documents
        """
        logger.info(f"Retrieving documents for query type: {query_type}")
        
        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Add filter based on query type if available in metadata
        filter_dict = {}
        if query_type != "general_fitness":
            filter_dict = {"category": query_type}
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=self.top_k,
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )
        
        # Extract documents from results
        documents = []
        for match in results.matches:
            if hasattr(match, 'metadata') and match.metadata:
                documents.append({
                    "id": match.id,
                    "score": match.score,
                    "text": match.metadata.get("text", ""),
                    "source": match.metadata.get("source", "Unknown"),
                    "category": match.metadata.get("category", "Unknown")
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
            category_info = f" [Category: {doc.get('category', 'Unknown')}]" if 'category' in doc else ""
            context_parts.append(f"Document {i+1} [Source: {doc['source']}]{category_info}:\n{doc['text']}\n")
        
        return "\n".join(context_parts)
    
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
        query_type = self.classify_query(question)
        
        # Retrieve documents based on query type
        documents = self.retrieve_documents(question, query_type)
        
        if not documents:
            return "I couldn't find any relevant information to answer your question."
        
        # Format context
        context = self.format_context(documents)
        
        # Select appropriate chain based on query type
        answer_chain = self.answer_chains[query_type]
        
        # Generate answer
        response = answer_chain.run(context=context, question=question)
        
        return response.strip()


if __name__ == "__main__":
    # Example usage
    rag = ModularRAG()
    question = "What should I eat before a workout?"
    answer = rag.answer_question(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
