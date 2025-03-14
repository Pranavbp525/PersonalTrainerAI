"""
Modular RAG Implementation for PersonalTrainerAI

This module implements a Modular RAG approach that:
1. Classifies queries by intent
2. Uses specialized retrievers for different query types
3. Implements HyDE for complex queries
4. Uses multi-stage retrieval with feedback
5. Provides template-based response generation
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

class ModularRAG:
    """
    A modular RAG implementation with query classification, specialized retrievers,
    HyDE, multi-stage retrieval, and template-based response generation.
    """
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        llm_model_name: str = "gpt-3.5-turbo",
        top_k: int = 8
    ):
        """
        Initialize the ModularRAG system.
        
        Args:
            embedding_model_name: Name of the HuggingFace embedding model
            llm_model_name: Name of the LLM model
            top_k: Default number of documents to retrieve
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
        
        # Define query classification prompt
        self.query_classification_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            You are a fitness expert classifying user questions into categories.
            Classify the following question into exactly one of these categories:
            - workout_planning: Questions about creating workout routines or plans
            - exercise_technique: Questions about how to perform specific exercises
            - nutrition_advice: Questions about diet, supplements, or nutrition
            - progress_tracking: Questions about tracking progress or results
            - injury_prevention: Questions about preventing or managing injuries
            - general_fitness: General questions about fitness principles
            
            Respond with only the category name, nothing else.
            
            User question: {query}
            
            Category:
            """
        )
        
        # Create query classification chain
        self.query_classification_chain = LLMChain(llm=self.llm, prompt=self.query_classification_prompt)
        
        # Define HyDE prompt
        self.hyde_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            You are a fitness expert. Generate a detailed, hypothetical passage that would contain the answer to the following question.
            The passage should be factually accurate and comprehensive, as if extracted from a fitness textbook or research paper.
            
            Question: {query}
            
            Hypothetical passage:
            """
        )
        
        # Create HyDE chain
        self.hyde_chain = LLMChain(llm=self.llm, prompt=self.hyde_prompt)
        
        # Define response templates for different query types
        self.response_templates = {
            "workout_planning": PromptTemplate(
                input_variables=["query", "context"],
                template="""
                You are a professional personal trainer creating a workout plan. Use the following retrieved information to create a detailed workout plan for the user.
                
                Guidelines:
                - Create a structured workout plan with clear sections
                - Include specific exercises, sets, reps, and rest periods
                - Explain the purpose of each exercise or workout component
                - Provide progression options for different fitness levels
                - Include warm-up and cool-down recommendations
                - Add safety tips and form cues where appropriate
                
                Retrieved information:
                {context}
                
                User question: {query}
                
                Workout Plan:
                """
            ),
            "exercise_technique": PromptTemplate(
                input_variables=["query", "context"],
                template="""
                You are a professional personal trainer explaining exercise technique. Use the following retrieved information to provide detailed instructions on proper exercise form.
                
                Guidelines:
                - Break down the movement into clear steps
                - Highlight common form mistakes and how to avoid them
                - Explain muscle engagement and biomechanics
                - Provide cues that help with proper execution
                - Include modifications for different fitness levels if relevant
                - Add safety considerations
                
                Retrieved information:
                {context}
                
                User question: {query}
                
                Exercise Technique:
                """
            ),
            "nutrition_advice": PromptTemplate(
                input_variables=["query", "context"],
                template="""
                You are a fitness nutrition specialist. Use the following retrieved information to provide evidence-based nutrition advice.
                
                Guidelines:
                - Provide scientifically-backed nutritional recommendations
                - Explain the reasoning behind the advice
                - Include practical implementation tips
                - Acknowledge different dietary preferences when relevant
                - Mention timing considerations if applicable
                - Clarify how the nutrition advice supports fitness goals
                
                Retrieved information:
                {context}
                
                User question: {query}
                
                Nutrition Advice:
                """
            ),
            "default": PromptTemplate(
                input_variables=["query", "context", "category"],
                template="""
                You are a knowledgeable personal fitness trainer assistant specializing in {category}. Use the following retrieved information to answer the user's question.
                
                Guidelines:
                - Provide a comprehensive and detailed answer
                - Include specific recommendations when appropriate
                - Explain the scientific reasoning behind your advice when possible
                - Format your response in a clear, structured way
                - Cite sources when possible
                
                Retrieved information:
                {context}
                
                User question: {query}
                
                Your answer:
                """
            )
        }
        
        logger.info("ModularRAG initialized successfully")
    
    def classify_query(self, query: str) -> str:
        """
        Classify the query into a specific category.
        
        Args:
            query: User query
            
        Returns:
            Query category
        """
        logger.info(f"Classifying query: {query}")
        
        # Get classification from LLM
        category = self.query_classification_chain.run(query=query).strip().lower()
        
        # Validate category
        valid_categories = ["workout_planning", "exercise_technique", "nutrition_advice", 
                           "progress_tracking", "injury_prevention", "general_fitness"]
        
        if category not in valid_categories:
            logger.warning(f"Invalid category: {category}. Using default.")
            category = "general_fitness"
        
        logger.info(f"Query classified as: {category}")
        return category
    
    def generate_hyde_document(self, query: str) -> str:
        """
        Generate a hypothetical document that might contain the answer to the query.
        
        Args:
            query: User query
            
        Returns:
            Hypothetical document text
        """
        logger.info(f"Generating HyDE document for query: {query}")
        
        # Generate hypothetical document
        hyde_doc = self.hyde_chain.run(query=query)
        
        logger.info("HyDE document generated")
        return hyde_doc
    
    def retrieve_with_hyde(self, query: str, category: str) -> List[Dict[str, Any]]:
        """
        Retrieve documents using HyDE for complex queries.
        
        Args:
            query: User query
            category: Query category
            
        Returns:
            List of retrieved documents
        """
        logger.info(f"Retrieving with HyDE for category: {category}")
        
        # Generate hypothetical document
        hyde_doc = self.generate_hyde_document(query)
        
        # Embed the hypothetical document
        hyde_embedding = self.embedding_model.embed_query(hyde_doc)
        
        # Search Pinecone with HyDE embedding
        results = self.index.query(
            vector=hyde_embedding,
            top_k=self.top_k,
            include_metadata=True
        )
        
        logger.info(f"Retrieved {len(results['matches'])} documents with HyDE")
        return results['matches']
    
    def retrieve_with_metadata_filter(self, query: str, category: str) -> List[Dict[str, Any]]:
        """
        Retrieve documents with metadata filtering based on category.
        
        Args:
            query: User query
            category: Query category
            
        Returns:
            List of retrieved documents
        """
        logger.info(f"Retrieving with metadata filter for category: {category}")
        
        # Map category to metadata filter
        category_filters = {
            "workout_planning": {"source": "workout_plans"},
            "exercise_technique": {"source": "exercise_guides"},
            "nutrition_advice": {"source": "nutrition"},
            # Add more filters as needed
        }
        
        # Get filter for category if available
        filter_dict = category_filters.get(category, None)
        
        # Embed query
        query_embedding = self.embedding_model.embed_query(query)
        
        # Search Pinecone with filter if available
        if filter_dict:
            results = self.index.query(
                vector=query_embedding,
                top_k=self.top_k,
                include_metadata=True,
                filter=filter_dict
            )
        else:
            results = self.index.query(
                vector=query_embedding,
                top_k=self.top_k,
                include_metadata=True
            )
        
        logger.info(f"Retrieved {len(results['matches'])} documents with metadata filter")
        return results['matches']
    
    def multi_stage_retrieval(self, query: str, category: str) -> List[Dict[str, Any]]:
        """
        Perform multi-stage retrieval based on query category.
        
        Args:
            query: User query
            category: Query category
            
        Returns:
            List of retrieved documents
        """
        logger.info(f"Performing multi-stage retrieval for category: {category}")
        
        # Complex categories use HyDE
        complex_categories = ["workout_planning", "nutrition_advice"]
        
        if category in complex_categories:
            # Use HyDE for complex queries
            documents = self.retrieve_with_hyde(query, category)
        else:
            # Use metadata filtering for other queries
            documents = self.retrieve_with_metadata_filter(query, category)
        
        # If we didn't get enough results, fall back to regular retrieval
        if len(documents) < 3:
            logger.info("Insufficient results, falling back to regular retrieval")
            query_embedding = self.embedding_model.embed_query(query)
            results = self.index.query(
                vector=query_embedding,
                top_k=self.top_k,
                include_metadata=True
            )
            documents = results['matches']
        
        return documents
    
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
            
            # Format with source information
            source_info = f"{source}"
            if url:
                source_info += f" ({url})"
                
            context_part = f"Document {i+1}:\nTitle: {title}\nSource: {source_info}\nContent: {text}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def generate_response(self, query: str, context: str, category: str) -> str:
        """
        Generate a response using the appropriate template based on query category.
        
        Args:
            query: User query
            context: Retrieved context
            category: Query category
            
        Returns:
            Generated response
        """
        logger.info(f"Generating response for category: {category}")
        
        # Get appropriate template
        template = self.response_templates.get(category, self.response_templates["default"])
        
        # Create chain with template
        if category in self.response_templates:
            chain = LLMChain(llm=self.llm, prompt=template)
            response = chain.run(query=query, context=context)
        else:
            chain = LLMChain(llm=self.llm, prompt=template)
            response = chain.run(query=query, context=context, category=category)
        
        return response
    
    def query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query through the modular RAG pipeline.
        
        Args:
            query: User query
            
        Returns:
            Dictionary containing the query, category, retrieved documents, and generated response
        """
        logger.info(f"Processing query: {query}")
        
        # C<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>