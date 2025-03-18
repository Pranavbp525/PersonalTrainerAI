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

class ModularRAG:
    """
    Modular Retrieval Augmented Generation for fitness knowledge.
    
    This implementation uses a modular approach with query classification,
    specialized retrievers, and template-based response generation.
    """
    
    def __init__(
        self,
        vector_db_name: str = "fitness-knowledge",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.2,
        max_tokens: int = 500
    ):
        """
        Initialize the Modular RAG system.
        
        Args:
            vector_db_name: Name of the Pinecone vector database
            embedding_model_name: Name of the embedding model to use
            llm_model_name: Name of the language model to use
            temperature: Temperature parameter for the LLM
            max_tokens: Maximum tokens for LLM response
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
        
        # Define query classifiers and specialized retrievers
        self._setup_query_classifiers()
        self._setup_specialized_retrievers()
        self._setup_response_templates()

    def _setup_query_classifiers(self):
        """Set up query classification components."""
        # Query classification prompt
        self.query_classifier_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            Classify the following fitness-related query into exactly one of these categories:
            - workout_plan: Questions about workout routines, exercise plans, or training schedules
            - exercise_technique: Questions about how to perform specific exercises correctly
            - nutrition: Questions about diet, supplements, or nutritional advice
            - fitness_goal: Questions about achieving specific fitness goals
            - equipment: Questions about fitness equipment or gym tools
            - recovery: Questions about rest, recovery, or injury prevention
            - general: Other general fitness questions
            
            Query: {query}
            
            Category (just return the category name):
            """
        )
        
        # Create query classifier chain
        self.query_classifier_chain = LLMChain(
            llm=self.llm,
            prompt=self.query_classifier_prompt
        )

    def _setup_specialized_retrievers(self):
        """Set up specialized retrievers for different query types."""
        # Define specialized retrieval parameters for each query type
        self.retrieval_params = {
            "workout_plan": {
                "top_k": 7,
                "filter": {"type": "workout"},
                "hyde_enabled": True
            },
            "exercise_technique": {
                "top_k": 5,
                "filter": {"type": "exercise"},
                "hyde_enabled": True
            },
            "nutrition": {
                "top_k": 6,
                "filter": {"type": "nutrition"},
                "hyde_enabled": False
            },
            "fitness_goal": {
                "top_k": 8,
                "filter": None,  # No filter to get diverse information
                "hyde_enabled": True
            },
            "equipment": {
                "top_k": 4,
                "filter": {"type": "equipment"},
                "hyde_enabled": False
            },
            "recovery": {
                "top_k": 5,
                "filter": {"type": "recovery"},
                "hyde_enabled": False
            },
            "general": {
                "top_k": 6,
                "filter": None,
                "hyde_enabled": False
            }
        }
        
        # HyDE prompt for generating hypothetical documents
        self.hyde_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            You are an expert fitness trainer and nutritionist. 
            Write a detailed and informative passage that would be the perfect answer to the following question.
            Include specific details, examples, and explanations.
            
            Question: {query}
            
            Detailed answer:
            """
        )
        
        # Create HyDE chain
        self.hyde_chain = LLMChain(
            llm=self.llm,
            prompt=self.hyde_prompt
        )

    def _setup_response_templates(self):
        """Set up response templates for different query types."""
        # Define response templates for each query type
        self.response_templates = {
            "workout_plan": PromptTemplate(
                input_variables=["context", "query"],
                template="""
                You are a professional fitness trainer specializing in creating personalized workout plans.
                Use the following retrieved information to create a detailed workout plan that addresses the user's query.
                Include specific exercises, sets, reps, and rest periods where appropriate.
                
                Retrieved information:
                {context}
                
                User query: {query}
                
                Detailed workout plan:
                """
            ),
            "exercise_technique": PromptTemplate(
                input_variables=["context", "query"],
                template="""
                You are a fitness technique expert specializing in proper exercise form and execution.
                Use the following retrieved information to explain the correct technique for the exercise in question.
                Include step-by-step instructions, common mistakes to avoid, and form cues.
                
                Retrieved information:
                {context}
                
                User query: {query}
                
                Detailed technique explanation:
                """
            ),
            "nutrition": PromptTemplate(
                input_variables=["context", "query"],
                template="""
                You are a certified nutritionist specializing in fitness nutrition.
                Use the following retrieved information to provide evidence-based nutritional advice.
                Include specific recommendations, scientific explanations, and practical tips.
                
                Retrieved information:
                {context}
                
                User query: {query}
                
                Nutritional advice:
                """
            ),
            "fitness_goal": PromptTemplate(
                input_variables=["context", "query"],
                template="""
                You are a fitness coach specializing in helping clients achieve specific fitness goals.
                Use the following retrieved information to create a comprehensive plan to help the user reach their goal.
                Include training, nutrition, and lifestyle recommendations.
                
                Retrieved information:
                {context}
                
                User query: {query}
                
                Goal achievement plan:
                """
            ),
            "equipment": PromptTemplate(
                input_variables=["context", "query"],
                template="""
                You are a fitness equipment specialist with extensive knowledge of gym tools and home workout equipment.
                Use the following retrieved information to provide detailed information about the equipment in question.
                Include specifications, usage instructions, and recommendations.
                
                Retrieved information:
                {context}
                
                User query: {query}
                
                Equipment information:
                """
            ),
            "recovery": PromptTemplate(
                input_variables=["context", "query"],
                template="""
                You are a recovery and sports medicine specialist focusing on optimal recovery strategies.
                Use the following retrieved information to provide advice on recovery, rest, or injury prevention.
                Include scientific explanations and practical recommendations.
                
                Retrieved information:
                {context}
                
                User query: {query}
                
                Recovery advice:
                """
            ),
            "general": PromptTemplate(
                input_variables=["context", "query"],
                template="""
                You are a knowledgeable fitness expert with broad expertise across training, nutrition, and health.
                Use the following retrieved information to provide a comprehensive answer to the user's question.
                
                Retrieved information:
                {context}
                
                User query: {query}
                
                Answer:
                """
            )
        }

    def classify_query(self, query: str) -> str:
        """
        Classify the user query into a predefined category.
        
        Args:
            query: User query
            
        Returns:
            Query category
        """
        try:
            category = self.query_classifier_chain.run(query=query).strip().lower()
            
            # Ensure the category is valid
            if category not in self.retrieval_params:
                logger.warning(f"Invalid category: {category}. Defaulting to 'general'.")
                category = "general"
            
            logger.info(f"Query classified as: {category}")
            return category
        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            return "general"

    def generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical document using HyDE technique.
        
        Args:
            query: User query
            
        Returns:
            Hypothetical document text
        """
        try:
            hypothetical_doc = self.hyde_chain.run(query=query)
            logger.info("Generated hypothetical document for HyDE")
            return hypothetical_doc
        except Exception as e:
            logger.error(f"Error generating hypothetical document: {e}")
            return ""

    def retrieve_documents(
        self, 
        query: str, 
        category: str,
        use_hyde: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using the appropriate retriever for the query category.
        
        Args:
            query: User query
            category: Query category
            use_hyde: Whether to use HyDE technique
            
        Returns:
            List of retrieved documents
        """
        # Get retrieval parameters for the category
        params = self.retrieval_params.get(category, self.retrieval_params["general"])
        top_k = params["top_k"]
        metadata_filter = params["filter"]
        
        # Use HyDE if enabled for this category and requested
        if use_hyde and params["hyde_enabled"]:
            # Generate hypothetical document
            hyde_doc = self.generate_hypothetical_document(query)
            
            # Use hypothetical document for retrieval
            query_embedding = self.embeddings.embed_query(hyde_doc)
        else:
            # Use original query for retrieval
            query_embedding = self.embeddings.embed_query(query)
        
        # Prepare filter if any
        filter_dict = metadata_filter if metadata_filter else {}
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            filter=filter_dict,
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
        
        logger.info(f"Retrieved {len(documents)} documents for category: {category}")
        return documents

    def rerank_documents(
        self, 
        documents: List[Dict[str, Any]], 
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Rerank retrieved documents based on relevance to the query.
        
        Args:
            documents: List of retrieved documents
            query: User query
            
        Returns:
            Reranked list of documents
        """
        if not documents:
            return []
        
        # Use LLM to score document relevance
        rerank_prompt = PromptTemplate(
            input_variables=["query", "document"],
            template="""
            On a scale of 1 to 10, rate how relevant the following document is to the query.
            Only respond with a number from 1 to 10.
            
            Query: {query}
            
            Document: {document}
            
            Relevance score (1-10):
            """
        )
        
        rerank_chain = LLMChain(
            llm=self.llm,
            prompt=rerank_prompt
        )
        
        # Score each document
        scored_docs = []
        for doc in documents:
            try:
                score_text = rerank_chain.run(
                    query=query,
                    document=doc["text"]
                ).strip()
                
                # Extract numeric score
                try:
                    score = float(score_text)
                except ValueError:
                    # If we can't parse the score, use the original similarity score
                    score = doc["score"] * 10
                
                scored_docs.append((doc, score))
            except Exception as e:
                logger.error(f"Error scoring document: {e}")
                # Keep the document with its original score
                scored_docs.append((doc, doc["score"] * 10))
        
        # Sort by score in descending order
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return reranked documents
        return [doc for doc, _ in scored_docs]

    def generate_response(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        category: str
    ) -> str:
        """
        Generate a response using the appropriate template for the query category.
        
        Args:
            query: User query
            documents: List of retrieved documents
            category: Query category
            
        Returns:
            Generated response
        """
        # Get response template for the category
        template = self.response_templates.get(category, self.response_templates["general"])
        
        # Combine document texts for context
        context = "\n\n".join([f"Document {i+1}:\n{doc['text']}" for i, doc in enumerate(documents)])
        
        # Generate response
        try:
            response_chain = LLMChain(
                llm=self.llm,
                prompt=template
            )
            
            response = response_chain.run(
                context=context,
                query=query
            )
            
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while generating a response. Please try asking your question again."

    def answer_question(self, question: str) -> str:
        """
        Answer a fitness-related question using Modular RAG.
        
        Args:
            question: User question
            
        Returns:
            Answer to the question
        """
        # Step 1: Classify the query
        category = self.classify_query(question)
        
        # Step 2: Retrieve relevant documents
        params = self.retrieval_params[category]
        documents = self.retrieve_documents(
            query=question,
            category=category,
            use_hyde=params["hyde_enabled"]
        )
        
        # Step 3: Rerank documents if we have more than 3
        if len(documents) > 3:
            documents = self.rerank_documents(documents, question)
        
        # Step 4: Generate response using appropriate template
        response = self.generate_response(question, documents, category)
        
        return response

    def multi_stage_retrieval(self, question: str) -> str:
        """
        Perform multi-stage retrieval with feedback for complex questions.
        
        Args:
            question: User question
            
        Returns:
            Answer to the question
        """
        # Step 1: Classify the query
        category = self.classify_query(question)
        
        # Step 2: Initial retrieval
        documents = self.retrieve_documents(
            query=question,
            category=category,
            use_hyde=self.retrieval_params[category]["hyde_enabled"]
        )
        
        # Step 3: Analyze if we need more information
        analysis_prompt = PromptTemplate(
            input_variables=["question", "retrieved_info"],
            template="""
            Analyze if the retrieved information is sufficient to answer the user's question.
            If not, identify what specific additional information is needed.
            
            User question: {question}
            
            Retrieved information:
            {retrieved_info}
            
            Is this information sufficient? Answer YES or NO.
            If NO, what specific additional information is needed?
            """
        )
        
        analysis_chain = LLMChain(
            llm=self.llm,
            prompt=analysis_prompt
        )
        
        retrieved_info = "\n\n".join([doc["text"] for doc in documents])
        analysis = analysis_chain.run(
            question=question,
            retrieved_info=retrieved_info
        )
        
        # Step 4: Perform additional retrieval if needed
        if "NO" in analysis.upper():
            # Extract the missing information needed
            missing_info = analysis.split("NO")[-1].strip()
            
            # Generate a new query to find the missing information
            query_gen_prompt = PromptTemplate(
                input_variables=["original_question", "missing_info"],
                template="""
                Based on the original question and the identified missing information,
                formulate a new search query to find the specific missing information.
                
                Original question: {original_question}
                Missing information: {missing_info}
                
                New search query:
                """
            )
            
            query_gen_chain = LLMChain(
                llm=self.llm,
                prompt=query_gen_prompt
            )
            
            new_query = query_gen_chain.run(
                original_question=question,
                missing_info=missing_info
            )
            
            # Retrieve additional documents
            additional_docs = self.retrieve_documents(
                query=new_query,
                category=category,
                use_hyde=False
            )
            
            # Combine all documents
            all_docs = documents + additional_docs
            
            # Rerank the combined set
            documents = self.rerank_documents(all_docs, question)
        
        # Step 5: Generate final response
        response = self.generate_response(question, documents, category)
        
        return response
