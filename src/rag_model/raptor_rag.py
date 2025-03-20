"""
RAPTOR RAG Implementation for PersonalTrainerAI

This module implements a RAPTOR (Retrieval Augmented Prompt Tuning and Optimization with Reasoning)
approach for fitness knowledge.
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

class RaptorRAG:
    """
    A RAPTOR (Retrieval Augmented Prompt Tuning and Optimization with Reasoning) implementation
    for fitness knowledge.
    
    This implementation includes:
    - Multi-step reasoning
    - Prompt optimization
    - Self-reflection and correction
    """
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        llm_model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        top_k: int = 8
    ):
        """
        Initialize the RaptorRAG system.
        
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
        self.reasoning_llm = OpenAI(model_name=llm_model_name, temperature=0.3, openai_api_key=self.OPENAI_API_KEY)
        self.reflection_llm = OpenAI(model_name=llm_model_name, temperature=0.0, openai_api_key=self.OPENAI_API_KEY)
        
        # Define prompt templates
        self.retrieval_analysis_template = PromptTemplate(
            input_variables=["question", "documents"],
            template="""
            Analyze the following retrieved documents in relation to the fitness question.
            
            Question: {question}
            
            Retrieved documents:
            {documents}
            
            For each document, assess:
            1. Relevance to the question (scale 1-10)
            2. Key information it provides
            3. Any gaps or contradictions with other documents
            
            Then, identify what additional information might be needed to fully answer the question.
            
            Analysis:
            """
        )
        
        self.reasoning_template = PromptTemplate(
            input_variables=["question", "analysis", "documents"],
            template="""
            Based on the retrieved documents and analysis, reason through how to best answer the fitness question.
            
            Question: {question}
            
            Document Analysis:
            {analysis}
            
            Retrieved documents:
            {documents}
            
            Think step by step:
            1. What are the key concepts needed to answer this question?
            2. What evidence from the documents supports each concept?
            3. How should these concepts be organized in a comprehensive answer?
            4. Are there any cautions, exceptions, or personalization factors to consider?
            
            Reasoning:
            """
        )
        
        self.answer_template = PromptTemplate(
            input_variables=["question", "reasoning", "documents"],
            template="""
            You are a knowledgeable fitness trainer assistant. Use the following reasoning and retrieved information to answer the question.
            
            Question: {question}
            
            Reasoning process:
            {reasoning}
            
            Retrieved information:
            {documents}
            
            Provide a comprehensive, accurate, and well-structured answer based on the reasoning and retrieved information.
            Include specific details from the documents when relevant.
            If the information doesn't contain the answer, acknowledge the limitations of your knowledge.
            
            Answer:
            """
        )
        
        self.reflection_template = PromptTemplate(
            input_variables=["question", "answer", "documents"],
            template="""
            Critically evaluate the following answer to a fitness question.
            
            Question: {question}
            
            Answer: {answer}
            
            Retrieved documents:
            {documents}
            
            Evaluate the answer on:
            1. Accuracy (Does it align with the retrieved documents?)
            2. Completeness (Does it address all aspects of the question?)
            3. Clarity (Is it well-structured and easy to understand?)
            4. Evidence (Does it properly cite information from the documents?)
            
            Identify any issues that need correction.
            
            Reflection:
            """
        )
        
        self.correction_template = PromptTemplate(
            input_variables=["question", "answer", "reflection", "documents"],
            template="""
            You are a knowledgeable fitness trainer assistant. Revise the following answer based on critical reflection.
            
            Question: {question}
            
            Original answer: {answer}
            
            Critical reflection:
            {reflection}
            
            Retrieved documents:
            {documents}
            
            Provide an improved answer that addresses the issues identified in the reflection.
            The answer should be comprehensive, accurate, well-structured, and properly supported by the retrieved information.
            
            Improved answer:
            """
        )
        
        # Create LLM chains
        self.retrieval_analysis_chain = LLMChain(llm=self.reasoning_llm, prompt=self.retrieval_analysis_template)
        self.reasoning_chain = LLMChain(llm=self.reasoning_llm, prompt=self.reasoning_template)
        self.answer_chain = LLMChain(llm=self.llm, prompt=self.answer_template)
        self.reflection_chain = LLMChain(llm=self.reflection_llm, prompt=self.reflection_template)
        self.correction_chain = LLMChain(llm=self.llm, prompt=self.correction_template)
        
        logger.info("RaptorRAG initialized successfully")
    
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
    
    def format_documents(self, documents: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into a string.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted documents string
        """
        doc_parts = []
        for i, doc in enumerate(documents):
            doc_parts.append(f"Document {i+1} [Source: {doc['source']}, Relevance: {doc['score']:.2f}]:\n{doc['text']}\n")
        
        return "\n".join(doc_parts)
    
    def answer_question(self, question: str) -> str:
        """
        Answer a question using the RAPTOR RAG approach.
        
        Args:
            question: The question to answer
            
        Returns:
            Generated answer
        """
        logger.info(f"Answering question: {question}")
        
        # Step 1: Retrieve documents
        documents = self.retrieve_documents(question)
        
        if not documents:
            return "I couldn't find any relevant information to answer your question."
        
        # Format documents
        formatted_docs = self.format_documents(documents)
        
        # Step 2: Analyze retrieved documents
        logger.info("Analyzing retrieved documents")
        analysis = self.retrieval_analysis_chain.run(
            question=question,
            documents=formatted_docs
        )
        
        # Step 3: Reasoning
        logger.info("Performing reasoning")
        reasoning = self.reasoning_chain.run(
            question=question,
            analysis=analysis,
            documents=formatted_docs
        )
        
        # Step 4: Generate initial answer
        logger.info("Generating initial answer")
        initial_answer = self.answer_chain.run(
            question=question,
            reasoning=reasoning,
            documents=formatted_docs
        )
        
        # Step 5: Reflection
        logger.info("Reflecting on answer")
        reflection = self.reflection_chain.run(
            question=question,
            answer=initial_answer,
            documents=formatted_docs
        )
        
        # Step 6: Correction
        logger.info("Applying corrections")
        final_answer = self.correction_chain.run(
            question=question,
            answer=initial_answer,
            reflection=reflection,
            documents=formatted_docs
        )
        
        return final_answer.strip()


if __name__ == "__main__":
    # Example usage
    rag = RaptorRAG()
    question = "Design a weekly workout plan for weight loss and muscle toning."
    answer = rag.answer_question(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
