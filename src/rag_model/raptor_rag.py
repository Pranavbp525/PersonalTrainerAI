"""
RAPTOR RAG Implementation for PersonalTrainerAI

This module implements a RAPTOR (Retrieval Augmented Prompt Optimization and Reasoning) RAG approach
that uses a multi-step reasoning process with iterative retrieval to handle complex fitness questions.

Key features:
1. Query planning and decomposition
2. Iterative, multi-step retrieval
3. Reasoning over retrieved information
4. Self-reflection and refinement
5. Structured response synthesis
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional, Set, Tuple
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

class RAPTORRAG:
    """
    A RAPTOR RAG implementation that uses a multi-step reasoning process with
    iterative retrieval for complex fitness questions.
    """
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        llm_model_name: str = "gpt-3.5-turbo",
        reasoning_llm_model_name: str = "gpt-4",
        top_k: int = 5,
        max_iterations: int = 3
    ):
        """
        Initialize the RAPTOR RAG system.
        
        Args:
            embedding_model_name: Name of the HuggingFace embedding model
            llm_model_name: Name of the LLM model for basic tasks
            reasoning_llm_model_name: Name of the LLM model for reasoning tasks (typically more capable)
            top_k: Number of documents to retrieve per sub-question
            max_iterations: Maximum number of reasoning iterations
        """
        self.top_k = top_k
        self.max_iterations = max_iterations
        
        # Initialize embedding model
        logger.info(f"Initializing embedding model: {embedding_model_name}")
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        # Initialize Pinecone
        logger.info("Connecting to Pinecone")
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        self.index = pinecone.Index(INDEX_NAME)
        
        # Initialize LLMs
        logger.info(f"Initializing base LLM: {llm_model_name}")
        self.llm = OpenAI(model_name=llm_model_name, temperature=0.7, api_key=OPENAI_API_KEY)
        
        logger.info(f"Initializing reasoning LLM: {reasoning_llm_model_name}")
        self.reasoning_llm = OpenAI(model_name=reasoning_llm_model_name, temperature=0.2, api_key=OPENAI_API_KEY)
        
        # Define query planning prompt
        self.query_planning_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            You are a fitness expert planning how to answer a complex question about fitness, exercise, nutrition, or training.
            Break down the following question into 2-4 specific sub-questions that would help you provide a comprehensive answer.
            
            For each sub-question:
            1. Make it specific and focused on retrieving particular information
            2. Ensure it's directly relevant to the original question
            3. Phrase it as a standalone question
            
            Format your response as a JSON array of sub-questions.
            
            Original question: {query}
            
            Sub-questions:
            """
        )
        
        # Define reasoning prompt
        self.reasoning_prompt = PromptTemplate(
            input_variables=["query", "sub_questions", "retrieved_info", "current_reasoning"],
            template="""
            You are a fitness expert reasoning through a complex fitness question.
            
            Original question: {query}
            
            Sub-questions we've explored:
            {sub_questions}
            
            Information we've retrieved:
            {retrieved_info}
            
            Current reasoning (if any):
            {current_reasoning}
            
            Based on the above information, continue reasoning about the original question.
            Identify connections between different pieces of information, potential contradictions,
            and insights that help answer the original question.
            
            If you need additional information, specify what else you need to know.
            
            Your reasoning:
            """
        )
        
        # Define response synthesis prompt
        self.synthesis_prompt = PromptTemplate(
            input_variables=["query", "reasoning", "retrieved_info"],
            template="""
            You are a knowledgeable personal fitness trainer assistant. Synthesize a comprehensive answer to the user's question
            based on the reasoning and information provided.
            
            Guidelines:
            - Provide a detailed, well-structured answer
            - Include specific exercises, techniques, or training principles when relevant
            - Explain the scientific reasoning behind recommendations
            - Format your response with clear headings and bullet points where appropriate
            - Acknowledge any limitations or areas where more personalized information might be needed
            
            User question: {query}
            
            Reasoning process:
            {reasoning}
            
            Retrieved information:
            {retrieved_info}
            
            Your comprehensive answer:
            """
        )
        
        # Create chains
        self.query_planning_chain = LLMChain(llm=self.reasoning_llm, prompt=self.query_planning_prompt)
        self.reasoning_chain = LLMChain(llm=self.reasoning_llm, prompt=self.reasoning_prompt)
        self.synthesis_chain = LLMChain(llm=self.reasoning_llm, prompt=self.synthesis_prompt)
        
        logger.info("RAPTOR RAG initialized successfully")
    
    def plan_query(self, query: str) -> List[str]:
        """
        Break down a complex query into sub-questions.
        
        Args:
            query: Original user query
            
        Returns:
            List of sub-questions
        """
        logger.info(f"Planning query: {query}")
        
        try:
            # Generate sub-questions
            result = self.query_planning_chain.run(query=query)
            sub_questions = json.loads(result)
            logger.info(f"Generated {len(sub_questions)} sub-questions")
            return sub_questions
        except Exception as e:
            logger.error(f"Error planning query: {e}")
            # Return a simple breakdown as fallback
            return [
                f"What are the key concepts related to {query}?",
                f"What are the best practices for {query}?"
            ]
    
    def retrieve_for_question(self, question: str) -> List[Dict[str, Any]]:
        """
        Retrieve documents for a specific question.
        
        Args:
            question: Question to retrieve documents for
            
        Returns:
            List of retrieved documents
        """
        logger.info(f"Retrieving documents for: {question}")
        
        # Generate question embedding
        question_embedding = self.embedding_model.embed_query(question)
        
        # Search Pinecone
        results = self.index.query(
            vector=question_embedding,
            top_k=self.top_k,
            include_metadata=True
        )
        
        return results['matches']
    
    def extract_document_text(self, documents: List[Dict[str, Any]]) -> str:
        """
        Extract text from retrieved documents.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Concatenated document text
        """
        text = ""
        for i, doc in enumerate(documents):
            metadata = doc.get('metadata', {})
            doc_text = metadata.get('text', '')
            if doc_text:
                text += f"Document {i+1}:\n{doc_text}\n\n"
        
        return text
    
    def reason_iteratively(self, query: str, sub_questions: List[str]) -> Tuple[str, Dict[str, str]]:
        """
        Perform iterative reasoning over retrieved information.
        
        Args:
            query: Original user query
            sub_questions: List of sub-questions
            
        Returns:
            Tuple of (final reasoning, retrieved information by question)
        """
        logger.info("Starting iterative reasoning process")
        
        # Initialize
        current_reasoning = ""
        retrieved_info_by_question = {}
        
        # First, retrieve information for each sub-question
        for question in sub_questions:
            documents = self.retrieve_for_question(question)
            retrieved_text = self.extract_document_text(documents)
            retrieved_info_by_question[question] = retrieved_text
        
        # Format sub-questions and retrieved info for the prompt
        sub_questions_text = "\n".join([f"- {q}" for q in sub_questions])
        retrieved_info_text = ""
        for question, info in retrieved_info_by_question.items():
            retrieved_info_text += f"For question: {question}\n{info}\n\n"
        
        # Perform iterative reasoning
        for i in range(self.max_iterations):
            logger.info(f"Reasoning iteration {i+1}/{self.max_iterations}")
            
            # Run reasoning step
            reasoning_result = self.reasoning_chain.run(
                query=query,
                sub_questions=sub_questions_text,
                retrieved_info=retrieved_info_text,
                current_reasoning=current_reasoning
            )
            
            # Update current reasoning
            if i == 0:
                current_reasoning = reasoning_result
            else:
                current_reasoning += f"\n\nIteration {i+1}:\n{reasoning_result}"
            
            # Check if additional information is requested
            if "need additional information" not in reasoning_result.lower() and "need more information" not in reasoning_result.lower():
                break
        
        return current_reasoning, retrieved_info_by_question
    
    def synthesize_response(self, query: str, reasoning: str, retrieved_info: Dict[str, str]) -> str:
        """
        Synthesize a final response based on reasoning and retrieved information.
        
        Args:
            query: Original user query
            reasoning: Reasoning process
            retrieved_info: Retrieved information by question
            
        Returns:
            Synthesized response
        """
        logger.info("Synthesizing final response")
        
        # Format retrieved info for the prompt
        retrieved_info_text = ""
        for question, info in retrieved_info.items():
            retrieved_info_text += f"For question: {question}\n{info}\n\n"
        
        # Generate response
        response = self.synthesis_chain.run(
            query=query,
            reasoning=reasoning,
            retrieved_info=retrieved_info_text
        )
        
        return response
    
    def query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query using the RAPTOR RAG approach.
        
        Args:
            query: User query
            
        Returns:
            Dictionary containing the query results
        """
        logger.info(f"Processing query with RAPTOR RAG: {query}")
        
        try:
            # Step 1: Plan the query by breaking it down into sub-questions
            sub_questions = self.plan_query(query)
            
            # Step 2: Perform iterative reasoning over retrieved information
            reasoning, retrieved_info = self.reason_iteratively(query, sub_questions)
            
            # Step 3: Synthesize the final response
            response = self.synthesize_response(query, reasoning, retrieved_info)
            
            return {
                "query": query,
                "response": response,
                "sub_questions": sub_questions,
                "reasoning": reasoning
            }
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "query": query,
                "error": str(e)
            }


# Example usage
if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='RAPTOR RAG for PersonalTrainerAI')
    parser.add_argument('--query', type=str,
                        help='Query to process (if not provided, interactive mode is used)')
    
    args = parser.parse_args()
    
    # Initialize RAPTOR RAG
    raptor_rag = RAPTORRAG()
    
    # Process query if provided
    if args.query:
        result = raptor_rag.query(args.query)
        print(f"\nQuery: {result['query']}")
        print(f"\nResponse: {result['response']}")
    else:
        # Interactive mode
        print(f"\nPersonalTrainerAI RAPTOR RAG")
        print("Type 'exit' to quit")
        
        while True:
            query = input("\nEnter your fitness question: ")
            
            if query.lower() in ['exit', 'quit', 'q']:
                break
            
            result = raptor_rag.query(query)
            
            if 'error' in result:
                print(f"\nError: {result['error']}")
            else:
                print(f"\nResponse: {result['response']}")
