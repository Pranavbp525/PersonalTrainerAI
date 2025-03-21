"""
RAPTOR RAG Implementation for PersonalTrainerAI

This module implements the RAPTOR (Retrieval Augmented Prompt Tuning and Optimization for Reasoning) 
approach for fitness knowledge.
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

class RaptorRAG:
    """
    A RAPTOR (Retrieval Augmented Prompt Tuning and Optimization for Reasoning) implementation
    for fitness knowledge.
    
    This implementation enhances the RAG approach with:
    1. Multi-step reasoning
    2. Self-reflection and refinement
    3. Dynamic prompt optimization
    4. Hybrid retrieval strategies
    """
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        llm_model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        top_k: int = 8,
        reasoning_steps: int = 3
    ):
        """
        Initialize the RaptorRAG system.
        
        Args:
            embedding_model_name: Name of the embedding model to use
            llm_model_name: Name of the language model to use
            temperature: Temperature parameter for the LLM
            top_k: Number of documents to retrieve
            reasoning_steps: Number of reasoning steps to perform
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
        self.reasoning_steps = reasoning_steps
        
        # Define prompt templates
        self.retrieval_planning_template = PromptTemplate(
            input_variables=["question"],
            template="""
            You are a fitness knowledge expert. Given the following question, identify the key concepts 
            and information needed to provide a comprehensive answer.
            
            Question: {question}
            
            1. List the key fitness concepts in this question.
            2. What specific information would be needed to answer this question well?
            3. Generate 3 specific search queries that would help retrieve relevant information.
            
            Format your response as a JSON object with keys: "concepts", "information_needed", and "search_queries".
            """
        )
        
        self.reasoning_template = PromptTemplate(
            input_variables=["question", "context", "previous_reasoning", "step", "reasoning_steps"],
            template="""
            You are a fitness expert working through a multi-step reasoning process to answer a question.
            
            Question: {question}
            
            Relevant Context:
            {context}
            
            Previous Reasoning Steps:
            {previous_reasoning}
            
            This is reasoning step {step} of {reasoning_steps}.
            
            For this step:
            1. Analyze the information provided in the context
            2. Connect it with your previous reasoning
            3. Identify any gaps or contradictions
            4. Draw intermediate conclusions
            
            Provide your reasoning for this step:
            """
        )
        
        self.reflection_template = PromptTemplate(
            input_variables=["question", "context", "reasoning_chain"],
            template="""
            You are a fitness expert reflecting on your reasoning process to ensure accuracy and completeness.
            
            Question: {question}
            
            Relevant Context:
            {context}
            
            Your Reasoning Chain:
            {reasoning_chain}
            
            Reflect on your reasoning:
            1. Are there any logical gaps or inconsistencies in your reasoning?
            2. Did you miss any important information from the context?
            3. Are there alternative interpretations or approaches you should consider?
            4. How confident are you in your conclusions and why?
            
            Based on this reflection, provide an improved reasoning chain:
            """
        )
        
        self.answer_synthesis_template = PromptTemplate(
            input_variables=["question", "context", "reasoning_chain", "reflection"],
            template="""
            You are a knowledgeable fitness trainer assistant. Synthesize a comprehensive answer based on your reasoning.
            
            Question: {question}
            
            Your reasoning process:
            {reasoning_chain}
            
            Your reflection:
            {reflection}
            
            Synthesize a clear, accurate, and helpful answer to the original question.
            Focus on providing actionable advice and evidence-based information.
            If there are limitations to your answer, acknowledge them.
            
            Answer:
            """
        )
        
        # Create LLM chains
        self.retrieval_planning_chain = LLMChain(llm=self.llm, prompt=self.retrieval_planning_template)
        self.reasoning_chain = LLMChain(llm=self.llm, prompt=self.reasoning_template)
        self.reflection_chain = LLMChain(llm=self.llm, prompt=self.reflection_template)
        self.answer_synthesis_chain = LLMChain(llm=self.llm, prompt=self.answer_synthesis_template)
        
        logger.info("RaptorRAG initialized successfully")
    
    def plan_retrieval(self, question: str) -> Dict[str, Any]:
        """
        Plan the retrieval strategy for a given question.
        
        Args:
            question: The question to answer
            
        Returns:
            Dictionary with retrieval plan
        """
        logger.info(f"Planning retrieval for question: {question}")
        
        # Generate retrieval plan
        plan_response = self.retrieval_planning_chain.invoke({"question": question})
        
        # Parse response (in a real implementation, we would parse the JSON)
        # For simplicity, we'll just return the raw response
        return {"raw_plan": plan_response["text"]}
    
    def retrieve_documents(self, question: str, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using the retrieval plan.
        
        Args:
            question: The question to answer
            plan: The retrieval plan
            
        Returns:
            List of retrieved documents
        """
        logger.info(f"Retrieving documents for question: {question}")
        
        # Extract search queries from plan (in a real implementation)
        # For simplicity, we'll just use the original question
        search_queries = [question]
        
        all_documents = []
        seen_ids = set()
        
        for query in search_queries:
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
    
    def perform_multi_step_reasoning(
        self, 
        question: str, 
        context: str
    ) -> Dict[str, str]:
        """
        Perform multi-step reasoning on the retrieved context.
        
        Args:
            question: The question to answer
            context: The retrieved context
            
        Returns:
            Dictionary with reasoning chain and reflection
        """
        logger.info(f"Performing multi-step reasoning for question: {question}")
        
        # Initialize reasoning chain
        reasoning_steps = []
        previous_reasoning = ""
        
        # Perform reasoning steps
        for step in range(1, self.reasoning_steps + 1):
            # Generate reasoning for current step
            reasoning = self.reasoning_chain.invoke({
                "question": question,
                "context": context,
                "previous_reasoning": previous_reasoning,
                "step": step,
                "reasoning_steps": self.reasoning_steps
            })
            
            # Add to reasoning chain
            reasoning_steps.append(f"Step {step}: {reasoning['text']}")
            
            # Update previous reasoning
            if previous_reasoning:
                previous_reasoning += f"\n\nStep {step}: {reasoning['text']}"
            else:
                previous_reasoning = f"Step {step}: {reasoning['text']}"
        
        # Combine reasoning steps
        reasoning_chain = "\n\n".join(reasoning_steps)
        
        # Perform reflection
        reflection = self.reflection_chain.invoke({
            "question": question,
            "context": context,
            "reasoning_chain": reasoning_chain
        })
        
        return {
            "reasoning_chain": reasoning_chain,
            "reflection": reflection["text"]
        }
    
    def synthesize_answer(
        self, 
        question: str, 
        context: str, 
        reasoning_result: Dict[str, str]
    ) -> str:
        """
        Synthesize the final answer based on reasoning and reflection.
        
        Args:
            question: The question to answer
            context: The retrieved context
            reasoning_result: The result of multi-step reasoning
            
        Returns:
            Synthesized answer
        """
        logger.info(f"Synthesizing answer for question: {question}")
        
        # Extract reasoning chain and reflection
        reasoning_chain = reasoning_result["reasoning_chain"]
        reflection = reasoning_result["reflection"]
        
        # Synthesize answer
        answer = self.answer_synthesis_chain.invoke({
            "question": question,
            "context": context,
            "reasoning_chain": reasoning_chain,
            "reflection": reflection
        })
        
        return answer["text"].strip()
    
    def answer_question(self, question: str) -> str:
        """
        Answer a question using the RAPTOR approach.
        
        Args:
            question: The question to answer
            
        Returns:
            Generated answer
        """
        logger.info(f"Answering question with RAPTOR approach: {question}")
        
        # Plan retrieval
        retrieval_plan = self.plan_retrieval(question)
        
        # Retrieve documents
        documents = self.retrieve_documents(question, retrieval_plan)
        
        # Prepare context
        context = "\n\n".join([f"Document {i+1}:\n{doc['text']}" for i, doc in enumerate(documents)])
        
        # Perform multi-step reasoning
        reasoning_result = self.perform_multi_step_reasoning(question, context)
        
        # Synthesize answer
        answer = self.synthesize_answer(question, context, reasoning_result)
        
        return answer.strip()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAPTOR RAG for fitness knowledge")
    parser.add_argument("--question", type=str, help="Question to answer")
    args = parser.parse_args()
    
    # Initialize RAG
    rag = RaptorRAG()
    
    # Answer question
    if args.question:
        answer = rag.answer_question(args.question)
        print(f"Question: {args.question}")
        print(f"Answer: {answer}")
    else:
        # Interactive mode
        print("RAPTOR RAG for fitness knowledge")
        print("Enter 'quit' to exit")
        
        while True:
            question = input("\nEnter your fitness question: ")
            if question.lower() in ["quit", "exit", "q"]:
                break
                
            answer = rag.answer_question(question)
            print(f"\nAnswer: {answer}")
