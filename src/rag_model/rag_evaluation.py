"""
RAG Evaluation Framework for PersonalTrainerAI

This module implements evaluation metrics and methods for comparing
different RAG implementations.
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import numpy as np
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Import RAG implementations
from naive_rag import NaiveRAG
from advanced_rag import AdvancedRAG
from modular_rag import ModularRAG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class RAGEvaluator:
    """
    Evaluator for comparing different RAG implementations.
    """
    
    def __init__(
        self,
        llm_model_name: str = "gpt-4",
        test_queries_path: Optional[str] = None
    ):
        """
        Initialize the RAG evaluator.
        
        Args:
            llm_model_name: Name of the LLM model for evaluation
            test_queries_path: Path to test queries JSON file
        """
        # Initialize evaluation LLM (using a more capable model for evaluation)
        logger.info(f"Initializing evaluation LLM: {llm_model_name}")
        self.eval_llm = OpenAI(model_name=llm_model_name, temperature=0.0, api_key=OPENAI_API_KEY)
        
        # Initialize RAG implementations
        self.rag_implementations = {
            "naive": NaiveRAG(),
            "advanced": AdvancedRAG(),
            "modular": ModularRAG()
        }
        
        # Load test queries if provided
        self.test_queries = []
        if test_queries_path:
            self.load_test_queries(test_queries_path)
        else:
            # Default test queries covering different fitness topics
            self.test_queries = [
                {
                    "query": "What's the best way to improve my bench press?",
                    "category": "exercise_technique"
                },
                {
                    "query": "Can you create a 4-day workout plan to build muscle?",
                    "category": "workout_planning"
                },
                {
                    "query": "What should I eat before and after a workout?",
                    "category": "nutrition_advice"
                },
                {
                    "query": "How can I track my fitness progress effectively?",
                    "category": "progress_tracking"
                },
                {
                    "query": "What are the best exercises for preventing lower back pain?",
                    "category": "injury_prevention"
                }
            ]
        
        # Define evaluation prompts
        self.relevance_prompt = PromptTemplate(
            input_variables=["query", "response"],
            template="""
            You are an expert fitness evaluator assessing the relevance of a response to a user query.
            
            User query: {query}
            
            Response to evaluate: {response}
            
            On a scale of 1-10, rate how relevant the response is to the user's query.
            Consider whether the response directly addresses the question and provides the information the user is seeking.
            
            Provide only a numeric score from 1-10, with no explanation.
            """
        )
        
        self.factual_accuracy_prompt = PromptTemplate(
            input_variables=["query", "response"],
            template="""
            You are an expert fitness evaluator assessing the factual accuracy of a response to a user query.
            
            User query: {query}
            
            Response to evaluate: {response}
            
            On a scale of 1-10, rate the factual accuracy of the response.
            Consider whether the information provided is scientifically sound and aligns with established fitness principles.
            
            Provide only a numeric score from 1-10, with no explanation.
            """
        )
        
        self.completeness_prompt = PromptTemplate(
            input_variables=["query", "response"],
            template="""
            You are an expert fitness evaluator assessing the completeness of a response to a user query.
            
            User query: {query}
            
            Response to evaluate: {response}
            
            On a scale of 1-10, rate how complete the response is.
            Consider whether the response covers all aspects of the query and provides comprehensive information.
            
            Provide only a numeric score from 1-10, with no explanation.
            """
        )
        
        self.hallucination_prompt = PromptTemplate(
            input_variables=["query", "response", "context"],
            template="""
            You are an expert fitness evaluator assessing whether a response contains hallucinations.
            
            User query: {query}
            
            Retrieved context: {context}
            
            Response to evaluate: {response}
            
            On a scale of 1-10, rate how well the response sticks to information in the retrieved context (10 = perfectly grounded, 1 = completely hallucinated).
            Consider whether the response contains information not present in the context or contradicts the context.
            
            Provide only a numeric score from 1-10, with no explanation.
            """
        )
        
        # Create evaluation chains
        self.relevance_chain = LLMChain(llm=self.eval_llm, prompt=self.relevance_prompt)
        self.factual_accuracy_chain = LLMChain(llm=self.eval_llm, prompt=self.factual_accuracy_prompt)
        self.completeness_chain = LLMChain(llm=self.eval_llm, prompt=self.completeness_prompt)
        self.hallucination_chain = LLMChain(llm=self.eval_llm, prompt=self.hallucination_prompt)
        
        logger.info("RAG evaluator initialized successfully")
    
    def load_test_queries(self, file_path: str) -> None:
        """
        Load test queries from a JSON file.
        
        Args:
            file_path: Path to the JSON file containing test queries
        """
        logger.info(f"Loading test queries from {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                self.test_queries = json.load(f)
            
            logger.info(f"Loaded {len(self.test_queries)} test queries")
        except Exception as e:
            logger.error(f"Error loading test queries: {e}")
            # Use default queries if loading fails
            logger.info("Using default test queries")
    
    def save_test_queries(self, file_path: str) -> None:
        """
        Save test queries to a JSON file.
        
        Args:
            file_path: Path to save the test queries
        """
        logger.info(f"Saving test queries to {file_path}")
        
        try:
            with open(file_path, 'w') as f:
                json.dump(self.test_queries, f, indent=2)
            
            logger.info(f"Saved {len(self.test_queries)} test queries")
        except Exception as e:
            logger.error(f"Error saving test queries: {e}")
    
    def generate_test_queries(self, num_queries: int = 5) -> None:
        """
        Generate test queries using LLM.
        
        Args:
            num_queries: Number of test queries to generate
        """
        logger.info(f"Generating {num_queries} test queries")
        
        # Define prompt for generating test queries
        generate_queries_prompt = PromptTemplate(
            input_variables=[],
            template="""
            Generate {num_queries} diverse fitness-related questions that a user might ask a personal trainer AI.
            Include questions about workout planning, exercise technique, nutrition, progress tracking, and injury prevention.
            
            Format your response as a JSON array of objects, each with 'query' and 'category' fields.
            Example:
            [
                {{"query": "What's the best way to improve my bench press?", "category": "exercise_technique"}},
                {{"query": "Can you create a 4-day workout plan to build muscle?", "category": "workout_planning"}}
            ]
            """.format(num_queries=num_queries)
        )
        
        # Create chain for generating queries
        generate_queries_chain = LLMChain(llm=self.eval_llm, prompt=generate_queries_prompt)
        
        try:
            # Generate queries
            result = generate_queries_chain.run()
            
            # Parse JSON result
            generated_queries = json.loads(result)
            
            # Add to test queries
            self.test_queries.extend(generated_queries)
            
            logger.info(f"Generated {len(generated_queries)} test queries")
        except Exception as e:
            logger.error(f"Error generating test queries: {e}")
    
    def evaluate_response(self, query: str, response: str, context: str = "") -> Dict[str, float]:
        """
        Evaluate a single response using multiple metrics.
        
        Args:
            query: User query
            response: Generated response
            context: Retrieved context (for hallucination evaluation)
            
        Returns:
            Dictionary of evaluation scores
        """
        logger.info(f"Evaluating response for query: {query}")
        
        scores = {}
        
        # Evaluate relevance
        try:
            relevance_score = float(self.relevance_chain.run(query=query, response=response).strip())
            scores["relevance"] = relevance_score
        except Exception as e:
            logger.error(f"Error evaluating relevance: {e}")
            scores["relevance"] = 5.0  # Default score
        
        # Evaluate factual accuracy
        try:
            factual_accuracy_score = float(self.factual_accuracy_chain.run(query=query, response=response).strip())
            scores["factual_accuracy"] = factual_accuracy_score
        except Exception as e:
            logger.error(f"Error evaluating factual accuracy: {e}")
            scores["factual_accuracy"] = 5.0  # Default score
        
        # Evaluate completeness
        try:
            completeness_score = float(self.completeness_chain.run(query=query, response=response).strip())
            scores["completeness"] = completeness_score
        except Exception as e:
            logger.error(f"Error evaluating completeness: {e}")
            scores["completeness"] = 5.0  # Default score
        
        # Evaluate hallucination if context is provided
        if context:
            try:
                hallucination_score = float(self.hallucination_chain.run(
                    query=query, response=response, context=context).strip())
                scores["hallucination"] = hallucination_score
            except Exception as e:
                logger.error(f"Error evaluating hallucination: {e}")
                scores["hallucination"] = 5.0  # Default score
        
        # Calculate overall score (average of all metrics)
        scores["overall"] = sum(scores.values()) / len(scores)
        
        logger.info(f"Evaluation scores: {scores}")
        return scores
    
    def evaluate_implementation(self, implementation_name: str) -> Dict[str, Any]:
        """
        Evaluate a specific RAG implementation using test queries.
        
        Args:
            implementation_name: Name of the RAG implementation to evaluate
            
        Returns:
            Dictionary of evaluation results
        """
        logger.info(f"Evaluating implementation: {implementation_name}")
        
        if implementation_name not in self.rag_implementations:
            logger.error(f"Implementation {implementation_name} not found")
            return {"error": f"Implementation {implementation_name} not found"}
        
        rag = self.rag_implementations[implementation_name]
        results = []
        
        for query_obj in self.test_queries:
            query = query_obj["query"]
            
            # Process query with RAG implementation
            try:
                rag_result = rag.query(query)
                response = rag_result["response"]
                
                # Get context for hallucination evaluation
                context = ""
                if "retrieved_documents" in rag_result:
                    # Format context from retrieved documents
                    if hasattr(rag, "format_context"):
                        context = rag.format_context(rag_result["retrieved_documents"])
                
                # Evaluate response
                scores = self.evaluate_response(query, response, context)
                
                # Add to results
                results.append({
                    "query": query,
                    "response": response,
                    "scores": scores
                })
                
                logger.info(f"Evaluated query: {query}")
            except Exception as e:
                logger.error(f"Error processing query {query}: {e}")
                results.append({
                    "query": query,
                    "error": str(e)
                })
        
        # Calculate average scores
        avg_scores = {}
        for metric in ["relevance", "factual_accuracy", "completeness", "hallucination", "overall"]:
            scores = [r["scores"].get(metric, 0) for r in results if "scores" in r and metric in r["scores"]]
            if scores:
                avg_scores[metric] = sum(scores) / len(scores)
        
        return {
            "implementation": implementation_name,
            "results": results,
            "average_scores": avg_scores
        }
    
    def compare_implementations(self) -> Dict[str, Any]:
        """
        Compare all RAG implementations using test queries.
        
        Returns:
            Dictionary of comparison results
        """
        logger.info("Comparing all RAG implementations")
        
        comparison = {}
        
        for implementation_name in self.rag_implementations:
            comparison[implementation_name] = self.evaluate_implementation(implementation_name)
        
        # Determine best implementation based on overall score
        best_implementation = None
        best_score = -1
        
        for implementation_name, results in comparison.items():
            if "average_scores" in results and "overall" in results["average_scores"]:
                overall_score = results["average_scores"]["overall"]
                if overall_score > best_score:
                    best_score = overall_score
                    best_implementation = implementation_name
        
        return {
            "comparison": comparison,
            "best_implementation": best_implementation,
            "best_score": best_score
        }
    
    def save_evaluation_results(self, results: Dict[str, Any], file_path: str) -> None:
        """
        Save evaluation results to a JSON file.
        
        Args:
            results: Evaluation results
            file_path: Path to save the results
        """
        logger.info(f"Saving evaluation results to {file_path}")
        
        try:
            with open(file_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info("Evaluation results saved successfully")
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")


# Example usage
if __name__ == "__main__":
    # Initialize evaluator
    evaluator = RA<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>