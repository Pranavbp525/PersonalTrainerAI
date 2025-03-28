"""
Advanced RAG Evaluation Framework for PersonalTrainerAI

This script implements a comprehensive evaluation framework for RAG systems using:
1. RAGAS metrics (faithfulness, answer relevancy, context relevancy, context precision)
2. Custom fitness domain-specific metrics
3. Human-like evaluation with ground truth comparison
4. Retrieval-focused evaluation metrics

Usage:
    python advanced_rag_evaluation.py --output-dir results
"""

import os
import json
import time
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Test queries with ground truth answers for evaluation
TEST_QUERIES_WITH_GROUND_TRUTH = [
    {
        "query": "How much protein should I consume daily for muscle growth?",
        "ground_truth": "For muscle growth, consume 1.6-2.2g of protein per kg of bodyweight daily. Athletes and those in intense training may need up to 2.2g/kg. Spread protein intake throughout the day, with 20-40g per meal. Good sources include lean meats, dairy, eggs, and plant-based options like legumes and tofu. Timing protein around workouts (within 2 hours) can enhance muscle protein synthesis."
    },
    {
        "query": "What are the best exercises for core strength?",
        "ground_truth": "The best exercises for core strength include compound movements like squats, deadlifts, and overhead presses that engage the entire core. Specific core exercises include planks (and variations), hollow holds, dead bugs, Russian twists, and cable rotations. Anti-extension exercises (ab rollouts), anti-rotation exercises (Pallof press), and anti-lateral flexion exercises (suitcase carries) target different aspects of core stability. Progressive overload and proper form are essential for developing functional core strength."
    },
    {
        "query": "How do I create a balanced workout routine for beginners?",
        "ground_truth": "A balanced beginner workout routine should include 2-3 full-body strength sessions weekly, focusing on compound exercises (squats, pushups, rows, lunges). Start with bodyweight or light weights, 2-3 sets of 10-15 reps. Include 20-30 minutes of moderate cardio 2-3 times weekly (walking, cycling). Add 5-10 minutes of dynamic stretching before workouts and 5-10 minutes of static stretching after. Rest at least one day between strength sessions. Progress gradually by adding weight or reps every 1-2 weeks. The routine should be sustainable and enjoyable to build consistency."
    },
    {
        "query": "What's the optimal rest period between strength training sessions?",
        "ground_truth": "The optimal rest period between strength training sessions targeting the same muscle groups is typically 48-72 hours to allow for proper recovery and muscle growth. Beginners may need closer to 72 hours, while advanced lifters might recover in 48 hours. For full-body workouts, allow 48+ hours between sessions. With split routines (different muscle groups on different days), you can train daily while still providing adequate rest for each muscle group. Factors affecting recovery include training intensity, volume, nutrition, sleep quality, age, and fitness level. Signs of inadequate recovery include persistent soreness, decreased performance, and fatigue."
    },
    {
        "query": "How can I improve my running endurance?",
        "ground_truth": "To improve running endurance: 1) Gradually increase weekly mileage by no more than 10% per week; 2) Incorporate long, slow runs at conversational pace to build aerobic base; 3) Add interval training (e.g., 400m repeats) and tempo runs (comfortably hard pace) weekly; 4) Include cross-training like cycling or swimming 1-2 times weekly; 5) Strengthen supporting muscles with exercises like squats and lunges; 6) Maintain proper nutrition with adequate carbohydrates and hydration; 7) Ensure sufficient recovery with rest days and sleep; 8) Practice consistent pacing during runs; 9) Consider periodization with build and recovery phases; 10) Be patient as endurance improvements take time."
    }
]

class AdvancedRAGEvaluator:
    """
    Advanced evaluator for comparing different RAG implementations using multiple evaluation frameworks.
    """
    
    def __init__(
        self,
        output_dir: str = "results",
        test_queries: List[Dict[str, str]] = None,
        evaluation_llm_model: str = "gpt-4",
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize the advanced RAG evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
            test_queries: List of test queries with ground truth to evaluate
            evaluation_llm_model: LLM model to use for evaluation
            embedding_model: Embedding model for semantic similarity calculations
        """
        self.output_dir = output_dir
        self.test_queries = test_queries or TEST_QUERIES_WITH_GROUND_TRUTH
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize evaluation LLM
        self.evaluation_llm = ChatOpenAI(
            model_name=evaluation_llm_model,
            temperature=0.0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize embeddings model
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Import RAG implementations directly in the method to avoid import errors
        self.rag_implementations = {}
        self._initialize_rag_implementations()
        
        # Define evaluation metrics
        self.ragas_metrics = [
            "faithfulness",
            "answer_relevancy",
            "context_relevancy",
            "context_precision"
        ]
        
        self.custom_metrics = [
            "fitness_domain_accuracy",
            "scientific_correctness",
            "practical_applicability",
            "safety_consideration"
        ]
        
        self.retrieval_metrics = [
            "retrieval_precision",
            "retrieval_recall"
        ]
        
        self.human_eval_metrics = [
            "answer_completeness",
            "answer_conciseness",
            "answer_helpfulness"
        ]
        
        # All metrics combined
        self.all_metrics = (
            self.ragas_metrics + 
            self.custom_metrics + 
            self.retrieval_metrics + 
            self.human_eval_metrics
        )
        
        logger.info(f"AdvancedRAGEvaluator initialized with {len(self.test_queries)} test queries")
    
    def _initialize_rag_implementations(self):
        """Initialize RAG implementations with proper imports."""
        try:
            # Import Advanced RAG
            from src.rag_model.advanced_rag import AdvancedRAG
            self.rag_implementations["advanced"] = AdvancedRAG()
            logger.info("Advanced RAG implementation initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Advanced RAG: {e}")
        
        try:
            # Import Modular RAG
            from src.rag_model.modular_rag import ModularRAG
            self.rag_implementations["modular"] = ModularRAG()
            logger.info("Modular RAG implementation initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Modular RAG: {e}")
        
        try:
            # Import RAPTOR RAG
            from src.rag_model.raptor_rag import RaptorRAG
            self.rag_implementations["raptor"] = RaptorRAG()
            logger.info("RAPTOR RAG implementation initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAPTOR RAG: {e}")
    
    def evaluate_faithfulness(self, query: str, response: str, contexts: List[str]) -> float:
        """
        Evaluate the faithfulness of a response (whether it's supported by the retrieved contexts).
        
        Args:
            query: The original query
            response: The generated response
            contexts: The retrieved contexts used to generate the response
            
        Returns:
            Faithfulness score (0-10)
        """
        # Combine contexts
        combined_context = "\n\n".join(contexts)
        
        # Prompt for faithfulness evaluation
        faithfulness_prompt = f"""
        You are evaluating the faithfulness of an AI assistant's response to a fitness-related query.
        Faithfulness measures whether the response is factually consistent with the provided context.
        
        Query: {query}
        
        Retrieved Context:
        {combined_context}
        
        Response to Evaluate:
        {response}
        
        Instructions:
        1. Identify all factual claims in the response.
        2. For each claim, determine if it is:
           - Fully supported by the context (assign 1 point)
           - Partially supported by the context (assign 0.5 points)
           - Not supported by the context or contradicted (assign 0 points)
        3. Calculate the faithfulness score as: (sum of points) / (total number of claims) * 10
        
        Provide your evaluation as a JSON object with:
        1. "claims": List of identified claims with their support status
        2. "score": The calculated faithfulness score (0-10)
        """
        
        # Get evaluation from LLM
        evaluation_result = self.evaluation_llm.invoke(faithfulness_prompt)
        evaluation_text = evaluation_result.content
        
        # Extract score from evaluation
        try:
            import re
            import json
            
            # Try to parse JSON directly
            try:
                result = json.loads(evaluation_text)
                if "score" in result:
                    return float(result["score"])
            except:
                pass
            
            # Fallback: extract score using regex
            score_match = r'"score":\s*([0-9.]+)'
            match = re.search(score_match, evaluation_text)
            if match:
                return float(match.group(1))
            
            # Second fallback: look for a score out of 10
            score_match = r'([0-9.]+)\s*/\s*10'
            match = re.search(score_match, evaluation_text)
            if match:
                return float(match.group(1))
            
            # Final fallback
            return 5.0
            
        except Exception as e:
            logger.error(f"Error extracting faithfulness score: {e}")
            return 5.0
    
    def evaluate_answer_relevancy(self, query: str, response: str) -> float:
        """
        Evaluate how relevant the response is to the query.
        
        Args:
            query: The original query
            response: The generated response
            
        Returns:
            Answer relevancy score (0-10)
        """
        # Calculate semantic similarity between query and response
        query_embedding = self.embeddings.embed_query(query)
        response_embedding = self.embeddings.embed_query(response)
        
        # Convert to numpy arrays
        query_embedding_np = np.array(query_embedding).reshape(1, -1)
        response_embedding_np = np.array(response_embedding).reshape(1, -1)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(query_embedding_np, response_embedding_np)[0][0]
        
        # Scale similarity to 0-10 range
        base_score = similarity * 10
        
        # Use LLM to refine the score
        relevancy_prompt = f"""
        You are evaluating the relevancy of an AI assistant's response to a fitness-related query.
        
        Query: {query}
        
        Response to Evaluate:
        {response}
        
        Instructions:
        1. Determine how directly the response addresses the specific question asked.
        2. Consider whether the response contains information that answers the core question.
        3. Evaluate if the response provides practical, actionable information related to the query.
        4. Ignore information that is correct but not relevant to the query.
        
        The initial relevancy score based on semantic similarity is: {base_score:.2f}/10
        
        Adjust this score based on your evaluation. Provide your final score (0-10) and a brief explanation.
        Format your response as a JSON object with "score" and "explanation" fields.
        """
        
        # Get evaluation from LLM
        evaluation_result = self.evaluation_llm.invoke(relevancy_prompt)
        evaluation_text = evaluation_result.content
        
        # Extract score from evaluation
        try:
            import re
            import json
            
            # Try to parse JSON directly
            try:
                result = json.loads(evaluation_text)
                if "score" in result:
                    return float(result["score"])
            except:
                pass
            
            # Fallback: extract score using regex
            score_match = r'"score":\s*([0-9.]+)'
            match = re.search(score_match, evaluation_text)
            if match:
                return float(match.group(1))
            
            # Second fallback
            score_match = r'([0-9.]+)\s*/\s*10'
            match = re.search(score_match, evaluation_text)
            if match:
                return float(match.group(1))
            
            # Final fallback: use the base score
            return base_score
            
        except Exception as e:
            logger.error(f"Error extracting answer relevancy score: {e}")
            return base_score
    
    def evaluate_context_relevancy(self, query: str, contexts: List[str]) -> float:
        """
        Evaluate how relevant the retrieved contexts are to the query.
        
        Args:
            query: The original query
            contexts: The retrieved contexts
            
        Returns:
            Context relevancy score (0-10)
        """
        if not contexts:
            return 0.0
        
        # Calculate semantic similarity between query and each context
        query_embedding = self.embeddings.embed_query(query)
        context_embeddings = [self.embeddings.embed_query(context) for context in contexts]
        
        # Convert to numpy arrays
        query_embedding_np = np.array(query_embedding).reshape(1, -1)
        context_embeddings_np = np.array(context_embeddings)
        
        # Calculate cosine similarity for each context
        similarities = cosine_similarity(query_embedding_np, context_embeddings_np)[0]
        
        # Calculate average similarity
        avg_similarity = np.mean(similarities)
        
        # Scale to 0-10 range
        return avg_similarity * 10
    
    def evaluate_context_precision(self, query: str, contexts: List[str]) -> float:
        """
        Evaluate the precision of the retrieved contexts (ratio of relevant contexts).
        
        Args:
            query: The original query
            contexts: The retrieved contexts
            
        Returns:
            Context precision score (0-10)
        """
        if not contexts:
            return 0.0
        
        # Prompt for context precision evaluation
        precision_prompt = f"""
        You are evaluating the precision of retrieved contexts for a fitness-related query.
        
        Query: {query}
        
        Retrieved Contexts:
        {json.dumps(contexts, indent=2)}
        
        Instructions:
        1. For each context, determine if it contains information relevant to answering the query.
        2. Classify each context as "relevant" or "not relevant".
        3. Calculate precision as: (number of relevant contexts) / (total number of contexts) * 10
        
        Provide your evaluation as a JSON object with:
        1. "relevant_contexts": List of indices of relevant contexts (0-based)
        2. "score": The calculated precision score (0-10)
        """
        
        # Get evaluation from LLM
        evaluation_result = self.evaluation_llm.invoke(precision_prompt)
        evaluation_text = evaluation_result.content
        
        # Extract score from evaluation
        try:
            import re
            import json
            
            # Try to parse JSON directly
            try:
                result = json.loads(evaluation_text)
                if "score" in result:
                    return float(result["score"])
                elif "relevant_contexts" in result:
                    # Calculate score from relevant contexts
                    precision = len(result["relevant_contexts"]) / len(contexts)
                    return precision * 10
            except:
                pass
            
            # Fallback: extract score using regex
            score_match = r'"score":\s*([0-9.]+)'
            match = re.search(score_match, evaluation_text)
            if match:
                return float(match.group(1))
            
            # Second fallback
            score_match = r'([0-9.]+)\s*/\s*10'
            match = re.search(score_match, evaluation_text)
            if match:
                return float(match.group(1))
            
            # Final fallback
            return 5.0
            
        except Exception as e:
            logger.error(f"Error extracting context precision score: {e}")
            return 5.0
    
    def evaluate_fitness_domain_accuracy(self, query: str, response: str) -> float:
        """
        Evaluate the accuracy of fitness-specific information in the response.
        
        Args:
            query: The original query
            response: The generated response
            
        Returns:
            Fitness domain accuracy score (0-10)
        """
        # Prompt for fitness domain accuracy evaluation
        accuracy_prompt = f"""
        You are a fitness expert evaluating the domain-specific accuracy of an AI assistant's response.
        
        Query: {query}
        
        Response to Evaluate:
        {response}
        
        Instructions:
        1. Identify all fitness-specific claims, recommendations, and explanations in the response.
        2. Evaluate each for scientific accuracy according to current fitness research and best practices.
        3. Consider factors like:
           - Accuracy of physiological explanations
           - Correctness of training principles
           - Alignment with evidence-based nutrition guidelines
           - Appropriate exercise recommendations
        4. Rate the overall fitness domain accuracy on a scale of 0-10.
        
        Provide your evaluation as a JSON object with:
        1. "analysis": Brief analysis of the fitness-specific content
        2. "score": The fitness domain accuracy score (0-10)
        """
        
        # Get evaluation from LLM
        evaluation_result = self.evaluation_llm.invoke(accuracy_prompt)
        evaluation_text = evaluation_result.content
        
        # Extract score from evaluation
        try:
            import re
            import json
            
            # Try to parse JSON directly
            try:
                result = json.loads(evaluation_text)
                if "score" in result:
                    return float(result["score"])
            except:
                pass
            
            # Fallback: extract score using regex
            score_match = r'"score":\s*([0-9.]+)'
            match = re.search(score_match, evaluation_text)
            if match:
                return float(match.group(1))
            
            # Second fallback
            score_match = r'([0-9.]+)\s*/\s*10'
            match = re.search(score_match, evaluation_text)
            if match:
                return float(match.group(1))
            
            # Final fallback
            return 5.0
            
        except Exception as e:
            logger.error(f"Error extracting fitness domain accuracy score: {e}")
            return 5.0
    
    def evaluate_scientific_correctness(self, response: str) -> float:
        """
        Evaluate the scientific correctness of the response.
        
        Args:
            response: The generated response
            
        Returns:
            Scientific correctness score (0-10)
        """
        # Prompt for scientific correctness evaluation
        correctness_prompt = f"""
        You are a scientific expert evaluating the scientific correctness of an AI assistant's response about fitness.
        
        Response to Evaluate:
        {response}
        
        Instructions:
        1. Identify all scientific claims and explanations in the response.
        2. Evaluate each for correctness according to current scientific understanding.
        3. Consider factors like:
           - Accuracy of physiological mechanisms described
           - Correct use of scientific terminology
           - Alignment with peer-reviewed research
           - Absence of pseudoscientific claims
        4. Rate the overall scientific correctness on a scale of 0-10.
        
        Provide your evaluation as a JSON object with:
        1. "analysis": Brief analysis of the scientific content
        2. "score": The scientific correctness score (0-10)
        """
        
        # Get evaluation from LLM
        evaluation_result = self.evaluation_llm.invoke(correctness_prompt)
        evaluation_text = evaluation_result.content
        
        # Extract score from evaluation
        try:
            import re
            import json
            
            # Try to parse JSON directly
            try:
                result = json.loads(evaluation_text)
                if "score" in result:
                    return float(result["score"])
            except:
                pass
            
            # Fallback: extract score using regex
            score_match = r'"score":\s*([0-9.]+)'
            match = re.search(score_match, evaluation_text)
            if match:
                return float(match.group(1))
            
            # Second fallback
            score_match = r'([0-9.]+)\s*/\s*10'
            match = re.search(score_match, evaluation_text)
            if match:
                return float(match.group(1))
            
            # Final fallback
            return 5.0
            
        except Exception as e:
            logger.error(f"Error extracting scientific correctness score: {e}")
            return 5.0
    
    def evaluate_practical_applicability(self, query: str, response: str) -> float:
        """
        Evaluate how practically applicable the response is for the user.
        
        Args:
            query: The original query
            response: The generated response
            
        Returns:
            Practical applicability score (0-10)
        """
        # Prompt for practical applicability evaluation
        applicability_prompt = f"""
        You are evaluating how practically applicable an AI assistant's fitness advice is for users.
        
        Query: {query}
        
        Response to Evaluate:
        {response}
        
        Instructions:
        1. Assess how actionable the advice is for an average person.
        2. Evaluate whether the response provides clear, specific steps or guidelines.
        3. Consider if the advice is realistic and implementable for most people.
        4. Determine if the response accounts for different fitness levels or limitations.
        5. Rate the overall practical applicability on a scale of 0-10.
        
        Provide your evaluation as a JSON object with:
        1. "analysis": Brief analysis of the practical applicability
        2. "score": The practical applicability score (0-10)
        """
        
        # Get evaluation from LLM
        evaluation_result = self.evaluation_llm.invoke(applicability_prompt)
        evaluation_text = evaluation_result.content
        
        # Extract score from evaluation
        try:
            import re
            import json
            
            # Try to parse JSON directly
            try:
                result = json.loads(evaluation_text)
                if "score" in result:
                    return float(result["score"])
            except:
                pass
            
            # Fallback: extract score using regex
            score_match = r'"score":\s*([0-9.]+)'
            match = re.search(score_match, evaluation_text)
            if match:
                return float(match.group(1))
            
            # Second fallback
            score_match = r'([0-9.]+)\s*/\s*10'
            match = re.search(score_match, evaluation_text)
            if match:
                return float(match.group(1))
            
            # Final fallback
            return 5.0
            
        except Exception as e:
            logger.error(f"Error extracting practical applicability score: {e}")
            return 5.0
    
    def evaluate_safety_consideration(self, response: str) -> float:
        """
        Evaluate how well the response considers safety aspects.
        
        Args:
            response: The generated response
            
        Returns:
            Safety consideration score (0-10)
        """
        # Prompt for safety consideration evaluation
        safety_prompt = f"""
        You are evaluating how well an AI assistant's fitness advice considers safety aspects.
        
        Response to Evaluate:
        {response}
        
        Instructions:
        1. Identify any fitness recommendations or exercise advice in the response.
        2. Assess whether the response includes appropriate safety precautions.
        3. Evaluate if the advice accounts for potential risks or contraindications.
        4. Consider if the response mentions form, technique, or progression guidelines.
        5. Determine if the response avoids potentially harmful recommendations.
        6. Rate the overall safety consideration on a scale of 0-10.
        
        Provide your evaluation as a JSON object with:
        1. "analysis": Brief analysis of the safety considerations
        2. "score": The safety consideration score (0-10)
        """
        
        # Get evaluation from LLM
        evaluation_result = self.evaluation_llm.invoke(safety_prompt)
        evaluation_text = evaluation_result.content
        
        # Extract score from evaluation
        try:
            import re
            import json
            
            # Try to parse JSON directly
            try:
                result = json.loads(evaluation_text)
                if "score" in result:
                    return float(result["score"])
            except:
                pass
            
            # Fallback: extract score using regex
            score_match = r'"score":\s*([0-9.]+)'
            match = re.search(score_match, evaluation_text)
            if match:
                return float(match.group(1))
            
            # Second fallback
            score_match = r'([0-9.]+)\s*/\s*10'
            match = re.search(score_match, evaluation_text)
            if match:
                return float(match.group(1))
            
            # Final fallback
            return 5.0
            
        except Exception as e:
            logger.error(f"Error extracting safety consideration score: {e}")
            return 5.0
    
    def evaluate_retrieval_precision(self, query: str, contexts: List[str]) -> float:
        """
        Evaluate the precision of the retrieval (ratio of relevant contexts).
        
        Args:
            query: The original query
            contexts: The retrieved contexts
            
        Returns:
            Retrieval precision score (0-10)
        """
        # This is similar to context_precision but focused on retrieval performance
        return self.evaluate_context_precision(query, contexts)
    
    def evaluate_retrieval_recall(self, query: str, contexts: List[str], ground_truth: str) -> float:
        """
        Evaluate the recall of the retrieval (how much of the ground truth is covered).
        
        Args:
            query: The original query
            contexts: The retrieved contexts
            ground_truth: The ground truth answer
            
        Returns:
            Retrieval recall score (0-10)
        """
        if not contexts:
            return 0.0
        
        # Prompt for retrieval recall evaluation
        recall_prompt = f"""
        You are evaluating the recall of a retrieval system for a fitness-related query.
        
        Query: {query}
        
        Ground Truth Answer (contains all key information that should be retrieved):
        {ground_truth}
        
        Retrieved Contexts:
        {json.dumps(contexts, indent=2)}
        
        Instructions:
        1. Identify all key information points in the ground truth answer.
        2. For each key point, determine if it is covered in any of the retrieved contexts.
        3. Calculate recall as: (number of key points covered) / (total number of key points) * 10
        
        Provide your evaluation as a JSON object with:
        1. "key_points": List of key information points from the ground truth
        2. "covered_points": List of key points that are covered in the retrieved contexts
        3. "score": The calculated recall score (0-10)
        """
        
        # Get evaluation from LLM
        evaluation_result = self.evaluation_llm.invoke(recall_prompt)
        evaluation_text = evaluation_result.content
        
        # Extract score from evaluation
        try:
            import re
            import json
            
            # Try to parse JSON directly
            try:
                result = json.loads(evaluation_text)
                if "score" in result:
                    return float(result["score"])
                elif "key_points" in result and "covered_points" in result:
                    # Calculate score from covered points
                    recall = len(result["covered_points"]) / len(result["key_points"])
                    return recall * 10
            except:
                pass
            
            # Fallback: extract score using regex
            score_match = r'"score":\s*([0-9.]+)'
            match = re.search(score_match, evaluation_text)
            if match:
                return float(match.group(1))
            
            # Second fallback
            score_match = r'([0-9.]+)\s*/\s*10'
            match = re.search(score_match, evaluation_text)
            if match:
                return float(match.group(1))
            
            # Final fallback
            return 5.0
            
        except Exception as e:
            logger.error(f"Error extracting retrieval recall score: {e}")
            return 5.0
    
    def evaluate_answer_completeness(self, response: str, ground_truth: str) -> float:
        """
        Evaluate how complete the response is compared to the ground truth.
        
        Args:
            response: The generated response
            ground_truth: The ground truth answer
            
        Returns:
            Answer completeness score (0-10)
        """
        # Prompt for answer completeness evaluation
        completeness_prompt = f"""
        You are evaluating the completeness of an AI assistant's response compared to a ground truth answer.
        
        Ground Truth Answer (contains all key information that should be included):
        {ground_truth}
        
        Response to Evaluate:
        {response}
        
        Instructions:
        1. Identify all key information points in the ground truth answer.
        2. For each key point, determine if it is covered in the response.
        3. Calculate completeness as: (number of key points covered) / (total number of key points) * 10
        
        Provide your evaluation as a JSON object with:
        1. "key_points": List of key information points from the ground truth
        2. "covered_points": List of key points that are covered in the response
        3. "score": The calculated completeness score (0-10)
        """
        
        # Get evaluation from LLM
        evaluation_result = self.evaluation_llm.invoke(completeness_prompt)
        evaluation_text = evaluation_result.content
        
        # Extract score from evaluation
        try:
            import re
            import json
            
            # Try to parse JSON directly
            try:
                result = json.loads(evaluation_text)
                if "score" in result:
                    return float(result["score"])
                elif "key_points" in result and "covered_points" in result:
                    # Calculate score from covered points
                    completeness = len(result["covered_points"]) / len(result["key_points"])
                    return completeness * 10
            except:
                pass
            
            # Fallback: extract score using regex
            score_match = r'"score":\s*([0-9.]+)'
            match = re.search(score_match, evaluation_text)
            if match:
                return float(match.group(1))
            
            # Second fallback
            score_match = r'([0-9.]+)\s*/\s*10'
            match = re.search(score_match, evaluation_text)
            if match:
                return float(match.group(1))
            
            # Final fallback
            return 5.0
            
        except Exception as e:
            logger.error(f"Error extracting answer completeness score: {e}")
            return 5.0
    
    def evaluate_answer_conciseness(self, response: str) -> float:
        """
        Evaluate how concise the response is without sacrificing important information.
        
        Args:
            response: The generated response
            
        Returns:
            Answer conciseness score (0-10)
        """
        # Prompt for answer conciseness evaluation
        conciseness_prompt = f"""
        You are evaluating the conciseness of an AI assistant's response to a fitness-related query.
        
        Response to Evaluate:
        {response}
        
        Instructions:
        1. Assess whether the response is appropriately concise while still being informative.
        2. Identify any unnecessary repetition, verbosity, or tangential information.
        3. Consider if the response could be more direct without losing important content.
        4. Rate the overall conciseness on a scale of 0-10, where:
           - 0: Extremely verbose with significant unnecessary content
           - 5: Moderately concise with some unnecessary elements
           - 10: Perfectly concise with all information being relevant and non-redundant
        
        Provide your evaluation as a JSON object with:
        1. "analysis": Brief analysis of the response's conciseness
        2. "score": The conciseness score (0-10)
        """
        
        # Get evaluation from LLM
        evaluation_result = self.evaluation_llm.invoke(conciseness_prompt)
        evaluation_text = evaluation_result.content
        
        # Extract score from evaluation
        try:
            import re
            import json
            
            # Try to parse JSON directly
            try:
                result = json.loads(evaluation_text)
                if "score" in result:
                    return float(result["score"])
            except:
                pass
            
            # Fallback: extract score using regex
            score_match = r'"score":\s*([0-9.]+)'
            match = re.search(score_match, evaluation_text)
            if match:
                return float(match.group(1))
            
            # Second fallback
            score_match = r'([0-9.]+)\s*/\s*10'
            match = re.search(score_match, evaluation_text)
            if match:
                return float(match.group(1))
            
            # Final fallback
            return 5.0
            
        except Exception as e:
            logger.error(f"Error extracting answer conciseness score: {e}")
            return 5.0
    
    def evaluate_answer_helpfulness(self, query: str, response: str) -> float:
        """
        Evaluate how helpful the response is for the user.
        
        Args:
            query: The original query
            response: The generated response
            
        Returns:
            Answer helpfulness score (0-10)
        """
        # Prompt for answer helpfulness evaluation
        helpfulness_prompt = f"""
        You are evaluating how helpful an AI assistant's response is for a fitness-related query.
        
        Query: {query}
        
        Response to Evaluate:
        {response}
        
        Instructions:
        1. Assess how well the response addresses the user's specific needs.
        2. Evaluate whether the response provides actionable information.
        3. Consider if the response is clear, well-structured, and easy to understand.
        4. Determine if the response anticipates follow-up questions or concerns.
        5. Rate the overall helpfulness on a scale of 0-10.
        
        Provide your evaluation as a JSON object with:
        1. "analysis": Brief analysis of the response's helpfulness
        2. "score": The helpfulness score (0-10)
        """
        
        # Get evaluation from LLM
        evaluation_result = self.evaluation_llm.invoke(helpfulness_prompt)
        evaluation_text = evaluation_result.content
        
        # Extract score from evaluation
        try:
            import re
            import json
            
            # Try to parse JSON directly
            try:
                result = json.loads(evaluation_text)
                if "score" in result:
                    return float(result["score"])
            except:
                pass
            
            # Fallback: extract score using regex
            score_match = r'"score":\s*([0-9.]+)'
            match = re.search(score_match, evaluation_text)
            if match:
                return float(match.group(1))
            
            # Second fallback
            score_match = r'([0-9.]+)\s*/\s*10'
            match = re.search(score_match, evaluation_text)
            if match:
                return float(match.group(1))
            
            # Final fallback
            return 5.0
            
        except Exception as e:
            logger.error(f"Error extracting answer helpfulness score: {e}")
            return 5.0
    
    def evaluate_response(
        self, 
        query: str, 
        response: str, 
        contexts: List[str], 
        ground_truth: str
    ) -> Dict[str, float]:
        """
        Evaluate a response based on multiple metrics.
        
        Args:
            query: The original query
            response: The generated response
            contexts: The retrieved contexts used to generate the response
            ground_truth: The ground truth answer
            
        Returns:
            Dictionary of evaluation scores
        """
        scores = {}
        
        # RAGAS metrics
        scores["faithfulness"] = self.evaluate_faithfulness(query, response, contexts)
        scores["answer_relevancy"] = self.evaluate_answer_relevancy(query, response)
        scores["context_relevancy"] = self.evaluate_context_relevancy(query, contexts)
        scores["context_precision"] = self.evaluate_context_precision(query, contexts)
        
        # Custom fitness domain metrics
        scores["fitness_domain_accuracy"] = self.evaluate_fitness_domain_accuracy(query, response)
        scores["scientific_correctness"] = self.evaluate_scientific_correctness(response)
        scores["practical_applicability"] = self.evaluate_practical_applicability(query, response)
        scores["safety_consideration"] = self.evaluate_safety_consideration(response)
        
        # Retrieval metrics
        scores["retrieval_precision"] = self.evaluate_retrieval_precision(query, contexts)
        scores["retrieval_recall"] = self.evaluate_retrieval_recall(query, contexts, ground_truth)
        
        # Human-like evaluation metrics
        scores["answer_completeness"] = self.evaluate_answer_completeness(response, ground_truth)
        scores["answer_conciseness"] = self.evaluate_answer_conciseness(response)
        scores["answer_helpfulness"] = self.evaluate_answer_helpfulness(query, response)
        
        # Calculate overall score (weighted average)
        weights = {
            # RAGAS metrics (40%)
            "faithfulness": 0.15,
            "answer_relevancy": 0.1,
            "context_relevancy": 0.075,
            "context_precision": 0.075,
            
            # Custom fitness domain metrics (30%)
            "fitness_domain_accuracy": 0.1,
            "scientific_correctness": 0.075,
            "practical_applicability": 0.075,
            "safety_consideration": 0.05,
            
            # Retrieval metrics (15%)
            "retrieval_precision": 0.075,
            "retrieval_recall": 0.075,
            
            # Human-like evaluation metrics (15%)
            "answer_completeness": 0.05,
            "answer_conciseness": 0.05,
            "answer_helpfulness": 0.05
        }
        
        weighted_sum = sum(scores[metric] * weights[metric] for metric in weights)
        scores["overall"] = weighted_sum
        
        return scores
    
    def get_retrieved_contexts(self, implementation_name: str, query: str) -> List[str]:
        """
        Get the retrieved contexts for a query from a specific implementation.
        
        Args:
            implementation_name: Name of the RAG implementation
            query: The query to retrieve contexts for
            
        Returns:
            List of retrieved contexts
        """
        if implementation_name not in self.rag_implementations:
            logger.error(f"Unknown implementation: {implementation_name}")
            return []
        
        rag = self.rag_implementations[implementation_name]
        
        try:
            # Use the return_contexts parameter if the implementation supports it
            if hasattr(rag, 'answer_question') and 'return_contexts' in inspect.signature(rag.answer_question).parameters:
                _, contexts = rag.answer_question(query, return_contexts=True)
                return contexts
            else:
                # Fallback to the old behavior
                logger.warning(f"Implementation {implementation_name} does not support returning contexts")
                return []
        except Exception as e:
            logger.error(f"Error retrieving contexts from {implementation_name}: {e}")
            return []
    
    def evaluate_implementation(self, implementation_name: str) -> Dict[str, Any]:
        """
        Evaluate a specific RAG implementation.
        
        Args:
            implementation_name: Name of the RAG implementation
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating {implementation_name} RAG implementation")
        
        if implementation_name not in self.rag_implementations:
            logger.error(f"Unknown implementation: {implementation_name}")
            return {"error": f"Unknown implementation: {implementation_name}"}
        
        rag = self.rag_implementations[implementation_name]
        results = []
        total_response_time = 0
        
        for i, query_data in enumerate(self.test_queries):
            query = query_data["query"]
            ground_truth = query_data["ground_truth"]
            
            logger.info(f"Processing query {i+1}/{len(self.test_queries)}: {query[:50]}...")
            
            # Measure response time and get response
            start_time = time.time()
            
            # Try to get contexts along with the response if supported
            try:
                if hasattr(rag, 'answer_question') and 'return_contexts' in inspect.signature(rag.answer_question).parameters:
                    response, contexts = rag.answer_question(query, return_contexts=True)
                else:
                    response = rag.answer_question(query)
                    contexts = self.get_retrieved_contexts(implementation_name, query)
            except Exception as e:
                logger.error(f"Error getting response and contexts: {e}")
                response = "Error generating response"
                contexts = []
            
            end_time = time.time()
            response_time = end_time - start_time
            total_response_time += response_time
            
            # Log context information for debugging
            logger.info(f"Retrieved {len(contexts)} contexts for evaluation")
            
            # Evaluate response
            evaluation = self.evaluate_response(query, response, contexts, ground_truth)
            
            # Store result
            results.append({
                "query": query,
                "response": response,
                "ground_truth": ground_truth,
                "contexts": contexts,
                "evaluation": evaluation,
                "response_time": response_time
            })
            
            logger.info(f"Query {i+1} completed. Overall score: {evaluation['overall']:.2f}/10.0")
        
        # Calculate average scores
        average_scores = {}
        for metric in self.all_metrics + ["overall"]:
            average_scores[metric] = sum(r["evaluation"][metric] for r in results) / len(results)
        
        # Calculate average response time
        average_response_time = total_response_time / len(self.test_queries)
        
        return {
            "implementation": implementation_name,
            "results": results,
            "average_scores": average_scores,
            "average_response_time": average_response_time
        }
    
    def compare_implementations(self) -> Dict[str, Any]:
        """
        Compare all RAG implementations.
        
        Returns:
            Dictionary with comparison results
        """
        logger.info("Starting RAG implementation comparison")
        
        results = {}
        for implementation_name in self.rag_implementations:
            results[implementation_name] = self.evaluate_implementation(implementation_name)
        
        # Determine best implementation
        best_implementation = None
        best_score = -1
        
        for implementation_name, implementation_results in results.items():
            if "average_scores" in implementation_results and "overall" in implementation_results["average_scores"]:
                overall_score = implementation_results["average_scores"]["overall"]
                if overall_score > best_score:
                    best_score = overall_score
                    best_implementation = implementation_name
        
        # Create comparison results
        comparison_results = {
            "comparison": results,
            "best_implementation": best_implementation,
            "best_score": best_score
        }
        
        # Save results
        self.save_results(comparison_results)
        
        # Generate comparison charts
        self.generate_comparison_charts(results)
        
        return comparison_results
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """
        Save evaluation results to file.
        
        Args:
            results: Evaluation results to save
        """
        output_file = os.path.join(self.output_dir, "advanced_evaluation_results.json")
        
        # Convert results to JSON-serializable format
        serializable_results = json.loads(json.dumps(results, default=str))
        
        with open(output_file, "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {output_file}")
    
    def generate_comparison_charts(self, results: Dict[str, Any]) -> None:
        """
        Generate comparison charts for visualization.
        
        Args:
            results: Evaluation results to visualize
        """
        # Extract data for charts
        implementations = list(results.keys())
        
        # Group metrics by category
        metric_groups = {
            "RAGAS Metrics": self.ragas_metrics,
            "Fitness Domain Metrics": self.custom_metrics,
            "Retrieval Metrics": self.retrieval_metrics,
            "Human Evaluation Metrics": self.human_eval_metrics
        }
        
        # Create a figure for each metric group
        for group_name, metrics in metric_groups.items():
            plt.figure(figsize=(12, 8))
            
            # Set up bar positions
            x = np.arange(len(implementations))
            width = 0.8 / len(metrics)
            offsets = np.linspace(-0.4 + width/2, 0.4 - width/2, len(metrics))
            
            # Plot bars for each metric
            for i, metric in enumerate(metrics):
                values = []
                for implementation in implementations:
                    impl_results = results[implementation]
                    if "average_scores" in impl_results and metric in impl_results["average_scores"]:
                        values.append(impl_results["average_scores"][metric])
                    else:
                        values.append(0)
                
                plt.bar(x + offsets[i], values, width=width, label=metric.replace("_", " ").title())
            
            plt.xlabel("RAG Implementation")
            plt.ylabel("Score (0-10)")
            plt.title(f"RAG Implementation Comparison: {group_name}")
            plt.xticks(x, implementations)
            plt.legend()
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.savefig(os.path.join(self.output_dir, f"{group_name.lower().replace(' ', '_')}_comparison.png"))
        
        # Create overall comparison chart
        plt.figure(figsize=(10, 6))
        overall_scores = []
        for implementation in implementations:
            impl_results = results[implementation]
            if "average_scores" in impl_results and "overall" in impl_results["average_scores"]:
                overall_scores.append(impl_results["average_scores"]["overall"])
            else:
                overall_scores.append(0)
        
        plt.bar(implementations, overall_scores)
        plt.xlabel("RAG Implementation")
        plt.ylabel("Overall Score (0-10)")
        plt.title("RAG Implementation Comparison: Overall Performance")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.savefig(os.path.join(self.output_dir, "overall_comparison.png"))
        
        # Create response time comparison chart
        plt.figure(figsize=(10, 6))
        response_times = []
        for implementation in implementations:
            impl_results = results[implementation]
            if "average_response_time" in impl_results:
                response_times.append(impl_results["average_response_time"])
            else:
                response_times.append(0)
        
        plt.bar(implementations, response_times)
        plt.xlabel("RAG Implementation")
        plt.ylabel("Average Response Time (seconds)")
        plt.title("RAG Implementation Comparison: Response Time")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.savefig(os.path.join(self.output_dir, "response_time_comparison.png"))
        
        logger.info(f"Comparison charts saved to {self.output_dir}")
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """
        Print a summary of the evaluation results.
        
        Args:
            results: Evaluation results to summarize
        """
        print("\n" + "="*80)
        print("ADVANCED RAG IMPLEMENTATION COMPARISON RESULTS")
        print("="*80)
        
        if "best_implementation" in results and results["best_implementation"]:
            print(f"\nBest implementation: {results['best_implementation'].upper()}")
            print(f"Best overall score: {results['best_score']:.2f}/10.0")
            
            print("\nDetailed scores by implementation:")
            for implementation, implementation_results in results["comparison"].items():
                if "average_scores" in implementation_results:
                    print(f"\n{implementation.upper()} RAG:")
                    
                    # Group metrics by category for better readability
                    metric_groups = {
                        "RAGAS Metrics": self.ragas_metrics,
                        "Fitness Domain Metrics": self.custom_metrics,
                        "Retrieval Metrics": self.retrieval_metrics,
                        "Human Evaluation Metrics": self.human_eval_metrics,
                        "Overall": ["overall"]
                    }
                    
                    for group_name, metrics in metric_groups.items():
                        print(f"  {group_name}:")
                        for metric in metrics:
                            if metric in implementation_results["average_scores"]:
                                score = implementation_results["average_scores"][metric]
                                print(f"    {metric.replace('_', ' ').title()}: {score:.2f}/10.0")
                    
                    print(f"  Response Time: {implementation_results.get('average_response_time', 0):.2f}s")
        else:
            print("\nNo valid comparison results available.")
        
        print("\nDetailed evaluation results saved to:", os.path.join(self.output_dir, "advanced_evaluation_results.json"))
        print("Comparison charts saved to:", self.output_dir)
        print("="*80)


def main():
    """Main function to run the advanced RAG evaluation."""
    parser = argparse.ArgumentParser(description="Advanced RAG Evaluation for PersonalTrainerAI")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--evaluation-model", type=str, default="gpt-4", help="LLM model to use for evaluation")
    parser.add_argument("--implementation", type=str, choices=["advanced", "modular", "raptor", "all"], 
                        default="all", help="RAG implementation to evaluate")
    parser.add_argument("--num-queries", type=int, default=5, 
                        help="Number of test queries to evaluate (max 5)")
    args = parser.parse_args()
    
    # Limit number of queries if specified
    test_queries = TEST_QUERIES_WITH_GROUND_TRUTH[:min(args.num_queries, len(TEST_QUERIES_WITH_GROUND_TRUTH))]
    
    # Initialize evaluator
    evaluator = AdvancedRAGEvaluator(
        output_dir=args.output_dir,
        test_queries=test_queries,
        evaluation_llm_model=args.evaluation_model
    )
    
    # Run evaluation
    if args.implementation == "all":
        results = evaluator.compare_implementations()
        evaluator.print_summary(results)
    else:
        results = evaluator.evaluate_implementation(args.implementation)
        
        # Print summary for single implementation
        print(f"\nEvaluation results for {args.implementation.upper()} RAG:")
        
        # Group metrics by category for better readability
        metric_groups = {
            "RAGAS Metrics": evaluator.ragas_metrics,
            "Fitness Domain Metrics": evaluator.custom_metrics,
            "Retrieval Metrics": evaluator.retrieval_metrics,
            "Human Evaluation Metrics": evaluator.human_eval_metrics,
            "Overall": ["overall"]
        }
        
        for group_name, metrics in metric_groups.items():
            print(f"  {group_name}:")
            for metric in metrics:
                if metric in results["average_scores"]:
                    score = results["average_scores"][metric]
                    print(f"    {metric.replace('_', ' ').title()}: {score:.2f}/10.0")
        
        print(f"  Response Time: {results.get('average_response_time', 0):.2f}s")
        
        # Save individual results
        individual_results = {
            "comparison": {args.implementation: results},
            "best_implementation": args.implementation,
            "best_score": results["average_scores"]["overall"]
        }
        evaluator.save_results(individual_results)
        print(f"\nDetailed evaluation results saved to: {os.path.join(args.output_dir, 'advanced_evaluation_results.json')}")


if __name__ == "__main__":
    main()
