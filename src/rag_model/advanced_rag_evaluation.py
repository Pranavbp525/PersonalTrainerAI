"""
Simplified RAG Evaluation Framework for PersonalTrainerAI with MLflow
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import json
import time
import argparse
import logging
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI # OpenAIEmbeddings - Not used for HF
# --- CORRECTED IMPORT BELOW ---
from langchain_community.embeddings import HuggingFaceEmbeddings
# --- END CORRECTION ---
from src.rag_model.mlflow.mlflow_rag_tracker import MLflowRAGTracker
import inspect
import re

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
    }
]

class AdvancedRAGEvaluator:
    """
    Simplified evaluator for comparing different RAG implementations with MLflow.
    """

    def __init__(
        self,
        output_dir: str = "results",
        test_queries: List[Dict[str, str]] = None,
        evaluation_llm_model: str = "gpt-4o-mini",
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    ):
        """
        Initialize the RAG evaluator.
        """
        self.output_dir = output_dir
        self.test_queries = test_queries or TEST_QUERIES_WITH_GROUND_TRUTH
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Initializing evaluation LLM: {evaluation_llm_model}")
        try:
            self.evaluation_llm = ChatOpenAI(
                model_name=evaluation_llm_model,
                temperature=0.0,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            logger.info("Evaluation LLM initialized.")
        except Exception as e:
             logger.error(f"Failed to initialize evaluation LLM: {e}", exc_info=True)
             raise

        logger.info(f"Initializing HuggingFaceEmbeddings with model: {embedding_model}")
        try:
            # Use the correct argument name 'model_name' and remove invalid args
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': 'cpu'} # Explicitly use CPU
            )
            logger.info("HuggingFaceEmbeddings initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFaceEmbeddings: {e}", exc_info=True)
            raise # Re-raise to fail the initialization if embeddings are crucial

        self.rag_implementations = self._initialize_rag_implementations()

        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        logger.info(f"Initializing MLflowRAGTracker with tracking URI: {mlflow_tracking_uri}")
        try:
            self.mlflow_tracker = MLflowRAGTracker(
                experiment_name="rag_evaluation",
                tracking_uri=mlflow_tracking_uri
            )
            logger.info("MLflowRAGTracker initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize MLflowRAGTracker: {e}", exc_info=True)
            # Decide if you want to raise here or allow running without MLflow
            self.mlflow_tracker = None # Set to None if failed
            logger.warning("MLflow tracking disabled due to initialization error.")


        self.evaluation_metrics = {
            "faithfulness": self.evaluate_faithfulness,
            "answer_relevancy": self.evaluate_answer_relevancy,
            "context_relevancy": self.evaluate_context_relevancy,
            "context_precision": self.evaluate_context_precision,
            "fitness_domain_accuracy": self.evaluate_fitness_domain_accuracy,
            "scientific_correctness": self.evaluate_scientific_correctness,
            "practical_applicability": self.evaluate_practical_applicability,
            "safety_consideration": self.evaluate_safety_consideration,
            "retrieval_precision": self.evaluate_retrieval_precision,
            "retrieval_recall": self.evaluate_retrieval_recall,
            "answer_completeness": self.evaluate_answer_completeness,
            "answer_conciseness": self.evaluate_answer_conciseness,
            "answer_helpfulness": self.evaluate_answer_helpfulness,
        }

        self.metric_weights = { # Simplified weights
            "faithfulness": 0.15, "answer_relevancy": 0.1,
            "context_relevancy": 0.075, "context_precision": 0.075,
            "fitness_domain_accuracy": 0.1, "scientific_correctness": 0.075,
            "practical_applicability": 0.075, "safety_consideration": 0.05,
            "retrieval_precision": 0.075, "retrieval_recall": 0.075,
            "answer_completeness": 0.05, "answer_conciseness": 0.05,
            "answer_helpfulness": 0.05
        }

    # --- Rest of the class methods (_initialize_rag_implementations, _extract_score, evaluate_*, etc.) ---
    # --- No changes needed in the other methods for THIS specific error ---

    def _initialize_rag_implementations(self):
        """Initialize RAG implementations with proper imports."""
        # --- Load .env here ---
        logger.info("Attempting to load .env within _initialize_rag_implementations")
        env_path = os.path.join("/opt/airflow", ".env") # Absolute path
        if os.path.exists(env_path):
            loaded = load_dotenv(dotenv_path=env_path, override=True, verbose=True)
            logger.info(f"_initialize_rag_implementations: load_dotenv result: {loaded}")
            logger.info(f"_initialize_rag_implementations: LANGSMITH_API_KEY: {os.getenv('LANGSMITH_API_KEY')}") # Check key
            
        else:
            logger.warning(".env file not found within _initialize_rag_implementations")
        # --- End Load Env ---

        implementations = {}

        try:
            from rag_model.advanced_rag import AdvancedRAG # Use relative import from src
            implementations["advanced"] = AdvancedRAG()
            logger.info("Advanced RAG initialized")
        except Exception as e:
            logger.error(f"Failed to init Advanced RAG: {e}")
        # ... (rest of the method unchanged) ...
        try:
            from rag_model.modular_rag import ModularRAG # Use relative import from src
            implementations["modular"] = ModularRAG()
            logger.info("Modular RAG initialized")
        except Exception as e:
            logger.error(f"Failed to init Modular RAG: {e}")

        try:
            from rag_model.raptor_rag import RaptorRAG # Use relative import from src
            implementations["raptor"] = RaptorRAG()
            logger.info("Raptor RAG initialized")
        except Exception as e:
            logger.error(f"Failed to init Raptor RAG: {e}")
        return implementations

    def _extract_score(self, evaluation_text: str) -> float:
        # ... (method unchanged) ...
        try:
            if isinstance(evaluation_text, (int, float)):
                return float(evaluation_text)
            # Try parsing as JSON first (robust)
            try:
                result = json.loads(evaluation_text)
                if isinstance(result, dict) and "score" in result:
                    return float(result["score"])
            except json.JSONDecodeError:
                pass # Ignore if not JSON and try regex

            # Regex for "score": X.Y
            score_match = r'"score":\s*([0-9.]+)'
            match = re.search(score_match, evaluation_text, re.IGNORECASE)
            if match:
                return float(match.group(1))

            # Regex for X.Y / 10
            score_match = r'([0-9.]+)\s*/\s*10'
            match = re.search(score_match, evaluation_text)
            if match:
                # Scale score to 0-10 range if needed, or assume it's already 0-10
                # If the LLM *always* returns 0-10, just return float(match.group(1))
                # If the LLM *always* returns score / N, then adjust calculation
                # Assuming LLM returns 0-10 score directly here based on prompt
                return float(match.group(1))

             # Regex for just a number 0-10 at the end or start
            score_match = r'(?:^|\s|\:)([0-9](?:\.[0-9]+)?|10)(?:$|\s|/)'
            match = re.search(score_match, evaluation_text)
            if match:
                return float(match.group(1))


            logger.warning(f"Could not extract score reliably from: {evaluation_text[:100]}...")
            return 5.0  # Default score if nothing matches

        except Exception as e:
            logger.error(f"Error extracting score: {e}")
            return 5.0


    def evaluate_faithfulness(self, query: str, response: str, contexts: List[str], ground_truth: str) -> float:
        # ... (method unchanged) ...
        combined_context = "\n\n".join(contexts)
        faithfulness_prompt = f"""
        You are evaluating the faithfulness of an AI assistant's response to a fitness-related query.
        Query: {query}
        Retrieved Context: {combined_context}
        Response to Evaluate: {response}
        Instructions: Identify claims in the response and determine if they are supported by the context.
        Calculate faithfulness score as: (supported claims) / (total claims) * 10. Provide only the score.
        """
        evaluation_result = self.evaluation_llm.invoke(faithfulness_prompt)
        return self._extract_score(evaluation_result.content)


    def evaluate_answer_relevancy(self, query: str, response: str) -> float:
        # ... (method unchanged) ...
        query_embedding = self.embeddings.embed_query(query)
        response_embedding = self.embeddings.embed_query(response)
        # Ensure embeddings are numpy arrays for cosine_similarity
        query_embedding_np = np.array(query_embedding).reshape(1, -1)
        response_embedding_np = np.array(response_embedding).reshape(1, -1)

        similarity = cosine_similarity(query_embedding_np, response_embedding_np)[0][0]
        # Scale similarity (which is -1 to 1) to 0-10 range.
        # Simple linear scaling: score = (similarity + 1) * 5
        base_score = max(0, min(10, (similarity + 1) * 5)) # Clamp between 0 and 10

        relevancy_prompt = f"""
        You are evaluating the relevancy of an AI assistant's response to a fitness-related query.
        Query: {query}
        Response to Evaluate: {response}
        Instructions: Determine how directly the response addresses the question.
        A score based on semantic similarity is: {base_score:.2f}/10. Adjust this score based on your overall evaluation of relevancy, focusing on whether the core question was answered. Provide only the final score (0-10).
        """
        evaluation_result = self.evaluation_llm.invoke(relevancy_prompt)
        return self._extract_score(evaluation_result.content)

    def evaluate_context_relevancy(self, query: str, contexts: List[str]) -> float:
        # ... (method unchanged) ...
        if not contexts: return 0.0
        try:
            query_embedding = self.embeddings.embed_query(query)
            # Embed contexts - handle potential errors during batch embedding
            context_embeddings = self.embeddings.embed_documents(contexts)
            # Ensure lists are not empty before converting to numpy arrays
            if not query_embedding or not context_embeddings: return 0.0
            similarities = cosine_similarity(np.array(query_embedding).reshape(1, -1), np.array(context_embeddings))
            avg_similarity = np.mean(similarities)
            # Scale similarity (-1 to 1) to 0-10
            return max(0, min(10, (avg_similarity + 1) * 5))
        except Exception as e:
            logger.error(f"Error calculating context relevancy: {e}")
            return 0.0


    def evaluate_context_precision(self, query: str, contexts: List[str]) -> float:
        # ... (method unchanged) ...
        if not contexts: return 0.0
        precision_prompt = f"""
        You are evaluating the precision of retrieved contexts for a fitness-related query.
        Query: {query}
        Retrieved Contexts: {json.dumps(contexts, indent=2)}
        Instructions: Determine if each context contains information relevant to answering the query.
        Calculate precision as: (relevant contexts) / (total contexts) * 10. Provide only the score (0-10).
        """
        evaluation_result = self.evaluation_llm.invoke(precision_prompt)
        return self._extract_score(evaluation_result.content)


    def evaluate_fitness_domain_accuracy(self, query: str, response: str) -> float:
        # ... (method unchanged) ...
        accuracy_prompt = f"""
        You are a fitness expert evaluating the domain-specific accuracy of an AI assistant's response.
        Query: {query}
        Response to Evaluate: {response}
        Instructions: Evaluate fitness-specific claims for scientific accuracy based on current exercise science and nutritional guidelines.
        Rate the overall fitness domain accuracy on a scale of 0-10. Provide only the score.
        """
        evaluation_result = self.evaluation_llm.invoke(accuracy_prompt)
        return self._extract_score(evaluation_result.content)

    def evaluate_scientific_correctness(self, response: str) -> float:
        # ... (method unchanged) ...
        correctness_prompt = f"""
        You are a scientific expert evaluating the scientific correctness of an AI assistant's response about fitness.
        Response to Evaluate: {response}
        Instructions: Evaluate scientific claims for correctness, avoiding pseudoscience or unsupported claims.
        Rate the overall scientific correctness on a scale of 0-10. Provide only the score.
        """
        evaluation_result = self.evaluation_llm.invoke(correctness_prompt)
        return self._extract_score(evaluation_result.content)

    def evaluate_practical_applicability(self, query: str, response: str) -> float:
        # ... (method unchanged) ...
        applicability_prompt = f"""
        You are evaluating how practically applicable an AI assistant's fitness advice is for users.
        Query: {query}
        Response to Evaluate: {response}
        Instructions: Assess how actionable the advice is for an average person seeking fitness information. Is it realistic and easy to implement?
        Rate the overall practical applicability on a scale of 0-10. Provide only the score.
        """
        evaluation_result = self.evaluation_llm.invoke(applicability_prompt)
        return self._extract_score(evaluation_result.content)

    def evaluate_safety_consideration(self, response: str) -> float:
        # ... (method unchanged) ...
        safety_prompt = f"""
        You are evaluating how well an AI assistant's fitness advice considers safety aspects.
        Response to Evaluate: {response}
        Instructions: Assess whether the response includes appropriate safety precautions, warnings about potential risks, or suggestions for modifications where necessary.
        Rate the overall safety consideration on a scale of 0-10. Provide only the score.
        """
        evaluation_result = self.evaluation_llm.invoke(safety_prompt)
        return self._extract_score(evaluation_result.content)

    def evaluate_retrieval_precision(self, query: str, contexts: List[str]) -> float:
        # ... (method unchanged, calls context_precision) ...
        return self.evaluate_context_precision(query, contexts)

    def evaluate_retrieval_recall(self, query: str, contexts: List[str], ground_truth: str) -> float:
        # ... (method unchanged) ...
        if not contexts: return 0.0
        recall_prompt = f"""
        You are evaluating the recall of a retrieval system for a fitness-related query.
        Query: {query}
        Ground Truth Answer: {ground_truth}
        Retrieved Contexts: {json.dumps(contexts, indent=2)}
        Instructions: Identify key information points present in the ground truth. Determine how many of these key points are mentioned or supported by the retrieved contexts.
        Calculate recall as: (covered key points) / (total key points in ground truth) * 10. Provide only the score (0-10).
        """
        evaluation_result = self.evaluation_llm.invoke(recall_prompt)
        return self._extract_score(evaluation_result.content)


    def evaluate_answer_completeness(self, response: str, ground_truth: str) -> float:
        # ... (method unchanged) ...
        completeness_prompt = f"""
        You are evaluating the completeness of an AI assistant's response compared to a ground truth answer.
        Ground Truth Answer: {ground_truth}
        Response to Evaluate: {response}
        Instructions: Identify key information points present in the ground truth. Determine how many of these key points are covered in the AI's response.
        Calculate completeness as: (covered key points) / (total key points in ground truth) * 10. Provide only the score (0-10).
        """
        evaluation_result = self.evaluation_llm.invoke(completeness_prompt)
        return self._extract_score(evaluation_result.content)


    def evaluate_answer_conciseness(self, query: str, response: str) -> float:
        # ... (method unchanged) ...
        conciseness_prompt = f"""
        You are evaluating the conciseness of an AI assistant's response to a fitness-related query.
        Query: {query}
        Response to Evaluate: {response}
        Instructions: Assess whether the response is direct and to the point, avoiding unnecessary jargon or verbosity, while still being adequately informative.
        Rate the overall conciseness on a scale of 0-10 (where 10 is perfectly concise and informative). Provide only the score.
        """
        evaluation_result = self.evaluation_llm.invoke(conciseness_prompt)
        return self._extract_score(evaluation_result.content)


    def evaluate_answer_helpfulness(self, query: str, response: str) -> float:
        # ... (method unchanged) ...
        helpfulness_prompt = f"""
        You are evaluating how helpful an AI assistant's response is for a user asking a fitness-related query.
        Query: {query}
        Response to Evaluate: {response}
        Instructions: Consider if the response directly answers the query, provides accurate and actionable information, and addresses the likely intent behind the user's question.
        Rate the overall helpfulness on a scale of 0-10 (where 10 is extremely helpful). Provide only the score.
        """
        evaluation_result = self.evaluation_llm.invoke(helpfulness_prompt)
        return self._extract_score(evaluation_result.content)


    def evaluate_response(self, query: str, response: str, contexts: List[str], ground_truth: str) -> Dict[str, float]:
         # ... (method unchanged) ...
        scores = {}
        # Ensure evaluation LLM and embeddings are available
        if not self.evaluation_llm or not self.embeddings:
             logger.error("Cannot evaluate response: Evaluation LLM or Embeddings model not initialized.")
             return {"overall": 0.0} # Return default zero score if models missing

        for metric_name, metric_fn in self.evaluation_metrics.items():
            logger.debug(f"Evaluating metric: {metric_name}")
            try:
                # Pass only required arguments based on metric type
                sig = inspect.signature(metric_fn)
                params = sig.parameters
                kwargs_to_pass = {}
                if 'query' in params: kwargs_to_pass['query'] = query
                if 'response' in params: kwargs_to_pass['response'] = response
                if 'contexts' in params: kwargs_to_pass['contexts'] = contexts
                if 'ground_truth' in params: kwargs_to_pass['ground_truth'] = ground_truth

                scores[metric_name] = metric_fn(**kwargs_to_pass)
                logger.debug(f"Metric {metric_name} score: {scores[metric_name]}")

            except Exception as e:
                logger.error(f"Error evaluating metric '{metric_name}': {e}", exc_info=True)
                scores[metric_name] = 0.0  # Assign default score on error

        # Calculate weighted overall score
        weighted_sum = 0
        total_weight = 0
        for metric, weight in self.metric_weights.items():
            if metric in scores:
                weighted_sum += scores[metric] * weight
                total_weight += weight
            else:
                logger.warning(f"Metric '{metric}' defined in weights but not found in calculated scores.")

        # Normalize overall score by total weight used (in case some metrics failed)
        scores["overall"] = (weighted_sum / total_weight) * 10.0 if total_weight > 0 else 0.0
        logger.info(f"Calculated overall weighted score: {scores['overall']:.2f}/10.0")
        return scores


    def get_retrieved_contexts(self, implementation_name: str, query: str) -> List[str]:
         # ... (method unchanged) ...
        if implementation_name not in self.rag_implementations:
            logger.error(f"Unknown implementation: {implementation_name}")
            return []
        rag = self.rag_implementations[implementation_name]
        try:
            if hasattr(rag, 'answer_question') and 'return_contexts' in inspect.signature(rag.answer_question).parameters:
                # Assume it returns (response, contexts)
                _, contexts = rag.answer_question(query, return_contexts=True)
                # Ensure contexts is a list of strings
                if isinstance(contexts, list) and all(isinstance(c, str) for c in contexts):
                    return contexts
                elif isinstance(contexts, list): # Maybe list of Documents? Extract page_content
                     return [getattr(c, 'page_content', str(c)) for c in contexts]
                else:
                    logger.warning(f"Got contexts in unexpected format from {implementation_name}: {type(contexts)}")
                    return []
            else:
                logger.warning(f"Implementation {implementation_name} does not have 'answer_question' method supporting 'return_contexts'. Cannot get contexts.")
                return []
        except Exception as e:
            logger.error(f"Error retrieving contexts from {implementation_name} for query '{query[:50]}...': {e}", exc_info=True)
            return []

    def evaluate_implementation(self, rag_instance, implementation_name=None):
         # ... (method largely unchanged, ensure mlflow_tracker check) ...
        # Determine implementation name if not provided
        if implementation_name is None:
            cls_name = rag_instance.__class__.__name__.lower()
            if "advanced" in cls_name: implementation_name = "advanced"
            elif "modular" in cls_name: implementation_name = "modular"
            elif "raptor" in cls_name: implementation_name = "raptor"
            else: implementation_name = cls_name

        logger.info(f"--- Evaluating RAG Implementation: {implementation_name} ---")

        # Re-get the instance from the dictionary to be sure we use the initialized one
        if implementation_name not in self.rag_implementations:
            logger.error(f"Implementation '{implementation_name}' not found in initialized implementations.")
            return {"error": f"Implementation '{implementation_name}' not found."}
        rag = self.rag_implementations[implementation_name]

        all_results_for_impl = []
        total_response_time = 0

        for i, query_data in enumerate(tqdm(self.test_queries, desc=f"Evaluating {implementation_name}")):
            query = query_data["query"]
            ground_truth = query_data["ground_truth"]
            logger.debug(f"Processing query {i+1}/{len(self.test_queries)}: {query[:50]}...")

            start_time = time.time()
            response = f"Error: Could not get answer from {implementation_name}"
            contexts = []
            try:
                # Get response and contexts together if possible
                if hasattr(rag, 'answer_question') and 'return_contexts' in inspect.signature(rag.answer_question).parameters:
                    response, contexts = rag.answer_question(query, return_contexts=True)
                    # Ensure contexts format is list of strings
                    if isinstance(contexts, list) and contexts and not isinstance(contexts[0], str):
                         contexts = [getattr(c, 'page_content', str(c)) for c in contexts] # Extract page content if Document objects
                    elif not isinstance(contexts, list): contexts = []

                # Fallback if return_contexts not supported
                else:
                    response = rag.answer_question(query)
                    # Attempt to get contexts separately if needed by evaluation metrics
                    if any(m in ["faithfulness", "context_relevancy", "context_precision", "retrieval_recall", "retrieval_precision"] for m in self.evaluation_metrics):
                         contexts = self.get_retrieved_contexts(implementation_name, query) # Call the separate function

            except Exception as e:
                logger.error(f"Error getting response/contexts from {implementation_name} for query '{query[:50]}...': {e}", exc_info=True)
                response = f"Error during answer generation: {e}"
                contexts = [] # Ensure contexts is empty on error

            end_time = time.time()
            response_time = end_time - start_time
            total_response_time += response_time
            logger.debug(f"Response time: {response_time:.2f}s")

            logger.debug(f"Retrieved {len(contexts)} contexts for evaluation.")

            # Evaluate the response
            try:
                evaluation = self.evaluate_response(query, response, contexts, ground_truth)
            except Exception as e:
                logger.error(f"Error during evaluation call for query '{query[:50]}...': {e}", exc_info=True)
                evaluation = {} # Assign empty dict on evaluation error

            all_results_for_impl.append({
                "query": query, "response": response, "ground_truth": ground_truth,
                "contexts": contexts, "evaluation": evaluation, "response_time": response_time
            })
            logger.info(f"Query {i+1} ({implementation_name}) completed. Overall score: {evaluation.get('overall', 0.0):.2f}/10.0")


        # Calculate average scores
        avg_scores = {}
        if all_results_for_impl: # Ensure there are results
            all_metrics_found = list(all_results_for_impl[0].get("evaluation", {}).keys()) # Get metrics from first result
            for metric in all_metrics_found:
                 # Calculate mean only for metrics that were successfully calculated (not 0.0 due to error)
                 valid_scores = [r["evaluation"].get(metric, 0.0) for r in all_results_for_impl if metric in r.get("evaluation", {})]
                 avg_scores[metric] = np.mean(valid_scores) if valid_scores else 0.0
        else:
             logger.warning(f"No results generated for implementation {implementation_name}, cannot calculate average scores.")


        avg_response_time = (total_response_time / len(self.test_queries)) if self.test_queries else 0

        # --- Log to MLflow (Check if tracker exists) ---
        if self.mlflow_tracker:
            logger.info(f"Logging results for {implementation_name} to MLflow...")
            # Prepare parameters (extract from rag_instance if possible)
            parameters_to_log = {
                "embedding_model": getattr(rag, "embedding_model_name", getattr(rag, "embedding_model", "unknown")), # Try specific attribute first
                "llm_model": getattr(rag, "llm_model_name", getattr(rag, "llm_model", "unknown")),
                # Add other known parameters if available
                "chunk_size": getattr(rag, "chunk_size", "unknown"),
                "chunk_overlap": getattr(rag, "chunk_overlap", "unknown"),
                "retrieval_k": getattr(rag, "retrieval_k", "unknown"),
                "temperature": getattr(rag, "temperature", "unknown"),
                "num_test_queries": len(self.test_queries),
                "evaluation_llm": self.evaluation_llm.model_name,

            }
            # Structure results for logging (metrics dict and results artifact)
            results_to_log = {
                "implementation": implementation_name,
                "average_scores": avg_scores,
                "average_response_time": avg_response_time,
                "detailed_results": all_results_for_impl # Log detailed results as artifact
            }

            try:
                 # Use the RAG logging function in the tracker
                 self.mlflow_tracker.log_rag_evaluation_results(
                      results=results_to_log, # Pass the structured results
                      implementation_name=implementation_name,
                      parameters=parameters_to_log
                 )
            except Exception as mlflow_err:
                 logger.error(f"Failed to log results to MLflow for {implementation_name}: {mlflow_err}", exc_info=True)
        else:
             logger.warning("MLflow tracker not available. Skipping MLflow logging.")


        # Return structure compatible with compare_implementations
        return {
            "implementation": implementation_name,
            "results": all_results_for_impl, # Detailed results per query
            "average_scores": avg_scores,
            "average_response_time": avg_response_time
        }


    def compare_implementations(self) -> Dict[str, Any]:
        """Compare all RAG implementations."""
        logger.info("--- Starting RAG Implementation Comparison ---")
        comparison_data = {} # Store results per implementation

        # Ensure RAG implementations are initialized
        if not self.rag_implementations:
             logger.error("No RAG implementations were initialized. Cannot compare.")
             return {"error": "No RAG implementations initialized."}

        for impl_name, rag_instance in self.rag_implementations.items():
            # Evaluate each implementation - this now handles MLflow logging internally
            impl_results = self.evaluate_implementation(rag_instance, impl_name)
            if "error" not in impl_results:
                comparison_data[impl_name] = impl_results # Store the returned dict
            else:
                logger.error(f"Evaluation failed for {impl_name}: {impl_results['error']}")


        # Determine best implementation based on overall average score
        best_implementation = None
        best_score = -1.0
        valid_results = {k: v for k, v in comparison_data.items() if v and "average_scores" in v}

        if valid_results:
            try:
                best_implementation = max(
                    valid_results,
                    key=lambda k: valid_results[k]["average_scores"].get("overall", -1.0)
                )
                best_score = valid_results[best_implementation]["average_scores"].get("overall", -1.0)
                logger.info(f"Best implementation determined: {best_implementation} (Score: {best_score:.2f})")
            except ValueError:
                 logger.warning("Could not determine best implementation (no valid scores found).")
        else:
             logger.warning("No valid results to compare implementations.")


        final_comparison_results = {
            "comparison": comparison_data, # Dict containing results for each implementation
            "best_implementation": best_implementation,
            "best_score": best_score
        }

        self.save_results(final_comparison_results)
        self.print_summary(final_comparison_results)

        # Optional: Log a summary comparison artifact to MLflow?
        # Could log comparison_df from rag_evaluation_mlflow.py idea here
        # if self.mlflow_tracker and valid_results:
        #     try:
        #         # Create comparison dataframe logic here...
        #         # self.mlflow_tracker.start_run(run_name="rag_comparison_summary")
        #         # self.mlflow_tracker.log_artifact(comparison_csv_path)
        #         # self.mlflow_tracker.end_run()
        #         pass
        #     except Exception as e:
        #          logger.error(f"Failed to log comparison summary to MLflow: {e}")

        logger.info("--- RAG Implementation Comparison Finished ---")
        return final_comparison_results


    def save_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results to file."""
        output_file = os.path.join(self.output_dir, "advanced_evaluation_results.json")
        try:
            with open(output_file, "w", encoding='utf-8') as f:
                # Handle potential non-serializable numpy types
                json.dump(results, f, indent=2, ensure_ascii=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))
            logger.info(f"Evaluation results saved to {output_file}")
        except Exception as e:
             logger.error(f"Failed to save results to {output_file}: {e}", exc_info=True)

    def print_summary(self, results: Dict[str, Any]) -> None:
        # ... (method unchanged) ...
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
                    avg_scores = implementation_results["average_scores"]
                    # Ensure 'overall' is printed last if it exists
                    metric_order = [m for m in avg_scores if m != 'overall'] + (['overall'] if 'overall' in avg_scores else [])
                    for metric in metric_order:
                        score = avg_scores.get(metric, 0.0)
                        print(f"    {metric.replace('_', ' ').title()}: {score:.2f}/10.0")
                    print(f"  Avg Response Time: {implementation_results.get('average_response_time', 0):.2f}s")
                else:
                    print(f"\n{implementation.upper()} RAG: No average scores calculated.")
        else:
            print("\nNo valid comparison results available.")
        print("\nDetailed evaluation results saved to:", os.path.join(self.output_dir, "advanced_evaluation_results.json"))
        print("="*80)

# --- Main Guard ---
def main():
    # ... (function unchanged) ...
    """Main function to run the RAG evaluation."""
    parser = argparse.ArgumentParser(description="RAG Evaluation for PersonalTrainerAI")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--evaluation-model", type=str, default="gpt-4o-mini", help="LLM model to use for evaluation")
    parser.add_argument("--implementation", type=str, choices=["advanced", "modular", "raptor", "all"],
                        default="all", help="RAG implementation to evaluate")
    # Corrected max num queries to match the constant list length
    parser.add_argument("--num-queries", type=int, default=len(TEST_QUERIES_WITH_GROUND_TRUTH),
                        help=f"Number of test queries to evaluate (max {len(TEST_QUERIES_WITH_GROUND_TRUTH)})")
    args = parser.parse_args()

    test_queries = TEST_QUERIES_WITH_GROUND_TRUTH[:min(args.num_queries, len(TEST_QUERIES_WITH_GROUND_TRUTH))]

    evaluator = AdvancedRAGEvaluator(
        output_dir=args.output_dir,
        test_queries=test_queries,
        evaluation_llm_model=args.evaluation_model
    )

    if args.implementation == "all":
        results = evaluator.compare_implementations()
    else:
        # Ensure specific implementation is initialized before evaluating
        if args.implementation in evaluator.rag_implementations:
             results = evaluator.evaluate_implementation(evaluator.rag_implementations[args.implementation], args.implementation)
        else:
             print(f"Error: Requested implementation '{args.implementation}' was not initialized or not found.")
             results = {}

    # Only print summary if results were generated
    if results and "error" not in results:
        evaluator.print_summary(results)


if __name__ == "__main__":
    main()