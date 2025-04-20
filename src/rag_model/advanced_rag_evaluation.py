# src/rag_model/advanced_rag_evaluation.py
"""
Simplified RAG Evaluation Framework for PersonalTrainerAI with MLflow & GCS Output
"""
import sys
import os
# Add project root to sys.path to allow importing other modules like gcs_utils
# Assumes this script is run from a context where /opt/airflow/app is the project root
project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)

import json
import time
import argparse
import logging
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.rag_model.mlflow.mlflow_rag_tracker import MLflowRAGTracker
import inspect
import re

# Attempt to import GCS utils - handle failure gracefully
try:
    from src.data_pipeline.gcs_utils import upload_string_to_gcs
    gcs_utils_available = True
except ImportError:
    logging.warning("Could not import gcs_utils. Saving results to GCS will be disabled.")
    gcs_utils_available = False
    def upload_string_to_gcs(*args, **kwargs): # Dummy function
        logging.error("GCS Utils not available, cannot upload string.")
        return False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from default location (.env in project root)
# Use the path expected inside the container based on docker-compose mount
dotenv_path = "/opt/airflow/app/.env" # Path inside container where .env is mounted
if os.path.exists(dotenv_path):
    loaded = load_dotenv(dotenv_path=dotenv_path, override=True)
    logger.info(f"Attempted to load .env file from {dotenv_path}. Load successful: {loaded}")
else:
    logger.warning(f".env file not found at {dotenv_path}. Relying on environment variables already set.")


# --- GCS Configuration for Output ---
EVALUATION_OUTPUT_BUCKET = "ragllm-454718-eval-results" # <<< CREATE THIS BUCKET IN GCP

# --- Test Queries (Keep unchanged) ---
TEST_QUERIES_WITH_GROUND_TRUTH = [
    # ... (your existing test queries) ...
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
    Results are saved to GCS or logged as MLflow artifacts.
    """

    def __init__(
        self,
        output_dir: str = "/tmp/rag_eval_output", # Default local temp dir if needed
        test_queries: List[Dict[str, str]] = None,
        evaluation_llm_model: str = "gpt-4o-mini",
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    ):
        """
        Initialize the RAG evaluator.
        """
        self.output_dir = output_dir # May not be used if saving direct to GCS/MLflow
        self.test_queries = test_queries or TEST_QUERIES_WITH_GROUND_TRUTH
        os.makedirs(output_dir, exist_ok=True) # Create temp dir just in case

        # Ensure required API keys are present
        openai_key = os.getenv("OPENAI_API_KEY")
        pinecone_key = os.getenv("PINECONE_API_KEY") # Assuming RAG models need this
        if not openai_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        # Add checks for other critical keys (like PINECONE_API_KEY) if needed by RAG implementations
        if not pinecone_key:
             logger.warning("PINECONE_API_KEY not found in environment. RAG initializations might fail.")


        logger.info(f"Initializing evaluation LLM: {evaluation_llm_model}")
        try:
            self.evaluation_llm = ChatOpenAI(
                model_name=evaluation_llm_model,
                temperature=0.0,
                openai_api_key=openai_key # Use variable
            )
            logger.info("Evaluation LLM initialized.")
        except Exception as e:
             logger.error(f"Failed to initialize evaluation LLM: {e}", exc_info=True)
             raise

        logger.info(f"Initializing HuggingFaceEmbeddings with model: {embedding_model}")
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': 'cpu'} # Explicitly use CPU
            )
            logger.info("HuggingFaceEmbeddings initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFaceEmbeddings: {e}", exc_info=True)
            raise # Re-raise to fail the initialization

        # Initialize RAG implementations (ensure they can access env vars)
        self.rag_implementations = self._initialize_rag_implementations()
        if not self.rag_implementations:
             logger.warning("No RAG implementations were successfully initialized. Evaluation may be limited.")

        # Initialize MLflow Tracker
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000") # Default if not set
        logger.info(f"Initializing MLflowRAGTracker with tracking URI: {mlflow_tracking_uri}")
        try:
            self.mlflow_tracker = MLflowRAGTracker(
                experiment_name="rag_evaluation", # Consider making this configurable
                tracking_uri=mlflow_tracking_uri
            )
            logger.info("MLflowRAGTracker initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize MLflowRAGTracker: {e}", exc_info=True)
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

    def _initialize_rag_implementations(self):
        """Initialize RAG implementations. Assumes env vars are loaded."""
        # Note: load_dotenv() was called at the module level already
        implementations = {}
        logger.info("Initializing RAG implementations...")
        # Ensure required keys for RAG models are available
        pinecone_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_key:
             logger.error("Cannot initialize RAG models: PINECONE_API_KEY is not set.")
             return implementations # Return empty if critical key missing

        try:
            from rag_model.advanced_rag import AdvancedRAG # Use relative import from src
            # Pass necessary env vars if the class doesn't load them itself
            implementations["advanced"] = AdvancedRAG()
            logger.info("Advanced RAG initialized")
        except Exception as e:
            logger.error(f"Failed to init Advanced RAG: {e}", exc_info=True)

        try:
            from rag_model.modular_rag import ModularRAG # Use relative import from src
            implementations["modular"] = ModularRAG()
            logger.info("Modular RAG initialized")
        except Exception as e:
            logger.error(f"Failed to init Modular RAG: {e}", exc_info=True)

        try:
            from rag_model.raptor_rag import RaptorRAG # Use relative import from src
            implementations["raptor"] = RaptorRAG()
            logger.info("Raptor RAG initialized")
        except Exception as e:
            logger.error(f"Failed to init Raptor RAG: {e}", exc_info=True)

        logger.info(f"Successfully initialized {len(implementations)} RAG implementations.")
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
                    score_val = result["score"]
                    if isinstance(score_val, (int, float)):
                        return float(score_val)
                    elif isinstance(score_val, str):
                        # Try converting string score if possible
                        return float(score_val.strip())
            except (json.JSONDecodeError, ValueError, TypeError):
                pass # Ignore if not JSON or score not valid, try regex

            # Regex for "score": X.Y or score: X or X/10 etc.
            score_patterns = [
                r'"score":\s*([0-9]+(?:\.[0-9]+)?)', # "score": 85.5
                r"'score':\s*([0-9]+(?:\.[0-9]+)?)", # 'score': 85.5
                r'score\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)', # score: 85 or score = 85
                r'([0-9]+(?:\.[0-9]+)?)\s*/\s*10', # 8.5 / 10
                r'Rating:\s*([0-9]+(?:\.[0-9]+)?)', # Rating: 9
                r'(?:^|\s|\b)([0-9]+(?:\.[0-9]+)?)(?:$|\s*(?:/10)?\b)' # Standalone number 0-10 potentially with /10
            ]

            for pattern in score_patterns:
                 match = re.search(pattern, evaluation_text, re.IGNORECASE)
                 if match:
                     try:
                         score = float(match.group(1))
                         # Clamp score to 0-10 range if it looks like it might be 0-100 from context
                         # Or just assume LLM follows 0-10 prompt
                         clamped_score = max(0.0, min(10.0, score))
                         logger.debug(f"Extracted score {score}, clamped to {clamped_score}")
                         return clamped_score
                     except (ValueError, IndexError):
                         continue # Ignore if group is not a valid float

            logger.warning(f"Could not extract score reliably from: {evaluation_text[:100]}...")
            return 5.0  # Default score if nothing matches

        except Exception as e:
            logger.error(f"Error extracting score: {e}", exc_info=True)
            return 5.0

    # --- Evaluation Metric Functions (evaluate_faithfulness, etc.) ---
    # Keep these methods as they were defined in your original file
    # Ensure they use self.evaluation_llm and self.embeddings correctly
    # ... (Insert your existing evaluate_* methods here) ...
    def evaluate_faithfulness(self, query: str, response: str, contexts: List[str], ground_truth: str) -> float:
        if not contexts: return 0.0 # Cannot assess faithfulness without context
        combined_context = "\n\n".join(contexts)
        faithfulness_prompt = f"""
        You are evaluating the faithfulness of an AI assistant's response to a fitness-related query based *only* on the provided context.
        Query: {query}
        Retrieved Context: {combined_context}
        Response to Evaluate: {response}
        Instructions: Identify factual claims made in the Response. For each claim, check if it is directly supported or reasonably inferable from the Retrieved Context. Ignore any prior knowledge.
        Calculate faithfulness score as: (number of claims supported by context) / (total number of factual claims in response) * 10.
        Output only a single JSON object containing the key "score" with the calculated numeric faithfulness score (0-10). Example: {{"score": 8.5}}
        """
        evaluation_result = self.evaluation_llm.invoke(faithfulness_prompt)
        return self._extract_score(evaluation_result.content)

    def evaluate_answer_relevancy(self, query: str, response: str) -> float:
        query_embedding = self.embeddings.embed_query(query)
        response_embedding = self.embeddings.embed_query(response)
        query_embedding_np = np.array(query_embedding).reshape(1, -1)
        response_embedding_np = np.array(response_embedding).reshape(1, -1)
        similarity = cosine_similarity(query_embedding_np, response_embedding_np)[0][0]
        base_score = max(0, min(10, (similarity + 1) * 5))
        relevancy_prompt = f"""
        You are evaluating the relevancy of an AI assistant's response to a fitness-related query.
        Query: {query}
        Response to Evaluate: {response}
        Instructions: Assess how well the response addresses the *specific* user query. Does it directly answer the question or go off-topic? Consider the semantic similarity score ({base_score:.2f}/10) as a baseline. Adjust based on whether the core intent of the query was met. Ignore factual accuracy for this metric.
        Output only a single JSON object containing the key "score" with the final numeric relevancy score (0-10). Example: {{"score": 9.0}}
        """
        evaluation_result = self.evaluation_llm.invoke(relevancy_prompt)
        return self._extract_score(evaluation_result.content)

    def evaluate_context_relevancy(self, query: str, contexts: List[str]) -> float:
        if not contexts: return 0.0
        try:
            query_embedding = self.embeddings.embed_query(query)
            context_embeddings = self.embeddings.embed_documents(contexts)
            if not query_embedding or not context_embeddings or len(context_embeddings) != len(contexts):
                 logger.warning("Context relevancy calculation skipped due to embedding issues.")
                 return 0.0
            similarities = cosine_similarity(np.array(query_embedding).reshape(1, -1), np.array(context_embeddings))
            avg_similarity = np.mean(similarities)
            return max(0, min(10, (avg_similarity + 1) * 5))
        except Exception as e:
            logger.error(f"Error calculating context relevancy embedding similarity: {e}")
            return 0.0

    def evaluate_context_precision(self, query: str, contexts: List[str]) -> float:
        if not contexts: return 0.0
        precision_prompt = f"""
        You are evaluating the precision of retrieved contexts for a fitness-related query. Precision measures the ratio of relevant contexts among the retrieved ones.
        Query: {query}
        Retrieved Contexts: {json.dumps(contexts, indent=2)}
        Instructions: For each context, determine if it contains information directly relevant and useful for answering the query.
        Calculate precision score as: (number of relevant contexts) / (total number of retrieved contexts) * 10.
        Output only a single JSON object containing the key "score" with the calculated numeric precision score (0-10). Example: {{"score": 7.5}}
        """
        evaluation_result = self.evaluation_llm.invoke(precision_prompt)
        return self._extract_score(evaluation_result.content)

    def evaluate_fitness_domain_accuracy(self, query: str, response: str) -> float:
        accuracy_prompt = f"""
        You are a fitness expert evaluating the domain-specific accuracy of an AI assistant's response based on established exercise science and nutritional guidelines.
        Query: {query}
        Response to Evaluate: {response}
        Instructions: Check all factual fitness-related claims in the response (e.g., recommended protein intake, exercise benefits, technique descriptions) for accuracy against common scientific consensus in the fitness domain. Penalize misinformation or significantly outdated advice.
        Output only a single JSON object containing the key "score" with the overall fitness domain accuracy score (0-10). Example: {{"score": 9.5}}
        """
        evaluation_result = self.evaluation_llm.invoke(accuracy_prompt)
        return self._extract_score(evaluation_result.content)

    def evaluate_scientific_correctness(self, response: str) -> float:
        correctness_prompt = f"""
        You are a scientific expert evaluating the scientific correctness and grounding of an AI assistant's response about fitness.
        Response to Evaluate: {response}
        Instructions: Evaluate if the scientific claims made (if any) are correct and generally accepted. Penalize pseudoscience, bro-science, or claims lacking credible scientific backing.
        Output only a single JSON object containing the key "score" with the overall scientific correctness score (0-10). Example: {{"score": 8.0}}
        """
        evaluation_result = self.evaluation_llm.invoke(correctness_prompt)
        return self._extract_score(evaluation_result.content)

    def evaluate_practical_applicability(self, query: str, response: str) -> float:
        applicability_prompt = f"""
        You are evaluating how practically applicable an AI assistant's fitness advice is for a typical user.
        Query: {query}
        Response to Evaluate: {response}
        Instructions: Assess how actionable and realistic the advice is. Is it presented clearly? Can a user reasonably implement the suggestions? Does it require specialized equipment or knowledge not commonly available?
        Output only a single JSON object containing the key "score" with the overall practical applicability score (0-10). Example: {{"score": 7.0}}
        """
        evaluation_result = self.evaluation_llm.invoke(applicability_prompt)
        return self._extract_score(evaluation_result.content)

    def evaluate_safety_consideration(self, response: str) -> float:
        safety_prompt = f"""
        You are evaluating how well an AI assistant's fitness advice considers safety aspects.
        Response to Evaluate: {response}
        Instructions: Assess if the response includes necessary safety precautions, warnings about potential risks (e.g., injury risk with certain exercises, consulting doctors), or suggestions for modifications based on fitness level or contraindications. Penalize advice that could be dangerous if followed literally without caution.
        Output only a single JSON object containing the key "score" with the overall safety consideration score (0-10). Example: {{"score": 6.5}}
        """
        evaluation_result = self.evaluation_llm.invoke(safety_prompt)
        return self._extract_score(evaluation_result.content)

    def evaluate_retrieval_precision(self, query: str, contexts: List[str]) -> float:
        # Re-use context_precision as retrieval precision
        return self.evaluate_context_precision(query, contexts)

    def evaluate_retrieval_recall(self, query: str, contexts: List[str], ground_truth: str) -> float:
        if not contexts or not ground_truth: return 0.0
        recall_prompt = f"""
        You are evaluating the recall of a retrieval system for a fitness-related query. Recall measures how much of the necessary information from the ideal answer is covered by the retrieved contexts.
        Query: {query}
        Ground Truth Answer (contains ideal information): {ground_truth}
        Retrieved Contexts: {json.dumps(contexts, indent=2)}
        Instructions: Identify the key pieces of information present in the Ground Truth Answer. Determine how many of these key points are substantially present or supported within *any* of the Retrieved Contexts.
        Calculate recall score as: (number of key points from ground truth found in contexts) / (total number of key points in ground truth) * 10.
        Output only a single JSON object containing the key "score" with the calculated numeric recall score (0-10). Example: {{"score": 6.0}}
        """
        evaluation_result = self.evaluation_llm.invoke(recall_prompt)
        return self._extract_score(evaluation_result.content)

    def evaluate_answer_completeness(self, response: str, ground_truth: str) -> float:
        if not ground_truth: return 0.0 # Cannot evaluate completeness without ground truth
        completeness_prompt = f"""
        You are evaluating the completeness of an AI assistant's response compared to a ground truth answer for a fitness query. Completeness measures if all aspects of the ground truth are addressed in the response.
        Ground Truth Answer: {ground_truth}
        Response to Evaluate: {response}
        Instructions: Identify the key information points or sub-questions addressed in the Ground Truth Answer. Determine how many of these key points are also substantially covered in the Response to Evaluate.
        Calculate completeness score as: (number of key points from ground truth covered in response) / (total number of key points in ground truth) * 10.
        Output only a single JSON object containing the key "score" with the calculated numeric completeness score (0-10). Example: {{"score": 7.0}}
        """
        evaluation_result = self.evaluation_llm.invoke(completeness_prompt)
        return self._extract_score(evaluation_result.content)

    def evaluate_answer_conciseness(self, query: str, response: str) -> float:
        conciseness_prompt = f"""
        You are evaluating the conciseness of an AI assistant's response to a fitness-related query.
        Query: {query}
        Response to Evaluate: {response}
        Instructions: Assess if the response is direct and to the point, avoiding unnecessary jargon, repetition, or verbosity, while still being adequately informative to answer the query. Penalize overly long or rambling answers.
        Output only a single JSON object containing the key "score" with the overall conciseness score (0-10, where 10 is perfectly concise). Example: {{"score": 8.0}}
        """
        evaluation_result = self.evaluation_llm.invoke(conciseness_prompt)
        return self._extract_score(evaluation_result.content)

    def evaluate_answer_helpfulness(self, query: str, response: str) -> float:
        helpfulness_prompt = f"""
        You are evaluating the overall helpfulness of an AI assistant's response for a user asking a fitness-related query.
        Query: {query}
        Response to Evaluate: {response}
        Instructions: Consider if the response directly and accurately answers the query, provides actionable information, is easy to understand, and addresses the likely intent behind the user's question in a constructive manner.
        Output only a single JSON object containing the key "score" with the overall helpfulness score (0-10, where 10 is extremely helpful). Example: {{"score": 8.5}}
        """
        evaluation_result = self.evaluation_llm.invoke(helpfulness_prompt)
        return self._extract_score(evaluation_result.content)

    # --- (End of evaluate_* methods) ---


    def evaluate_response(self, query: str, response: str, contexts: List[str], ground_truth: str) -> Dict[str, float]:
        # ... (method unchanged) ...
        scores = {}
        if not self.evaluation_llm or not self.embeddings:
             logger.error("Cannot evaluate response: Evaluation LLM or Embeddings model not initialized.")
             return {"overall": 0.0}

        for metric_name, metric_fn in self.evaluation_metrics.items():
            logger.debug(f"Evaluating metric: {metric_name}")
            try:
                sig = inspect.signature(metric_fn)
                params = sig.parameters
                kwargs_to_pass = {}
                if 'query' in params: kwargs_to_pass['query'] = query
                if 'response' in params: kwargs_to_pass['response'] = response
                if 'contexts' in params: kwargs_to_pass['contexts'] = contexts
                if 'ground_truth' in params: kwargs_to_pass['ground_truth'] = ground_truth

                score = metric_fn(**kwargs_to_pass)
                # Ensure score is float and clamped between 0 and 10
                scores[metric_name] = max(0.0, min(10.0, float(score)))
                logger.debug(f"Metric {metric_name} score: {scores[metric_name]:.2f}")

            except Exception as e:
                logger.error(f"Error evaluating metric '{metric_name}': {e}", exc_info=True)
                scores[metric_name] = 0.0

        weighted_sum = 0
        total_weight = 0
        for metric, weight in self.metric_weights.items():
            if metric in scores:
                weighted_sum += scores[metric] * weight
                total_weight += weight
            else:
                logger.warning(f"Metric '{metric}' defined in weights but not found in calculated scores.")

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
                _, contexts = rag.answer_question(query, return_contexts=True)
                if isinstance(contexts, list) and all(isinstance(c, str) for c in contexts):
                    return contexts
                elif isinstance(contexts, list):
                     return [getattr(c, 'page_content', str(c)) for c in contexts]
                else:
                    logger.warning(f"Got contexts in unexpected format from {implementation_name}: {type(contexts)}")
                    return []
            # Fallback if context retrieval isn't built into answer_question
            elif hasattr(rag, 'retrieve_contexts'):
                 logger.debug(f"Using separate retrieve_contexts method for {implementation_name}")
                 contexts = rag.retrieve_contexts(query)
                 if isinstance(contexts, list) and all(isinstance(c, str) for c in contexts):
                    return contexts
                 elif isinstance(contexts, list):
                     return [getattr(c, 'page_content', str(c)) for c in contexts]
                 else:
                    logger.warning(f"Got contexts in unexpected format from {implementation_name}.retrieve_contexts: {type(contexts)}")
                    return []
            else:
                logger.warning(f"Implementation {implementation_name} lacks suitable method supporting context retrieval. Cannot get contexts.")
                return []
        except Exception as e:
            logger.error(f"Error retrieving contexts from {implementation_name} for query '{query[:50]}...': {e}", exc_info=True)
            return []

    def evaluate_implementation(self, rag_instance, implementation_name=None):
        # ... (method largely unchanged, ensure mlflow_tracker check) ...
        if implementation_name is None:
            cls_name = rag_instance.__class__.__name__.lower()
            if "advanced" in cls_name: implementation_name = "advanced"
            elif "modular" in cls_name: implementation_name = "modular"
            elif "raptor" in cls_name: implementation_name = "raptor"
            else: implementation_name = cls_name
        logger.info(f"--- Evaluating RAG Implementation: {implementation_name} ---")
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
                 if hasattr(rag, 'answer_question') and 'return_contexts' in inspect.signature(rag.answer_question).parameters:
                    response, contexts = rag.answer_question(query, return_contexts=True)
                    if isinstance(contexts, list) and contexts and not isinstance(contexts[0], str):
                         contexts = [getattr(c, 'page_content', str(c)) for c in contexts]
                    elif not isinstance(contexts, list): contexts = []
                 else:
                    response = rag.answer_question(query)
                    # Check if contexts are needed based on active metrics
                    needs_contexts = any(m in ["faithfulness", "context_relevancy", "context_precision", "retrieval_recall", "retrieval_precision"] for m in self.evaluation_metrics)
                    if needs_contexts:
                         contexts = self.get_retrieved_contexts(implementation_name, query)
            except Exception as e:
                logger.error(f"Error getting response/contexts from {implementation_name} for query '{query[:50]}...': {e}", exc_info=True)
                response = f"Error during answer generation: {e}"
                contexts = []
            end_time = time.time()
            response_time = end_time - start_time
            total_response_time += response_time
            logger.debug(f"Retrieved {len(contexts)} contexts. Response time: {response_time:.2f}s")
            try:
                evaluation = self.evaluate_response(query, response, contexts, ground_truth)
            except Exception as e:
                logger.error(f"Error during evaluation call for query '{query[:50]}...': {e}", exc_info=True)
                evaluation = {}
            all_results_for_impl.append({
                "query": query, "response": response, "ground_truth": ground_truth,
                "contexts": contexts, "evaluation": evaluation, "response_time": response_time
            })
            logger.info(f"Query {i+1} ({implementation_name}) completed. Overall score: {evaluation.get('overall', 0.0):.2f}/10.0")
        avg_scores = {}
        if all_results_for_impl:
            all_metrics_found = list(all_results_for_impl[0].get("evaluation", {}).keys())
            for metric in all_metrics_found:
                 valid_scores = [r["evaluation"].get(metric, 0.0) for r in all_results_for_impl if metric in r.get("evaluation", {})]
                 avg_scores[metric] = np.mean(valid_scores) if valid_scores else 0.0
        else: logger.warning(f"No results generated for implementation {implementation_name}, cannot calculate average scores.")
        avg_response_time = (total_response_time / len(self.test_queries)) if self.test_queries else 0
        if self.mlflow_tracker:
            logger.info(f"Logging results for {implementation_name} to MLflow...")
            parameters_to_log = {
                "embedding_model": getattr(rag, "embedding_model_name", getattr(rag, "embedding_model", "unknown")),
                "llm_model": getattr(rag, "llm_model_name", getattr(rag, "llm_model", "unknown")),
                "chunk_size": getattr(rag, "chunk_size", "unknown"),
                "chunk_overlap": getattr(rag, "chunk_overlap", "unknown"),
                "retrieval_k": getattr(rag, "retrieval_k", "unknown"),
                "temperature": getattr(rag, "temperature", "unknown"),
                "num_test_queries": len(self.test_queries),
                "evaluation_llm": self.evaluation_llm.model_name,
            }
            # Remove parameters with "unknown" value before logging
            parameters_to_log = {k: v for k, v in parameters_to_log.items() if v != "unknown"}

            # Log average scores as MLflow metrics
            metrics_to_log = {f"avg_{k}": v for k, v in avg_scores.items()}
            metrics_to_log["avg_response_time"] = avg_response_time

            try:
                 # Log everything under a single run for this implementation
                 run_name = f"RAG_Eval_{implementation_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                 with self.mlflow_tracker.start_run(run_name=run_name, tags={"implementation": implementation_name}):
                      self.mlflow_tracker.log_parameters(parameters_to_log)
                      self.mlflow_tracker.log_metrics(metrics_to_log)
                      # Log detailed results per query as a JSON artifact
                      detailed_results_str = json.dumps(all_results_for_impl, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))
                      self.mlflow_tracker.log_text(detailed_results_str, "detailed_evaluation_results.json")
                 logger.info(f"Successfully logged results for {implementation_name} to MLflow.")
            except Exception as mlflow_err:
                 logger.error(f"Failed to log results to MLflow for {implementation_name}: {mlflow_err}", exc_info=True)
                 if self.mlflow_tracker.is_active_run(): self.mlflow_tracker.end_run() # Ensure run ends on error
        else: logger.warning("MLflow tracker not available. Skipping MLflow logging.")
        return {
            "implementation": implementation_name,
            "results": all_results_for_impl,
            "average_scores": avg_scores,
            "average_response_time": avg_response_time
        }

    def compare_implementations(self) -> Dict[str, Any]:
        """Compare all RAG implementations."""
        logger.info("--- Starting RAG Implementation Comparison ---")
        comparison_data = {} # Store results per implementation
        if not self.rag_implementations:
             logger.error("No RAG implementations were initialized. Cannot compare.")
             return {"error": "No RAG implementations initialized."}
        for impl_name, rag_instance in self.rag_implementations.items():
            impl_results = self.evaluate_implementation(rag_instance, impl_name)
            if "error" not in impl_results:
                comparison_data[impl_name] = impl_results
            else:
                logger.error(f"Evaluation failed for {impl_name}: {impl_results['error']}")
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
            except ValueError: logger.warning("Could not determine best implementation (no valid scores found).")
        else: logger.warning("No valid results to compare implementations.")
        final_comparison_results = {
            "comparison": comparison_data,
            "best_implementation": best_implementation,
            "best_score": best_score
        }
        # Save results to GCS OR Log to MLflow
        self.save_results_to_gcs(final_comparison_results)
        # self.log_comparison_to_mlflow(final_comparison_results) # Alternative
        self.print_summary(final_comparison_results) # Print to console logs
        logger.info("--- RAG Implementation Comparison Finished ---")
        return final_comparison_results

    # --- UPDATED Save results to GCS ---
    def save_results_to_gcs(self, results: Dict[str, Any]) -> None:
        """Save final comparison results to GCS."""
        if not gcs_utils_available:
             logger.error("Cannot save results to GCS because gcs_utils failed to import.")
             return

        # Create a unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_blob = f"rag_comparison_results_{timestamp}.json"
        gcs_path = f"gs://{EVALUATION_OUTPUT_BUCKET}/{output_blob}"
        logger.info(f"Attempting to save final comparison results to {gcs_path}")

        try:
            json_string = json.dumps(results, indent=2, ensure_ascii=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))
            success = upload_string_to_gcs(EVALUATION_OUTPUT_BUCKET, output_blob, json_string)
            if success:
                logger.info(f"Final comparison results saved to {gcs_path}")
            else:
                logger.error(f"Failed to save final comparison results to GCS.")
        except Exception as e:
             logger.error(f"Failed to save final comparison results to {gcs_path}: {e}", exc_info=True)

    # --- (Optional) Alternative: Log comparison summary to MLflow ---
    def log_comparison_to_mlflow(self, results: Dict[str, Any]):
         if self.mlflow_tracker:
             results_json_string = json.dumps(results, indent=2, ensure_ascii=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))
             try:
                  summary_run_name = f"RAG_Comparison_Summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                  with self.mlflow_tracker.start_run(run_name=summary_run_name, tags={"evaluation_type": "rag_comparison_summary"}):
                      self.mlflow_tracker.log_text(results_json_string, "comparison_summary_results.json")
                      # Log best score/implementation as params/metrics too if desired
                      if results.get("best_implementation"):
                           self.mlflow_tracker.log_param("best_implementation", results["best_implementation"])
                           self.mlflow_tracker.log_metric("best_overall_score", results["best_score"])
                  logger.info("Logged comparison summary to MLflow.")
             except Exception as e:
                  logger.error(f"Failed to log comparison summary to MLflow: {e}", exc_info=True)
                  if self.mlflow_tracker.is_active_run(): self.mlflow_tracker.end_run()
         else:
              logger.warning("MLflow tracker not available, cannot log comparison summary.")


    def print_summary(self, results: Dict[str, Any]) -> None:
        # --- UPDATED Print Summary Path ---
        print("\n" + "="*80)
        print("ADVANCED RAG IMPLEMENTATION COMPARISON RESULTS")
        print("="*80)
        if "best_implementation" in results and results["best_implementation"]:
            print(f"\nBest implementation: {results['best_implementation'].upper()}")
            print(f"Best overall score: {results['best_score']:.2f}/10.0")
            print("\nDetailed average scores by implementation:")
            for implementation, implementation_results in results.get("comparison", {}).items():
                if "average_scores" in implementation_results:
                    print(f"\n{implementation.upper()} RAG:")
                    avg_scores = implementation_results["average_scores"]
                    metric_order = [m for m in avg_scores if m != 'overall'] + (['overall'] if 'overall' in avg_scores else [])
                    for metric in metric_order:
                        score = avg_scores.get(metric, 0.0)
                        print(f"    {metric.replace('_', ' ').title()}: {score:.2f}/10.0")
                    print(f"  Avg Response Time: {implementation_results.get('average_response_time', 0):.2f}s")
                else:
                    print(f"\n{implementation.upper()} RAG: No average scores calculated.")
        else:
            print("\nNo valid comparison results available.")

        # Update path to reflect GCS storage
        gcs_path_example = f"gs://{EVALUATION_OUTPUT_BUCKET}/rag_comparison_results_YYYYMMDD_HHMMSS.json"
        print(f"\nDetailed evaluation results saved to GCS bucket: {EVALUATION_OUTPUT_BUCKET} (example path: {gcs_path_example})")
        print("Detailed per-implementation results logged to MLflow.")
        print("="*80)

# --- Main Guard ---
def main():
    # ... (function unchanged) ...
    parser = argparse.ArgumentParser(description="RAG Evaluation for PersonalTrainerAI")
    parser.add_argument("--output-dir", type=str, default="/tmp/rag_eval_output", help="Temporary directory if needed")
    parser.add_argument("--evaluation-model", type=str, default="gpt-4o-mini", help="LLM model to use for evaluation")
    parser.add_argument("--implementation", type=str, choices=["advanced", "modular", "raptor", "all"],
                        default="all", help="RAG implementation to evaluate")
    parser.add_argument("--num-queries", type=int, default=len(TEST_QUERIES_WITH_GROUND_TRUTH),
                        help=f"Number of test queries to evaluate (max {len(TEST_QUERIES_WITH_GROUND_TRUTH)})")
    args = parser.parse_args()

    test_queries = TEST_QUERIES_WITH_GROUND_TRUTH[:min(args.num_queries, len(TEST_QUERIES_WITH_GROUND_TRUTH))]

    try:
        evaluator = AdvancedRAGEvaluator(
            output_dir=args.output_dir,
            test_queries=test_queries,
            evaluation_llm_model=args.evaluation_model
        )

        if args.implementation == "all":
            results = evaluator.compare_implementations()
        else:
            if args.implementation in evaluator.rag_implementations:
                 results = evaluator.evaluate_implementation(evaluator.rag_implementations[args.implementation], args.implementation)
            else:
                 logger.error(f"Error: Requested implementation '{args.implementation}' was not initialized or not found.")
                 results = {}

        # No need to call print_summary here, compare_implementations does it
        # if results and "error" not in results:
        #     evaluator.print_summary(results) # Already called within compare_implementations

    except Exception as e:
        logger.error(f"Evaluation script failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()