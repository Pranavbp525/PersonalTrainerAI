# src/rag_model/advanced_rag_evaluation.py
"""
Simplified RAG Evaluation Framework for PersonalTrainerAI with MLflow & GCS Output
"""
import sys
import os

# --- Datetime Import ---
from datetime import datetime # <<< ADDED: Import the datetime object

# Add project root to sys.path if needed (primarily for standalone runs)
# This path should align with how the code is structured relative to the execution point.
# Inside Airflow container, '/opt/airflow/app' is the project root.
project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root_path not in sys.path:
    # Use print here as logger might not be configured yet
    print(f"Appending to sys.path for standalone run: {project_root_path}")
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

# Use recommended langchain imports
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    logging.error("Could not import ChatOpenAI from langchain_openai. Ensure langchain-openai is installed.")
    # Define a dummy class or raise error if critical
    class ChatOpenAI: pass # Dummy

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    logging.info("Using HuggingFaceEmbeddings from langchain_huggingface.")
except ImportError:
    logging.warning("Could not import from langchain_huggingface. Falling back to langchain_community.")
    from langchain_community.embeddings import HuggingFaceEmbeddings

# Import local modules using absolute path from src (assuming src is in PYTHONPATH)
try:
    from src.rag_model.mlflow.mlflow_rag_tracker import MLflowRAGTracker
except ImportError as e:
     logging.error(f"Failed to import MLflowRAGTracker: {e}. MLflow logging will be disabled.")
     MLflowRAGTracker = None # Define as None if import fails

import inspect
import re

# Attempt to import GCS utils - handle failure gracefully
gcs_utils_available = False
try:
    # Use absolute path from src, as DAG adds src to sys.path
    from src.data_pipeline.gcs_utils import upload_string_to_gcs
    gcs_utils_available = True
    logging.info("Successfully imported gcs_utils.")
except ImportError:
    logging.warning("Could not import gcs_utils. Saving results to GCS will be disabled.")
    # Define dummy function
    def upload_string_to_gcs(*args, **kwargs):
        logging.error("GCS Utils not available, cannot upload string.")
        return False

# --- Configure logging ---
# Use getLogger without basicConfig; Airflow task handler will configure it.
# If run standalone, basicConfig might be needed in the __main__ block.
logger = logging.getLogger(__name__)
# Initial log to confirm module loading
logger.info(f"--- Logger {__name__} initialized ---")


# --- Load environment variables ---
# Load .env from the specified path within the container
dotenv_path = "/opt/airflow/app/.env" # Path defined in docker-compose mount
loaded_env = False
if os.path.exists(dotenv_path):
    # override=True means .env vars overwrite existing system env vars
    loaded_env = load_dotenv(dotenv_path=dotenv_path, override=True, verbose=False)
    logger.info(f"Attempted load of .env from {dotenv_path}. Success: {loaded_env}")
else:
    logger.warning(f".env file not found at {dotenv_path}. Relying on existing environment variables.")

# --- GCS Configuration for Output ---
# Get bucket name from env var, provide a default
EVALUATION_OUTPUT_BUCKET = os.getenv("EVAL_RESULTS_GCS_BUCKET", "ragllm-454718-eval-results") # <<< CREATE THIS BUCKET
logger.info(f"Evaluation results GCS Bucket: {EVALUATION_OUTPUT_BUCKET}")


# --- Test Queries (Keep unchanged) ---
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
    Evaluator for comparing different RAG implementations with MLflow and GCS output.
    """

    def __init__(
        self,
        output_dir: str = "/tmp/rag_eval_output", # Temp dir if needed locally
        test_queries: List[Dict[str, str]] = None,
        evaluation_llm_model: str = "gpt-4o-mini",
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    ):
        logger.info("--- Initializing AdvancedRAGEvaluator ---")
        self.output_dir = output_dir
        self.test_queries = test_queries or TEST_QUERIES_WITH_GROUND_TRUTH
        os.makedirs(output_dir, exist_ok=True) # Ensure temp dir exists

        # Validate required API keys from environment
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.pinecone_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_index = os.getenv("PINECONE_INDEX_NAME") # Need index name too

        if not self.openai_key: raise ValueError("OPENAI_API_KEY environment variable not set.")
        if not self.pinecone_key: raise ValueError("PINECONE_API_KEY environment variable not set.")
        if not self.pinecone_index: raise ValueError("PINECONE_INDEX_NAME environment variable not set.")

        logger.info(f"Initializing evaluation LLM: {evaluation_llm_model}")
        try:
            self.evaluation_llm = ChatOpenAI(model=evaluation_llm_model, temperature=0.0, openai_api_key=self.openai_key)
            logger.info("Evaluation LLM initialized.")
        except Exception as e: logger.error(f"Failed to init eval LLM: {e}", exc_info=True); raise

        logger.info(f"Initializing Embeddings model: {embedding_model}")
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={'device': 'cpu'})
            logger.info("Embeddings model initialized.")
        except Exception as e: logger.error(f"Failed to init Embeddings: {e}", exc_info=True); raise

        # Initialize RAG implementations
        self.rag_implementations = self._initialize_rag_implementations()

        # Initialize MLflow Tracker (handle failure)
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
        logger.info(f"Initializing MLflowRAGTracker (URI: {mlflow_tracking_uri})...")
        self.mlflow_tracker = None # Default to None
        if MLflowRAGTracker: # Check if class was imported successfully
            try:
                self.mlflow_tracker = MLflowRAGTracker(experiment_name="rag_evaluation", tracking_uri=mlflow_tracking_uri)
                logger.info("MLflowRAGTracker initialized successfully.")
            except Exception as e: logger.error(f"Failed MLflow init: {e}", exc_info=True); logger.warning("MLflow tracking disabled.")
        else: logger.warning("MLflowRAGTracker class not available. MLflow tracking disabled.")

        # Define metrics and weights (keep as before)
        self.evaluation_metrics = { "faithfulness": self.evaluate_faithfulness, "answer_relevancy": self.evaluate_answer_relevancy, "context_relevancy": self.evaluate_context_relevancy, "context_precision": self.evaluate_context_precision, "fitness_domain_accuracy": self.evaluate_fitness_domain_accuracy, "scientific_correctness": self.evaluate_scientific_correctness, "practical_applicability": self.evaluate_practical_applicability, "safety_consideration": self.evaluate_safety_consideration, "retrieval_precision": self.evaluate_retrieval_precision, "retrieval_recall": self.evaluate_retrieval_recall, "answer_completeness": self.evaluate_answer_completeness, "answer_conciseness": self.evaluate_answer_conciseness, "answer_helpfulness": self.evaluate_answer_helpfulness }
        self.metric_weights = { "faithfulness": 0.15, "answer_relevancy": 0.1, "context_relevancy": 0.075, "context_precision": 0.075, "fitness_domain_accuracy": 0.1, "scientific_correctness": 0.075, "practical_applicability": 0.075, "safety_consideration": 0.05, "retrieval_precision": 0.075, "retrieval_recall": 0.075, "answer_completeness": 0.05, "answer_conciseness": 0.05, "answer_helpfulness": 0.05 }
        logger.info("AdvancedRAGEvaluator initialized.")

    def _initialize_rag_implementations(self):
        """Initialize RAG implementations using absolute imports from src."""
        implementations = {}
        logger.info("Initializing RAG implementations...")
        # Assuming PINECONE_API_KEY and PINECONE_INDEX_NAME are validated in __init__
        try:
            # Corrected import path assuming src is in PYTHONPATH
            from src.rag_model.advanced_rag import AdvancedRAG
            # Pass env vars explicitly if needed, or rely on class internal loading
            implementations["advanced"] = AdvancedRAG(
                 # openai_api_key=self.openai_key, # Example if needed
                 # pinecone_api_key=self.pinecone_key, # Example if needed
                 # index_name=self.pinecone_index # Example if needed
            )
            logger.info("Advanced RAG initialized")
        except Exception as e: logger.error(f"Failed to init Advanced RAG: {e}", exc_info=True)

        try:
            from src.rag_model.modular_rag import ModularRAG
            implementations["modular"] = ModularRAG()
            logger.info("Modular RAG initialized")
        except Exception as e: logger.error(f"Failed to init Modular RAG: {e}", exc_info=True)

        try:
            from src.rag_model.raptor_rag import RaptorRAG
            implementations["raptor"] = RaptorRAG()
            logger.info("Raptor RAG initialized")
        except Exception as e: logger.error(f"Failed to init Raptor RAG: {e}", exc_info=True)

        logger.info(f"Initialized {len(implementations)} RAG implementations.")
        return implementations

    def _extract_score(self, evaluation_text: str) -> float:
        # (Keep this method as previously defined - seems robust)
        try:
            if isinstance(evaluation_text, (int, float)): return float(evaluation_text)
            try:
                result = json.loads(evaluation_text); score_val = result.get("score")
                if isinstance(score_val, (int, float)): return float(score_val)
                if isinstance(score_val, str): return float(score_val.strip())
            except: pass # Ignore JSON errors, try regex
            score_patterns = [r'"score":\s*([0-9]+(?:\.[0-9]+)?)', r"'score':\s*([0-9]+(?:\.[0-9]+)?)", r'score\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)', r'([0-9]+(?:\.[0-9]+)?)\s*/\s*10', r'Rating:\s*([0-9]+(?:\.[0-9]+)?)', r'(?:^|\s|\b)([0-9]+(?:\.[0-9]+)?)(?:$|\s*(?:/10)?\b)']
            for pattern in score_patterns:
                 match = re.search(pattern, evaluation_text, re.IGNORECASE)
                 if match:
                     try: score = float(match.group(1)); clamped_score = max(0.0, min(10.0, score)); logger.debug(f"Extracted score {score}, clamped to {clamped_score}"); return clamped_score
                     except: continue
            logger.warning(f"Could not extract score reliably from: {evaluation_text[:100]}..."); return 5.0
        except Exception as e: logger.error(f"Error extracting score: {e}", exc_info=True); return 5.0

    # --- Evaluation Metric Functions ---
    # --- (Keep all your evaluate_* methods as previously defined) ---
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
        try:
            query_embedding = self.embeddings.embed_query(query)
            response_embedding = self.embeddings.embed_query(response)
            query_embedding_np = np.array(query_embedding).reshape(1, -1)
            response_embedding_np = np.array(response_embedding).reshape(1, -1)
            similarity = cosine_similarity(query_embedding_np, response_embedding_np)[0][0]
            base_score = max(0, min(10, (similarity + 1) * 5))
        except Exception as e:
            logger.warning(f"Could not calculate embedding similarity for answer relevancy: {e}")
            base_score = 5.0 # Default baseline if embedding fails

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
            if not query_embedding or not context_embeddings or len(context_embeddings) != len(contexts): logger.warning("Context relevancy calculation skipped due to embedding issues."); return 0.0
            similarities = cosine_similarity(np.array(query_embedding).reshape(1, -1), np.array(context_embeddings))
            avg_similarity = np.mean(similarities)
            return max(0, min(10, (avg_similarity + 1) * 5))
        except Exception as e: logger.error(f"Error calculating context relevancy similarity: {e}"); return 0.0

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
        if not ground_truth: return 0.0
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
    # --- End Evaluation Methods ---

    def evaluate_response(self, query: str, response: str, contexts: List[str], ground_truth: str) -> Dict[str, float]:
        """Evaluate a single response across all defined metrics."""
        scores = {}
        if not self.evaluation_llm or not self.embeddings:
             logger.error("Cannot evaluate: LLM/Embeddings not initialized.")
             return {"overall": 0.0}

        for name, metric_fn in self.evaluation_metrics.items():
            logger.debug(f"Evaluating metric: {name}")
            try:
                sig = inspect.signature(metric_fn)
                kwargs = {k: v for k, v in locals().items() if k in sig.parameters} # Pass only required args
                score = metric_fn(**kwargs)
                scores[name] = max(0.0, min(10.0, float(score))) # Clamp score
                logger.debug(f"Metric {name} score: {scores[name]:.2f}")
            except Exception as e: logger.error(f"Error evaluating metric '{name}': {e}", exc_info=True); scores[name] = 0.0

        weighted_sum, total_weight = 0, 0
        for metric, weight in self.metric_weights.items():
            score = scores.get(metric) # Use get() for safety
            if score is not None:
                weighted_sum += score * weight; total_weight += weight
            else: logger.warning(f"Metric '{metric}' in weights but not scored.")

        scores["overall"] = (weighted_sum / total_weight * 10.0) if total_weight > 0 else 0.0
        logger.info(f"Calculated overall score: {scores['overall']:.2f}/10.0")
        return scores

    def get_retrieved_contexts(self, implementation_name: str, query: str) -> List[str]:
        """Retrieves contexts from a specific RAG implementation."""
        # (Keep this method as previously defined - seems robust)
        if implementation_name not in self.rag_implementations: logger.error(f"Unknown impl: {implementation_name}"); return []
        rag = self.rag_implementations[implementation_name]
        try:
            # Prefer method that returns context directly
            if hasattr(rag, 'answer_question') and 'return_contexts' in inspect.signature(rag.answer_question).parameters:
                _, contexts = rag.answer_question(query, return_contexts=True)
            elif hasattr(rag, 'retrieve_contexts'):
                contexts = rag.retrieve_contexts(query)
            else: logger.warning(f"{implementation_name} lacks context retrieval method."); return []

            # Normalize contexts to list of strings
            if isinstance(contexts, list): return [getattr(c, 'page_content', str(c)) for c in contexts]
            logger.warning(f"Contexts not list from {implementation_name}: {type(contexts)}"); return []
        except Exception as e: logger.error(f"Error retrieving contexts from {implementation_name}: {e}", exc_info=True); return []


    def evaluate_implementation(self, rag_instance, implementation_name: str) -> Dict[str, Any]:
        """Evaluates a single RAG implementation across all test queries."""
        # (Keep this method as previously defined - seems robust, uses MLflow tracker check)
        logger.info(f"--- Evaluating RAG Implementation: {implementation_name} ---")
        all_results = []
        total_time = 0
        for i, query_data in enumerate(tqdm(self.test_queries, desc=f"Evaluating {implementation_name}")):
            query, ground_truth = query_data["query"], query_data["ground_truth"]
            logger.debug(f"Processing query {i+1}: {query[:50]}...")
            start_time = time.time()
            response, contexts = f"Error: {implementation_name}", []
            try:
                if hasattr(rag_instance, 'answer_question') and 'return_contexts' in inspect.signature(rag_instance.answer_question).parameters:
                    response, contexts = rag_instance.answer_question(query, return_contexts=True)
                    if isinstance(contexts, list) and contexts and not isinstance(contexts[0], str): contexts = [getattr(c, 'page_content', str(c)) for c in contexts]
                    elif not isinstance(contexts, list): contexts = []
                else: response = rag_instance.answer_question(query); contexts = self.get_retrieved_contexts(implementation_name, query) # Fallback
            except Exception as e: logger.error(f"Error getting answer from {implementation_name}: {e}", exc_info=True); response = f"Error: {e}"
            response_time = time.time() - start_time; total_time += response_time
            logger.debug(f"Retrieved {len(contexts)} contexts. Response time: {response_time:.2f}s")
            try: evaluation = self.evaluate_response(query, response, contexts, ground_truth)
            except Exception as e: logger.error(f"Error evaluating response: {e}", exc_info=True); evaluation = {}
            all_results.append({"query": query, "response": response, "ground_truth": ground_truth, "contexts": contexts, "evaluation": evaluation, "response_time": response_time})
            logger.info(f"Query {i+1} ({implementation_name}) score: {evaluation.get('overall', 0.0):.2f}/10.0")

        avg_scores = {metric: np.mean([r["evaluation"].get(metric, 0.0) for r in all_results]) for metric in self.evaluation_metrics if any(metric in r["evaluation"] for r in all_results)}
        if "overall" in self.evaluation_metrics: avg_scores["overall"] = np.mean([r["evaluation"].get("overall", 0.0) for r in all_results]) # Recalculate overall avg
        avg_response_time = (total_time / len(self.test_queries)) if self.test_queries else 0

        if self.mlflow_tracker:
            logger.info(f"Logging {implementation_name} results to MLflow...")
            params = {"impl": implementation_name, "num_queries": len(self.test_queries), "eval_llm": self.evaluation_llm.model_name}
            metrics = {f"avg_{k}": v for k, v in avg_scores.items()}
            metrics["avg_response_time"] = avg_response_time
            try:
                run_name = f"RAG_Eval_{implementation_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                with self.mlflow_tracker.start_run(run_name=run_name, tags={"implementation": implementation_name}):
                     self.mlflow_tracker.log_parameters(params)
                     self.mlflow_tracker.log_metrics(metrics)
                     results_str = json.dumps(all_results, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))
                     self.mlflow_tracker.log_text(results_str, "detailed_eval_results.json")
                logger.info(f"Logged {implementation_name} results to MLflow.")
            except Exception as mlflow_err: logger.error(f"MLflow logging failed for {implementation_name}: {mlflow_err}", exc_info=True); self.mlflow_tracker.end_run_safe()
        else: logger.warning("MLflow tracker unavailable. Skipping logging.")
        return {"implementation": implementation_name, "results": all_results, "average_scores": avg_scores, "average_response_time": avg_response_time}


    def compare_implementations(self) -> Dict[str, Any]:
        """Compare all initialized RAG implementations."""
        logger.info("--- Starting RAG Implementation Comparison ---")
        comparison_data = {}
        if not self.rag_implementations: logger.error("No RAG implementations initialized."); return {"error": "No RAG implementations."}
        for impl_name, rag_instance in self.rag_implementations.items():
            impl_results = self.evaluate_implementation(rag_instance, impl_name)
            if impl_results and "error" not in impl_results: comparison_data[impl_name] = impl_results
            else: logger.error(f"Eval failed for {impl_name}: {impl_results.get('error', 'Unknown')}")
        best_impl, best_score = None, -1.0
        valid_results = {k: v for k, v in comparison_data.items() if v and v.get("average_scores")}
        if valid_results:
            try: best_impl = max(valid_results, key=lambda k: valid_results[k]["average_scores"].get("overall", -1.0)); best_score = valid_results[best_impl]["average_scores"].get("overall", -1.0)
            except ValueError: logger.warning("Could not determine best impl.")
        logger.info(f"Comparison finished. Best: {best_impl} (Score: {best_score:.2f})")
        final_results = {"comparison": comparison_data, "best_implementation": best_impl, "best_score": best_score}
        self.save_results_to_gcs(final_results)
        self.print_summary(final_results)
        logger.info("--- RAG Implementation Comparison Finished ---")
        return final_results

    # --- Save results to GCS ---
    def save_results_to_gcs(self, results: Dict[str, Any]) -> None:
        """Save final comparison results to GCS."""
        if not gcs_utils_available: logger.error("Cannot save to GCS: gcs_utils unavailable."); return
        # Use the datetime object imported at the top
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_blob = f"evaluation_results/rag_comparison_{timestamp}.json" # Use subfolder
        gcs_path = f"gs://{EVALUATION_OUTPUT_BUCKET}/{output_blob}"
        logger.info(f"Attempting to save final comparison results to {gcs_path}")
        try:
            json_string = json.dumps(results, indent=2, ensure_ascii=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))
            success = upload_string_to_gcs(EVALUATION_OUTPUT_BUCKET, output_blob, json_string)
            if success: logger.info(f"Final comparison results saved to {gcs_path}")
            else: logger.error(f"Failed to save final comparison results to GCS (upload failed).")
        except Exception as e: logger.error(f"Failed to save to {gcs_path}: {e}", exc_info=True)

    # --- Print Summary ---
    def print_summary(self, results: Dict[str, Any]) -> None:
        # (Keep this method as previously defined)
        print("\n" + "="*80); print("ADVANCED RAG IMPLEMENTATION COMPARISON RESULTS"); print("="*80)
        if results.get("best_implementation"):
            print(f"\nBest implementation: {results['best_implementation'].upper()}"); print(f"Best overall score: {results['best_score']:.2f}/10.0")
            print("\nDetailed average scores by implementation:")
            for impl, impl_results in results.get("comparison", {}).items():
                if impl_results and "average_scores" in impl_results:
                    print(f"\n{impl.upper()} RAG:")
                    avg_scores = impl_results["average_scores"]
                    order = sorted([m for m in avg_scores if m != 'overall']) + (['overall'] if 'overall' in avg_scores else [])
                    for metric in order: print(f"    {metric.replace('_', ' ').title()}: {avg_scores.get(metric, 0.0):.2f}/10.0")
                    print(f"  Avg Response Time: {impl_results.get('average_response_time', 0):.2f}s")
                else: print(f"\n{impl.upper()} RAG: No scores calculated.")
        else: print("\nNo valid comparison results available.")
        gcs_path_example = f"gs://{EVALUATION_OUTPUT_BUCKET}/evaluation_results/rag_comparison_YYYYMMDD_HHMMSS.json"
        print(f"\nDetailed results saved to GCS bucket: {EVALUATION_OUTPUT_BUCKET} (example path: {gcs_path_example})")
        if self.mlflow_tracker: print("Detailed per-implementation results also logged to MLflow.")
        print("="*80)

# --- Main Guard ---
def main():
    # (Keep this function as previously defined)
    parser = argparse.ArgumentParser(description="RAG Evaluation for PersonalTrainerAI")
    parser.add_argument("--output-dir", type=str, default="/tmp/rag_eval_output", help="Temporary directory if needed")
    parser.add_argument("--evaluation-model", type=str, default="gpt-4o-mini", help="LLM model for evaluation")
    parser.add_argument("--implementation", type=str, choices=["advanced", "modular", "raptor", "all"], default="all", help="RAG implementation(s) to evaluate")
    parser.add_argument("--num-queries", type=int, default=len(TEST_QUERIES_WITH_GROUND_TRUTH), help=f"Number of test queries (max {len(TEST_QUERIES_WITH_GROUND_TRUTH)})")
    args = parser.parse_args()

    # Reconfigure logging if run directly
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Running RAG evaluation script directly...")

    test_queries = TEST_QUERIES_WITH_GROUND_TRUTH[:min(args.num_queries, len(TEST_QUERIES_WITH_GROUND_TRUTH))]
    try:
        evaluator = AdvancedRAGEvaluator(output_dir=args.output_dir, test_queries=test_queries, evaluation_llm_model=args.evaluation_model)
        if args.implementation == "all": results = evaluator.compare_implementations()
        elif args.implementation in evaluator.rag_implementations: results = evaluator.evaluate_implementation(evaluator.rag_implementations[args.implementation], args.implementation)
        else: logger.error(f"Implementation '{args.implementation}' not found/initialized."); results = {}
        # Summary is printed by compare_implementations
    except Exception as e: logger.error(f"Evaluation script failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()