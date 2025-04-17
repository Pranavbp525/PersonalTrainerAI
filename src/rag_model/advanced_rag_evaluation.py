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
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community import HuggingFaceEmbeddings
from src.rag_model.mlflow.mlflow_rag_tracker import MLflowRAGTracker
import inspect  # Import the inspect module
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

        self.evaluation_llm = ChatOpenAI(
            model_name=evaluation_llm_model,
            temperature=0.0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        self.embeddings = HuggingFaceEmbeddings(
            model=embedding_model,
        )

        self.rag_implementations = self._initialize_rag_implementations()

        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        logger.info(f"Initializing MLflowRAGTracker with tracking URI: {mlflow_tracking_uri}")
        self.mlflow_tracker = MLflowRAGTracker(
            experiment_name="rag_evaluation",
            tracking_uri=mlflow_tracking_uri
        )

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
            "faithfulness": 0.15,
            "answer_relevancy": 0.1,
            "context_relevancy": 0.075,
            "context_precision": 0.075,
            "fitness_domain_accuracy": 0.1,
            "scientific_correctness": 0.075,
            "practical_applicability": 0.075,
            "safety_consideration": 0.05,
            "retrieval_precision": 0.075,
            "retrieval_recall": 0.075,
            "answer_completeness": 0.05,
            "answer_conciseness": 0.05,
            "answer_helpfulness": 0.05
        }

    def _initialize_rag_implementations(self):
        """Initialize RAG implementations with proper imports."""
        implementations = {}
        try:
            from src.rag_model.advanced_rag import AdvancedRAG
            implementations["advanced"] = AdvancedRAG()
            logger.info("Advanced RAG initialized")
        except Exception as e:
            logger.error(f"Failed to init Advanced RAG: {e}")

        try:
            from src.rag_model.modular_rag import ModularRAG
            implementations["modular"] = ModularRAG()
            logger.info("Modular RAG initialized")
        except Exception as e:
            logger.error(f"Failed to init Modular RAG: {e}")

        try:
            from src.rag_model.raptor_rag import RaptorRAG
            implementations["raptor"] = RaptorRAG()
            logger.info("Raptor RAG initialized")
        except Exception as e:
            logger.error(f"Failed to init Raptor RAG: {e}")
        return implementations

    def _extract_score(self, evaluation_text: str) -> float:
        """Extract score from LLM evaluation text."""
        try:
            if isinstance(evaluation_text, (int, float)):
                return float(evaluation_text)
            result = json.loads(evaluation_text)
            if "score" in result:
                return float(result["score"])
            score_match = r'"score":\s*([0-9.]+)'
            match = re.search(score_match, evaluation_text)
            if match:
                return float(match.group(1))
            score_match = r'([0-9.]+)\s*/\s*10'
            match = re.search(score_match, evaluation_text)
            if match:
                return float(match.group(1))
            return 5.0  # Default score
        except Exception as e:
            logger.error(f"Error extracting score: {e}")
            return 5.0

    def evaluate_faithfulness(self, query: str, response: str, contexts: List[str], ground_truth: str) -> float:
        """Evaluate faithfulness of the response."""
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
        """Evaluate relevancy of the response to the query."""
        query_embedding = self.embeddings.embed_query(query)
        response_embedding = self.embeddings.embed_query(response)
        similarity = cosine_similarity(np.array(query_embedding).reshape(1, -1), np.array(response_embedding).reshape(1, -1))[0][0]
        base_score = similarity * 10

        relevancy_prompt = f"""
        You are evaluating the relevancy of an AI assistant's response to a fitness-related query.
        Query: {query}
        Response to Evaluate: {response}
        Instructions: Determine how directly the response addresses the question.
        The initial relevancy score is: {base_score:.2f}/10. Adjust this score based on your evaluation. Provide only the score.
        """
        evaluation_result = self.evaluation_llm.invoke(relevancy_prompt)
        return self._extract_score(evaluation_result.content)

    def evaluate_context_relevancy(self, query: str, contexts: List[str]) -> float:
        """Evaluate relevancy of the retrieved contexts to the query."""
        if not contexts:
            return 0.0

        query_embedding = self.embeddings.embed_query(query)
        context_embeddings = [self.embeddings.embed_query(context) for context in contexts]
        similarities = cosine_similarity(np.array(query_embedding).reshape(1, -1), np.array(context_embeddings))
        avg_similarity = np.mean(similarities)
        return avg_similarity * 10

    def evaluate_context_precision(self, query: str, contexts: List[str]) -> float:
        """Evaluate precision of the retrieved contexts."""
        if not contexts:
            return 0.0

        precision_prompt = f"""
        You are evaluating the precision of retrieved contexts for a fitness-related query.
        Query: {query}
        Retrieved Contexts: {json.dumps(contexts, indent=2)}
        Instructions: Determine if each context contains information relevant to answering the query.
        Calculate precision as: (relevant contexts) / (total contexts) * 10. Provide only the score.
        """
        evaluation_result = self.evaluation_llm.invoke(precision_prompt)
        return self._extract_score(evaluation_result.content)

    def evaluate_fitness_domain_accuracy(self, query: str, response: str) -> float:
        """Evaluate accuracy of fitness-specific information in the response."""
        accuracy_prompt = f"""
        You are a fitness expert evaluating the domain-specific accuracy of an AI assistant's response.
        Query: {query}
        Response to Evaluate: {response}
        Instructions: Evaluate fitness-specific claims for scientific accuracy.
        Rate the overall fitness domain accuracy on a scale of 0-10. Provide only the score.
        """
        evaluation_result = self.evaluation_llm.invoke(accuracy_prompt)
        return self._extract_score(evaluation_result.content)

    def evaluate_scientific_correctness(self, response: str) -> float:
        """Evaluate scientific correctness of the response."""
        correctness_prompt = f"""
        You are a scientific expert evaluating the scientific correctness of an AI assistant's response about fitness.
        Response to Evaluate: {response}
        Instructions: Evaluate scientific claims for correctness.
        Rate the overall scientific correctness on a scale of 0-10. Provide only the score.
        """
        evaluation_result = self.evaluation_llm.invoke(correctness_prompt)
        return self._extract_score(evaluation_result.content)

    def evaluate_practical_applicability(self, query: str, response: str) -> float:
        """Evaluate practical applicability of the response."""
        applicability_prompt = f"""
        You are evaluating how practically applicable an AI assistant's fitness advice is for users.
        Query: {query}
        Response to Evaluate: {response}
        Instructions: Assess how actionable the advice is for an average person.
        Rate the overall practical applicability on a scale of 0-10. Provide only the score.
        """
        evaluation_result = self.evaluation_llm.invoke(applicability_prompt)
        return self._extract_score(evaluation_result.content)

    def evaluate_safety_consideration(self, response: str) -> float:
        """Evaluate safety considerations in the response."""
        safety_prompt = f"""
        You are evaluating how well an AI assistant's fitness advice considers safety aspects.
        Response to Evaluate: {response}
        Instructions: Assess whether the response includes appropriate safety precautions.
        Rate the overall safety consideration on a scale of 0-10. Provide only the score.
        """
        evaluation_result = self.evaluation_llm.invoke(safety_prompt)
        return self._extract_score(evaluation_result.content)

    def evaluate_retrieval_precision(self, query: str, contexts: List[str]) -> float:
        """Evaluate precision of the retrieval."""
        return self.evaluate_context_precision(query, contexts)

    def evaluate_retrieval_recall(self, query: str, contexts: List[str], ground_truth: str) -> float:
        """Evaluate recall of the retrieval."""
        if not contexts:
            return 0.0

        recall_prompt = f"""
        You are evaluating the recall of a retrieval system for a fitness-related query.
        Query: {query}
        Ground Truth Answer: {ground_truth}
        Retrieved Contexts: {json.dumps(contexts, indent=2)}
        Instructions: Identify key points in the ground truth and determine if they are covered in the contexts.
        Calculate recall as: (covered key points) / (total key points) * 10. Provide only the score.
        """
        evaluation_result = self.evaluation_llm.invoke(recall_prompt)
        return self._extract_score(evaluation_result.content)

    def evaluate_answer_completeness(self, response: str, ground_truth: str) -> float:
        """Evaluate completeness of the answer."""
        completeness_prompt = f"""
        You are evaluating the completeness of an AI assistant's response compared to a ground truth answer.
        Ground Truth Answer: {ground_truth}
        Response to Evaluate: {response}
        Instructions: Identify key points in the ground truth and determine if they are covered in the response.
        Calculate completeness as: (covered key points) / (total key points) * 10. Provide only the score.
        """
        evaluation_result = self.evaluation_llm.invoke(completeness_prompt)
        return self._extract_score(evaluation_result.content)

    def evaluate_answer_conciseness(self, query: str, response: str) -> float:
        """Evaluate conciseness of the answer."""
        conciseness_prompt = f"""
        You are evaluating the conciseness of an AI assistant's response to a fitness-related query.
        Response to Evaluate: {response}
        Instructions: Assess whether the response is concise while being informative.
        Rate the overall conciseness on a scale of 0-10. Provide only the score.
        """
        evaluation_result = self.evaluation_llm.invoke(conciseness_prompt)
        return self._extract_score(evaluation_result.content)

    def evaluate_answer_helpfulness(self, query: str, response: str) -> float:
        """Evaluate helpfulness of the answer."""
        helpfulness_prompt = f"""
        You are evaluating how helpful an AI assistant's response is for a fitness-related query.
        Query: {query}
        Response to Evaluate: {response}
        Instructions: Assess how well the response addresses the user's needs and provides actionable information.
        Rate the overall helpfulness on a scale of 0-10. Provide only the score.
        """
        evaluation_result = self.evaluation_llm.invoke(helpfulness_prompt)
        return self._extract_score(evaluation_result.content)
    
    def evaluate_response(
        self,
        query: str,
        response: str,
        contexts: List[str],
        ground_truth: str
    ) -> Dict[str, float]:
        """Evaluate a response based on multiple metrics."""
        scores = {}
        for metric_name, metric_fn in self.evaluation_metrics.items():
            try:
                if metric_name in ["faithfulness", "retrieval_recall", "answer_completeness"]:
                    scores[metric_name] = metric_fn(query, response, contexts, ground_truth)
                elif metric_name in ["context_relevancy", "context_precision", "retrieval_precision"]:
                    scores[metric_name] = metric_fn(query, contexts)
                else:
                    scores[metric_name] = metric_fn(query, response)
            except Exception as e:
                logger.error(f"Error evaluating {metric_name}: {e}")
                scores[metric_name] = 0.0  # Assign a default score

        weighted_sum = sum(scores[metric] * self.metric_weights[metric] for metric in self.metric_weights)
        scores["overall"] = weighted_sum
        return scores

    def get_retrieved_contexts(self, implementation_name: str, query: str) -> List[str]:
        """Get retrieved contexts for a query."""
        if implementation_name not in self.rag_implementations:
            logger.error(f"Unknown implementation: {implementation_name}")
            return []

        rag = self.rag_implementations[implementation_name]

        try:
            if hasattr(rag, 'answer_question') and 'return_contexts' in inspect.signature(rag.answer_question).parameters:
                _, contexts = rag.answer_question(query, return_contexts=True)
                return contexts
            else:
                logger.warning(f"Implementation {implementation_name} does not support returning contexts")
                return []
        except Exception as e:
            logger.error(f"Error retrieving contexts from {implementation_name}: {e}")
            return []

    def evaluate_implementation(self, rag_instance, implementation_name=None):
        """Evaluate a specific RAG implementation."""
        if isinstance(rag_instance, str):
            implementation_key = rag_instance.lower()
            if implementation_key not in self.rag_implementations:
                logger.error(f"Unknown implementation: {rag_instance}")
                return {"error": f"Unknown implementation: {rag_instance}"}
            else:
                rag_instance = self.rag_implementations[implementation_key]
                implementation_name = implementation_key
        else:
            cls_name = rag_instance.__class__.__name__.lower()
            if "advanced" in cls_name:
                implementation_name = "advanced"
            elif "modular" in cls_name:
                implementation_name = "modular"
            elif "raptor" in cls_name:
                implementation_name = "raptor"
            else:
                implementation_name = cls_name

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

            start_time = time.time()
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

            logger.info(f"Retrieved {len(contexts)} contexts for evaluation")

            try:
                evaluation = self.evaluate_response(query, response, contexts, ground_truth)
            except Exception as e:
                logger.error(f"Error during evaluation: {e}")
                evaluation = {}

            results.append({
                "query": query,
                "response": response,
                "ground_truth": ground_truth,
                "contexts": contexts,
                "evaluation": evaluation,
                "response_time": response_time
            })

            logger.info(f"Query {i+1} completed. Overall score: {evaluation.get('overall', 0.0):.2f}/10.0")

        average_scores = {}
        for metric in self.evaluation_metrics:
            average_scores[metric] = np.mean([r["evaluation"].get(metric, 0.0) for r in results])
        average_scores["overall"] = np.mean([r["evaluation"].get("overall", 0.0) for r in results])

        average_response_time = total_response_time / len(self.test_queries)

        parameters = {
            "embedding_model": getattr(rag, "embedding_model", "unknown"),
            "llm_model": getattr(rag, "llm_model", "unknown"),
        }

        self.mlflow_tracker.log_rag_evaluation_results(
            results=results,
            implementation_name=implementation_name,
            parameters=parameters
        )

        return {
            "implementation": implementation_name,
            "results": results,
            "average_scores": average_scores,
            "average_response_time": average_response_time
        }

    def compare_implementations(self) -> Dict[str, Any]:
        """Compare all RAG implementations."""
        logger.info("Starting RAG implementation comparison")

        results = {}
        for implementation_name in self.rag_implementations:
            results[implementation_name] = self.evaluate_implementation(self.rag_implementations[implementation_name], implementation_name)

        best_implementation = max(results, key=lambda k: results[k]["average_scores"].get("overall", 0))
        best_score = results[best_implementation]["average_scores"].get("overall", 0)

        comparison_results = {
            "comparison": results,
            "best_implementation": best_implementation,
            "best_score": best_score
        }

        self.save_results(comparison_results)
        self.print_summary(comparison_results)

        return comparison_results

    def save_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results to file."""
        output_file = os.path.join(self.output_dir, "advanced_evaluation_results.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Evaluation results saved to {output_file}")

    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print a summary of the evaluation results."""
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
                    for metric, score in implementation_results["average_scores"].items():
                        print(f"    {metric.replace('_', ' ').title()}: {score:.2f}/10.0")
                    print(f"  Response Time: {implementation_results.get('average_response_time', 0):.2f}s")
        else:
            print("\nNo valid comparison results available.")

        print("\nDetailed evaluation results saved to:", os.path.join(self.output_dir, "advanced_evaluation_results.json"))
        print("="*80)

def main():
    """Main function to run the RAG evaluation."""
    parser = argparse.ArgumentParser(description="RAG Evaluation for PersonalTrainerAI")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--evaluation-model", type=str, default="gpt-4o-mini", help="LLM model to use for evaluation")
    parser.add_argument("--implementation", type=str, choices=["advanced", "modular", "raptor", "all"],
                        default="all", help="RAG implementation to evaluate")
    parser.add_argument("--num-queries", type=int, default=5,
                        help="Number of test queries to evaluate (max 5)")
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
        results = evaluator.evaluate_implementation(evaluator.rag_implementations[args.implementation], args.implementation)

    evaluator.print_summary(results)

if __name__ == "__main__":
    main()