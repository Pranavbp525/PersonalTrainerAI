import pytest
from unittest.mock import patch, MagicMock
from src.rag_model.advanced_rag_evaluation import AdvancedRAGEvaluator, TEST_QUERIES_WITH_GROUND_TRUTH

@pytest.fixture
def evaluator():
    """Fixture to create an AdvancedRAGEvaluator instance."""
    with patch('src.rag_model.advanced_rag_evaluation.ChatOpenAI'), \
         patch('src.rag_model.advanced_rag_evaluation.OpenAIEmbeddings'):
        return AdvancedRAGEvaluator(output_dir="test_results")

def test_initialize_evaluator(evaluator):
    """Test initialization of the evaluator."""
    assert evaluator.output_dir == "test_results"
    assert evaluator.test_queries == TEST_QUERIES_WITH_GROUND_TRUTH
    assert evaluator.evaluation_llm is not None
    assert evaluator.embeddings is not None

@pytest.mark.parametrize("query,response,contexts,expected_score", [
    ("How much protein should I consume daily?",
     "Consume 1.6-2.2g of protein per kg of bodyweight.",
     ["Consume 1.6-2.2g of protein per kg of bodyweight."],
     1.0)
])
def test_evaluate_faithfulness(evaluator, query, response, contexts, expected_score):
    """Test faithfulness evaluation."""
    score = evaluator.evaluate_faithfulness(query, response, contexts)
    assert isinstance(score, float)

# @pytest.mark.parametrize("query,response", [
#     ("How much protein should I consume daily?",
#      "Consume 1.6-2.2g of protein per kg of bodyweight.")
# ])
# def test_evaluate_answer_relevancy(evaluator, query, response):
#     """Test answer relevancy evaluation."""
#     score = evaluator.evaluate_answer_relevancy(query, response)
#     assert isinstance(score, float)
#
# @pytest.mark.parametrize("query,contexts", [
#     ("How much protein should I consume daily?",
#      ["Consume 1.6-2.2g of protein per kg of bodyweight."])
# ])
# def test_evaluate_context_relevancy(evaluator, query, contexts):
#     """Test context relevancy evaluation."""
#     score = evaluator.evaluate_context_relevancy(query, contexts)
#     assert isinstance(score, float)
#
# @pytest.mark.parametrize("query,contexts", [
#     ("How much protein should I consume daily?",
#      ["Consume 1.6-2.2g of protein per kg of bodyweight."])
# ])
# def test_evaluate_context_precision(evaluator, query, contexts):
#     """Test context precision evaluation."""
#     score = evaluator.evaluate_context_precision(query, contexts)
#     assert isinstance(score, float)
#
@pytest.mark.parametrize("query,response", [
    ("How much protein should I consume daily?",
     "Consume 1.6-2.2g of protein per kg of bodyweight.")
])
def test_evaluate_fitness_domain_accuracy(evaluator, query, response):
    """Test fitness domain accuracy evaluation."""
    score = evaluator.evaluate_fitness_domain_accuracy(query, response)
    assert isinstance(score, float)

@pytest.mark.parametrize("response", [
    ("Consume 1.6-2.2g of protein per kg of bodyweight.")
])
def test_evaluate_scientific_correctness(evaluator, response):
    """Test scientific correctness evaluation."""
    score = evaluator.evaluate_scientific_correctness(response)
    assert isinstance(score, float)

@pytest.mark.parametrize("query,response", [
    ("How much protein should I consume daily?",
     "Consume 1.6-2.2g of protein per kg of bodyweight.")
])
def test_evaluate_practical_applicability(evaluator, query, response):
    """Test practical applicability evaluation."""
    score = evaluator.evaluate_practical_applicability(query, response)
    assert isinstance(score, float)

@pytest.mark.parametrize("response", [
    ("Consume 1.6-2.2g of protein per kg of bodyweight.")
])
def test_evaluate_safety_consideration(evaluator, response):
    """Test safety consideration evaluation."""
    score = evaluator.evaluate_safety_consideration(response)
    assert isinstance(score, float)

# @pytest.mark.parametrize("query,contexts", [
#     ("How much protein should I consume daily?",
#      ["Consume 1.6-2.2g of protein per kg of bodyweight."])
# ])
# def test_evaluate_retrieval_precision(evaluator, query, contexts):
#     """Test retrieval precision evaluation."""
#     score = evaluator.evaluate_retrieval_precision(query, contexts)
#     assert isinstance(score, float)
#
# @pytest.mark.parametrize("query,contexts,ground_truth", [
#     ("How much protein should I consume daily?",
#      ["Consume 1.6-2.2g of protein per kg of bodyweight."],
#      "Consume 1.6-2.2g of protein per kg of bodyweight.")
# ])
# def test_evaluate_retrieval_recall(evaluator, query, contexts, ground_truth):
#     """Test retrieval recall evaluation."""
#     score = evaluator.evaluate_retrieval_recall(query, contexts, ground_truth)
#     assert isinstance(score, float)
#
@pytest.mark.parametrize("response,ground_truth", [
    ("Consume 1.6-2.2g of protein per kg of bodyweight.",
     "Consume 1.6-2.2g of protein per kg of bodyweight.")
])
def test_evaluate_answer_completeness(evaluator, response, ground_truth):
    """Test answer completeness evaluation."""
    score = evaluator.evaluate_answer_completeness(response, ground_truth)
    assert isinstance(score, float)

@pytest.mark.parametrize("response", [
    ("Consume 1.6-2.2g of protein per kg of bodyweight.")
])
def test_evaluate_answer_conciseness(evaluator, response):
    """Test answer conciseness evaluation."""
    score = evaluator.evaluate_answer_conciseness(response)
    assert isinstance(score, float)

@pytest.mark.parametrize("query,response", [
    ("How much protein should I consume daily?",
     "Consume 1.6-2.2g of protein per kg of bodyweight.")
])
def test_evaluate_answer_helpfulness(evaluator, query, response):
    """Test answer helpfulness evaluation."""
    score = evaluator.evaluate_answer_helpfulness(query, response)
    assert isinstance(score, float)