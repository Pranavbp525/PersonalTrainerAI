import pytest
from unittest.mock import patch, MagicMock
from src.rag_model.compare_rag_implementations import compare_rag_implementations

@pytest.fixture
def mock_evaluator():
    with patch('src.rag_model.compare_rag_implementations.RAGEvaluator') as MockRAGEvaluator:
        mock_evaluator = MockRAGEvaluator.return_value
        mock_evaluator.rag_implementations = ['naive', 'advanced', 'modular', 'graph', 'raptor']
        mock_evaluator.evaluate_implementation.return_value = {
            "average_scores": {"overall": 9.0}
        }
        yield mock_evaluator

def test_compare_rag_implementations_default(mock_evaluator):
    result = compare_rag_implementations()

    assert "comparison" in result
    assert "best_implementation" in result
    assert "best_score" in result
    assert result["best_implementation"] == "naive"
    assert result["best_score"] == 9.0
    assert mock_evaluator.evaluate_implementation.call_count == 5

def test_compare_rag_implementations_with_queries(mock_evaluator):
    mock_evaluator.rag_implementations = ['naive', 'advanced']
    mock_evaluator.evaluate_implementation.return_value = {
        "average_scores": {"overall": 8.5}
    }

    result = compare_rag_implementations(test_queries_path="test_queries.json")

    assert "comparison" in result
    assert "best_implementation" in result
    assert "best_score" in result
    assert result["best_implementation"] == "naive"
    assert result["best_score"] == 8.5
    assert mock_evaluator.evaluate_implementation.call_count == 2

def test_compare_rag_implementations_generate_queries(mock_evaluator):
    mock_evaluator.rag_implementations = ['naive']
    mock_evaluator.evaluate_implementation.return_value = {
        "average_scores": {"overall": 7.0}
    }

    result = compare_rag_implementations(generate_queries=10)

    assert "comparison" in result
    assert "best_implementation" in result
    assert "best_score" in result
    assert result["best_implementation"] == "naive"
    assert result["best_score"] == 7.0
    mock_evaluator.generate_test_queries.assert_called_once_with(10)
    mock_evaluator.save_test_queries.assert_called_once()

def test_compare_rag_implementations_build_graph(mock_evaluator):
    mock_evaluator.rag_implementations = ['graph']
    mock_evaluator.evaluate_implementation.return_value = {
        "average_scores": {"overall": 8.0}
    }

    result = compare_rag_implementations(build_graph=True)

    assert "comparison" in result
    assert "best_implementation" in result
    assert "best_score" in result
    assert result["best_implementation"] == "graph"
    assert result["best_score"] == 8.0
    assert mock_evaluator.graph_path is None

def test_generate_comparison_report(mock_evaluator):
    mock_evaluator.rag_implementations = ['naive', 'advanced']
    mock_evaluator.evaluate_implementation.return_value = {
        "average_scores": {"overall": 8.5},
        "average_scores_by_complexity": {"low": 9.0, "medium": 8.0, "high": 7.5},
        "average_scores_by_category": {
            "exercise_technique": 8.0,
            "workout_planning": 8.5,
            "nutrition_advice": 8.5,
            "progress_tracking": 8.0,
            "injury_prevention": 8.0,
            "exercise_science": 8.5,
            "complex_planning": 8.0
        },
        "average_response_time": 1.5
    }

    result = compare_rag_implementations()

    assert "comparison" in result
    assert "best_implementation" in result
    assert "best_score" in result
    assert result["best_implementation"] == "naive"
    assert result["best_score"] == 8.5
    assert mock_evaluator.evaluate_implementation.call_count == 2
    mock_evaluator.save_evaluation_results.assert_called_once()
    mock_evaluator.generate_comparison_charts.assert_called_once()
