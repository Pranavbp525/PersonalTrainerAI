import pytest
from unittest.mock import patch, MagicMock
from src.rag_model.simple_rag_evaluation import SimpleRAGEvaluator, TEST_QUERIES

@pytest.fixture
def mock_simple_rag_evaluator():
    with patch('src.rag_model.simple_rag_evaluation.ChatOpenAI') as MockChatOpenAI, \
         patch('src.rag_model.simple_rag_evaluation.AdvancedRAG') as MockAdvancedRAG, \
         patch('src.rag_model.simple_rag_evaluation.ModularRAG') as MockModularRAG, \
         patch('src.rag_model.simple_rag_evaluation.RaptorRAG') as MockRaptorRAG:
        
        mock_llm = MockChatOpenAI.return_value
        mock_advanced_rag = MockAdvancedRAG.return_value
        mock_modular_rag = MockModularRAG.return_value
        mock_raptor_rag = MockRaptorRAG.return_value
        
        evaluator = SimpleRAGEvaluator()
        evaluator.rag_implementations = {
            "advanced": mock_advanced_rag,
            "modular": mock_modular_rag,
            "raptor": mock_raptor_rag
        }
        
        yield evaluator, mock_llm, mock_advanced_rag, mock_modular_rag, mock_raptor_rag

def test_initialize_simple_rag_evaluator(mock_simple_rag_evaluator):
    evaluator, mock_llm, mock_advanced_rag, mock_modular_rag, mock_raptor_rag = mock_simple_rag_evaluator
    
    assert evaluator.output_dir == "results"
    assert evaluator.test_queries == TEST_QUERIES
    assert evaluator.evaluation_llm == mock_llm
    assert evaluator.rag_implementations["advanced"] == mock_advanced_rag
    assert evaluator.rag_implementations["modular"] == mock_modular_rag
    assert evaluator.rag_implementations["raptor"] == mock_raptor_rag

def test_evaluate_response(mock_simple_rag_evaluator):
    evaluator, mock_llm, _, _, _ = mock_simple_rag_evaluator
    
    mock_llm.invoke.return_value.content = json.dumps({
        "relevance": 9,
        "factual_accuracy": 8,
        "completeness": 7,
        "hallucination": 10,
        "relationship_awareness": 6,
        "reasoning_quality": 8
    })
    
    scores = evaluator.evaluate_response("What is the best exercise for core strength?", "Plank is the best exercise.")
    assert scores["relevance"] == 9
    assert scores["factual_accuracy"] == 8
    assert scores["completeness"] == 7
    assert scores["hallucination"] == 10
    assert scores["relationship_awareness"] == 6
    assert scores["reasoning_quality"] == 8
    assert "overall" in scores

def test_evaluate_implementation(mock_simple_rag_evaluator):
    evaluator, _, mock_advanced_rag, _, _ = mock_simple_rag_evaluator
    
    mock_advanced_rag.answer_question.return_value = "Plank is the best exercise."
    
    with patch.object(evaluator, 'evaluate_response', return_value={
        "relevance": 9,
        "factual_accuracy": 8,
        "completeness": 7,
        "hallucination": 10,
        "relationship_awareness": 6,
        "reasoning_quality": 8,
        "overall": 8.0
    }):
        results = evaluator.evaluate_implementation("advanced")
        assert results["implementation"] == "advanced"
        assert "results" in results
        assert "average_scores" in results
        assert "average_response_time" in results
        assert results["average_scores"]["overall"] == 8.0

def test_compare_implementations(mock_simple_rag_evaluator):
    evaluator, _, mock_advanced_rag, mock_modular_rag, mock_raptor_rag = mock_simple_rag_evaluator
    
    mock_advanced_rag.answer_question.return_value = "Advanced RAG response."
    mock_modular_rag.answer_question.return_value = "Modular RAG response."
    mock_raptor_rag.answer_question.return_value = "RAPTOR RAG response."
    
    with patch.object(evaluator, 'evaluate_response', return_value={
        "relevance": 9,
        "factual_accuracy": 8,
        "completeness": 7,
        "hallucination": 10,
        "relationship_awareness": 6,
        "reasoning_quality": 8,
        "overall": 8.0
    }):
        results = evaluator.compare_implementations()
        assert "comparison" in results
        assert "best_implementation" in results
        assert "best_score" in results
        assert results["best_implementation"] == "advanced"
        assert results["best_score"] == 8.0

def test_save_results(mock_simple_rag_evaluator, tmpdir):
    evaluator, _, _, _, _ = mock_simple_rag_evaluator
    evaluator.output_dir = tmpdir
    
    results = {
        "comparison": {
            "advanced": {
                "average_scores": {
                    "overall": 8.0
                }
            }
        },
        "best_implementation": "advanced",
        "best_score": 8.0
    }
    
    evaluator.save_results(results)
    output_file = os.path.join(tmpdir, "evaluation_results.json")
    with open(output_file, "r") as f:
        saved_results = json.load(f)
        
    assert saved_results == results

def test_generate_comparison_charts(mock_simple_rag_evaluator, tmpdir):
    evaluator, _, _, _, _ = mock_simple_rag_evaluator
    evaluator.output_dir = tmpdir
    
    results = {
        "advanced": {
            "average_scores": {
                "relevance": 9,
                "factual_accuracy": 8,
                "completeness": 7,
                "hallucination": 10,
                "relationship_awareness": 6,
                "reasoning_quality": 8,
                "overall": 8.0
            },
            "average_response_time": 1.2
        },
        "modular": {
            "average_scores": {
                "relevance": 9,
                "factual_accuracy": 8,
                "completeness": 7,
                "hallucination": 10,
                "relationship_awareness": 6,
                "reasoning_quality": 8,
                "overall": 8.0
            },
            "average_response_time": 1.5
        }
    }
    
    evaluator.generate_comparison_charts(results)
    assert os.path.exists(os.path.join(tmpdir, "metrics_comparison.png"))
    assert os.path.exists(os.path.join(tmpdir, "response_time_comparison.png"))

def test_print_summary(mock_simple_rag_evaluator, capsys):
    evaluator, _, _, _, _ = mock_simple_rag_evaluator
    
    results = {
        "comparison": {
            "advanced": {
                "average_scores": {
                    "relevance": 9,
                    "factual_accuracy": 8,
                    "completeness": 7,
                    "hallucination": 10,
                    "relationship_awareness": 6,
                    "reasoning_quality": 8,
                    "overall": 8.0
                },
                "average_response_time": 1.2
            }
        },
        "best_implementation": "advanced",
        "best_score": 8.0
    }
    
    evaluator.print_summary(results)
    captured = capsys.readouterr()
    assert "Best implementation: ADVANCED" in captured.out
    assert "Best overall score: 8.00/10.0" in captured.out
    assert "Relevance: 9.00/10.0" in captured.out
