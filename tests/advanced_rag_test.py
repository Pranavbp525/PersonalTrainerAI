import pytest
from unittest.mock import patch, MagicMock
from src.rag_model.advanced_rag import AdvancedRAG  # adjust based on your file name


def test_expand_query():
    # Create the RAG instance
    rag = AdvancedRAG()

    # Mock the chain after instantiation
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {
        "text": "1. What are the best cardio exercises?\n2. Effective cardio routines?\n3. How to improve cardiovascular fitness?"
    }

    # Inject the mock
    rag.query_expansion_chain = mock_chain

    # Run the test
    result = rag.expand_query("Best cardio?")

    # Assertions
    assert "What are the best cardio exercises?" in result
    assert "Best cardio?" in result  # Original query should be preserved
    assert len(result) == 4


def test_retrieve_documents():
    # Create mock embedder and Pinecone index
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 768

    mock_index = MagicMock()
    mock_index.query.return_value = {
        "matches": [
            {"id": "doc1", "metadata": {"text": "Document 1"}, "score": 0.95},
            {"id": "doc2", "metadata": {"text": "Document 2"}, "score": 0.92},
            {"id": "doc1", "metadata": {"text": "Duplicate Document"}, "score": 0.90},  # duplicate ID, should be skipped
        ]
    }

    rag = AdvancedRAG()
    rag.embedding_model = mock_embedder
    rag.index = mock_index

    queries = ["Best exercises for arms"]
    results = rag.retrieve_documents(queries)

    assert len(results) == 2
    assert results[0]["text"] == "Document 1"
    assert results[1]["text"] == "Document 2"


def test_rerank_documents():
    # Mock the reranking_chain with a scoring logic
    mock_reranking_chain = MagicMock()
    mock_reranking_chain.invoke.side_effect = [
        {"text": "9"},  # Above threshold
        {"text": "5"}   # Below threshold (threshold is 0.7 -> 7.0 out of 10)
    ]

    # Instantiate and inject the mock
    rag = AdvancedRAG()
    rag.reranking_chain = mock_reranking_chain

    # Input documents
    documents = [
        {"text": "doc1", "score": 0.9, "id": "1"},
        {"text": "doc2", "score": 0.8, "id": "2"},
    ]

    # Call rerank
    reranked = rag.rerank_documents("How to stay fit?", documents)

    # Assertions
    assert len(reranked) == 1
    assert reranked[0]["text"] == "doc1"
    assert reranked[0]["relevance_score"] == 9


def test_answer_question():
    # Mock chains
    mock_query_expansion = MagicMock()
    mock_query_expansion.invoke.return_value = {
        "text": "1. What are good core workouts?\n2. Best exercises for abs?\n3. Core strengthening exercises?"
    }

    mock_retrieve = MagicMock()
    mock_retrieve.return_value = [
        {"text": "doc1", "score": 0.9, "id": "1"},
        {"text": "doc2", "score": 0.85, "id": "2"}
    ]

    mock_rerank = MagicMock()
    mock_rerank.return_value = [
        {"text": "doc1", "score": 0.9, "relevance_score": 9, "id": "1"}
    ]

    mock_answer_gen = MagicMock()
    mock_answer_gen.invoke.return_value = {
        "text": "You can try planks, leg raises, and crunches for core strength."
    }

    # Instantiate RAG and inject mocks
    rag = AdvancedRAG()
    rag.query_expansion_chain = mock_query_expansion
    rag.retrieve_documents = mock_retrieve
    rag.rerank_documents = mock_rerank
    rag.answer_generation_chain = mock_answer_gen

    # Call method
    answer = rag.answer_question("How to strengthen my core?")

    # Assertions
    assert isinstance(answer, str)
    assert "planks" in answer or "crunches" in answer