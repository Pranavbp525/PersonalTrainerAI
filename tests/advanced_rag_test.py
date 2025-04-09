import pytest
from unittest.mock import patch, MagicMock
from src.rag_model.advanced_rag import AdvancedRAG

@pytest.fixture
def advanced_rag():
    with patch('src.rag_model.advanced_rag.HuggingFaceEmbeddings') as MockEmbeddings, \
         patch('src.rag_model.advanced_rag.Pinecone') as MockPinecone, \
         patch('src.rag_model.advanced_rag.ChatOpenAI') as MockChatOpenAI:
        MockEmbeddings.return_value = MagicMock()
        MockPinecone.return_value = MagicMock()
        MockPinecone.return_value.Index.return_value = MagicMock()
        MockChatOpenAI.return_value = MagicMock()
        return AdvancedRAG()

def test_initialization(advanced_rag):
    assert advanced_rag.embedding_model is not None
    assert advanced_rag.pc is not None
    assert advanced_rag.llm is not None

def test_expand_query(advanced_rag):
    mock_response = {"text": "1. expanded query 1\n2. expanded query 2\n3. expanded query 3"}
    advanced_rag.query_expansion_chain.invoke = MagicMock(return_value=mock_response)
    
    result = advanced_rag.expand_query("original query")
    
    assert len(result) == 4  # including the original query
    assert "expanded query 1" in result
    assert "expanded query 2" in result
    assert "expanded query 3" in result
    assert "original query" in result

def test_retrieve_documents(advanced_rag):
    mock_results = {
        "matches": [
            {"id": "1", "score": 0.9, "metadata": {"text": "document 1"}},
            {"id": "2", "score": 0.8, "metadata": {"text": "document 2"}}
        ]
    }
    advanced_rag.index.query = MagicMock(return_value=mock_results)
    advanced_rag.embedding_model.embed_query = MagicMock(return_value=[0.1, 0.2, 0.3])
    
    result = advanced_rag.retrieve_documents(["query 1", "query 2"])
    
    assert len(result) == 2
    assert result[0]["text"] == "document 1"
    assert result[1]["text"] == "document 2"

def test_rerank_documents(advanced_rag):
    mock_response = {"text": "8"}
    advanced_rag.reranking_chain.invoke = MagicMock(return_value=mock_response)
    
    documents = [
        {"text": "document 1", "score": 0.9, "id": "1"},
        {"text": "document 2", "score": 0.8, "id": "2"}
    ]
    result = advanced_rag.rerank_documents("query", documents)
    
    assert len(result) == 2
    assert result[0]["relevance_score"] == 8.0

def test_answer_question(advanced_rag):
    mock_expanded_queries = ["expanded query 1", "expanded query 2", "original query"]
    mock_documents = [
        {"text": "document 1", "score": 0.9, "relevance_score": 8.0, "id": "1"},
        {"text": "document 2", "score": 0.8, "relevance_score": 7.0, "id": "2"}
    ]
    mock_response = {"text": "This is the answer."}
    
    advanced_rag.expand_query = MagicMock(return_value=mock_expanded_queries)
    advanced_rag.retrieve_documents = MagicMock(return_value=mock_documents)
    advanced_rag.rerank_documents = MagicMock(return_value=mock_documents)
    advanced_rag.answer_generation_chain.invoke = MagicMock(return_value=mock_response)
    
    result = advanced_rag.answer_question("original question")
    
    assert result == "This is the answer."

if __name__ == "__main__":
    pytest.main()
