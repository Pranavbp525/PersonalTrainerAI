import pytest
from unittest.mock import patch, MagicMock
from src.rag_model.modular_rag import ModularRAG

@pytest.fixture
def mock_modular_rag():
    with patch('src.rag_model.modular_rag.Pinecone') as MockPinecone, \
         patch('src.rag_model.modular_rag.HuggingFaceEmbeddings') as MockEmbeddings, \
         patch('src.rag_model.modular_rag.ChatOpenAI') as MockChatOpenAI, \
         patch('src.rag_model.modular_rag.LLMChain') as MockLLMChain:
        
        mock_pinecone = MockPinecone.return_value
        mock_embeddings = MockEmbeddings.return_value
        mock_llm = MockChatOpenAI.return_value
        mock_llm_chain = MockLLMChain.return_value
        
        modular_rag = ModularRAG()
        
        yield modular_rag, mock_pinecone, mock_embeddings, mock_llm, mock_llm_chain

def test_initialize_modular_rag(mock_modular_rag):
    modular_rag, mock_pinecone, mock_embeddings, mock_llm, mock_llm_chain = mock_modular_rag
    
    assert modular_rag.top_k == 5
    assert modular_rag.embedding_model == mock_embeddings
    assert modular_rag.llm == mock_llm
    assert modular_rag.query_classification_chain == mock_llm_chain
    assert modular_rag.specialized_query_chain == mock_llm_chain
    assert modular_rag.answer_generation_chain == mock_llm_chain

def test_classify_query(mock_modular_rag):
    modular_rag, _, _, _, mock_llm_chain = mock_modular_rag
    
    mock_llm_chain.invoke.return_value = {"text": "nutrition_diet"}
    
    category = modular_rag.classify_query("What should I eat after a workout?")
    assert category == "nutrition_diet"

def test_specialize_query(mock_modular_rag):
    modular_rag, _, _, _, mock_llm_chain = mock_modular_rag
    
    mock_llm_chain.invoke.return_value = {"text": "What are the best foods to eat after a workout for muscle recovery?"}
    
    specialized_query = modular_rag.specialize_query("What should I eat after a workout?", "nutrition_diet")
    assert specialized_query == "What are the best foods to eat after a workout for muscle recovery?"

def test_retrieve_documents(mock_modular_rag):
    modular_rag, _, mock_embeddings, _, _ = mock_modular_rag
    
    mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
    modular_rag.index.query.return_value = {
        "matches": [
            {"id": "doc1", "score": 0.95, "metadata": {"text": "Sample text", "category": "nutrition_diet"}},
            {"id": "doc2", "score": 0.90, "metadata": {"text": "Another sample text", "category": "nutrition_diet"}}
        ]
    }
    
    documents = modular_rag.retrieve_documents("What are the best foods to eat after a workout for muscle recovery?", "nutrition_diet")
    assert len(documents) == 2
    assert documents[0]["id"] == "doc1"
    assert documents[1]["id"] == "doc2"

def test_answer_question(mock_modular_rag):
    modular_rag, _, _, _, mock_llm_chain = mock_modular_rag
    
    mock_llm_chain.invoke.side_effect = [
        {"text": "nutrition_diet"},
        {"text": "What are the best foods to eat after a workout for muscle recovery?"},
        {"text": "The best foods to eat after a workout are high in protein and carbohydrates, such as chicken, rice, and vegetables."}
    ]
    
    answer = modular_rag.answer_question("What should I eat after a workout?")
    assert "The best foods to eat after a workout are high in protein and carbohydrates" in answer
