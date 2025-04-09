import pytest
from unittest.mock import patch, MagicMock
from src.rag_model.raptor_rag import RaptorRAG

@pytest.fixture
def mock_raptor_rag():
    with patch('src.rag_model.raptor_rag.Pinecone') as MockPinecone, \
         patch('src.rag_model.raptor_rag.HuggingFaceEmbeddings') as MockEmbeddings, \
         patch('src.rag_model.raptor_rag.ChatOpenAI') as MockChatOpenAI, \
         patch('src.rag_model.raptor_rag.LLMChain') as MockLLMChain:
        
        mock_pinecone = MockPinecone.return_value
        mock_embeddings = MockEmbeddings.return_value
        mock_llm = MockChatOpenAI.return_value
        mock_llm_chain = MockLLMChain.return_value
        
        raptor_rag = RaptorRAG()
        
        yield raptor_rag, mock_pinecone, mock_embeddings, mock_llm, mock_llm_chain

def test_initialize_raptor_rag(mock_raptor_rag):
    raptor_rag, mock_pinecone, mock_embeddings, mock_llm, mock_llm_chain = mock_raptor_rag
    
    assert raptor_rag.top_k == 8
    assert raptor_rag.reasoning_steps == 3
    assert raptor_rag.embedding_model == mock_embeddings
    assert raptor_rag.llm == mock_llm
    assert raptor_rag.retrieval_planning_chain == mock_llm_chain
    assert raptor_rag.reasoning_chain == mock_llm_chain
    assert raptor_rag.reflection_chain == mock_llm_chain
    assert raptor_rag.answer_synthesis_chain == mock_llm_chain

def test_plan_retrieval(mock_raptor_rag):
    raptor_rag, _, _, _, mock_llm_chain = mock_raptor_rag
    
    mock_llm_chain.invoke.return_value = {"text": '{"concepts": ["nutrition"], "information_needed": ["protein intake"], "search_queries": ["best protein sources"]}'}
    
    plan = raptor_rag.plan_retrieval("What are the best sources of protein?")
    assert "raw_plan" in plan

def test_retrieve_documents(mock_raptor_rag):
    raptor_rag, _, mock_embeddings, _, _ = mock_raptor_rag
    
    mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
    raptor_rag.index.query.return_value = MagicMock(matches=[
        MagicMock(id="doc1", score=0.95, metadata={
            "text": "Sample text",
        }),
        MagicMock(id="doc2", score=0.90, metadata={
            "text": "Another sample text",
        })
    ])
    
    documents = raptor_rag.retrieve_documents("What are the best sources of protein?", {"raw_plan": ''})
    assert len(documents) == 2
    assert documents[0]["id"] == "doc1"
    assert documents[1]["id"] == "doc2"

def test_perform_multi_step_reasoning(mock_raptor_rag):
    raptor_rag, _, _, _, mock_llm_chain = mock_raptor_rag
    
    mock_llm_chain.invoke.side_effect = [
        {"text": "Step 1 reasoning"},
        {"text": "Step 2 reasoning"},
        {"text": "Step 3 reasoning"},
        {"text": "Reflection on reasoning"}
    ]
    
    reasoning_result = raptor_rag.perform_multi_step_reasoning("What are the best sources of protein?", "Sample context")
    assert "reasoning_chain" in reasoning_result
    assert "reflection" in reasoning_result

def test_synthesize_answer(mock_raptor_rag):
    raptor_rag, _, _, _, mock_llm_chain = mock_raptor_rag
    
    mock_llm_chain.invoke.return_value = {"text": "Synthesized answer"}
    
    reasoning_result = {
        "reasoning_chain": "Step 1: reasoning\n\nStep 2: reasoning\n\nStep 3: reasoning",
        "reflection": "Reflection on reasoning"
    }
    
    answer = raptor_rag.synthesize_answer("What are the best sources of protein?", "Sample context", reasoning_result)
    assert "Synthesized answer" in answer

def test_answer_question(mock_raptor_rag):
    raptor_rag, _, _, _, mock_llm_chain = mock_raptor_rag
    
    mock_llm_chain.invoke.side_effect = [
        {"text": '{"concepts": ["nutrition"], "information_needed": ["protein intake"], "search_queries": ["best protein sources"]}'},
        {"text": "Retrieved document text"},
        {"text": "Step 1 reasoning"},
        {"text": "Step 2 reasoning"},
        {"text": "Step 3 reasoning"},
        {"text": "Reflection on reasoning"},
        {"text": "Synthesized answer"}
    ]
    
    answer = raptor_rag.answer_question("What are the best sources of protein?")
    assert "Synthesized answer" in answer
