import pytest
from unittest.mock import patch, MagicMock
from src.rag_model.naive_rag import NaiveRAG

@pytest.fixture
def mock_naive_rag():
    with patch('src.rag_model.naive_rag.Pinecone') as MockPinecone, \
         patch('src.rag_model.naive_rag.HuggingFaceEmbeddings') as MockEmbeddings, \
         patch('src.rag_model.naive_rag.ChatOpenAI') as MockChatOpenAI, \
         patch('src.rag_model.naive_rag.LLMChain') as MockLLMChain:
        
        mock_pinecone = MockPinecone.return_value
        mock_embeddings = MockEmbeddings.return_value
        mock_llm = MockChatOpenAI.return_value
        mock_llm_chain = MockLLMChain.return_value
        
        naive_rag = NaiveRAG()
        
        yield naive_rag, mock_pinecone, mock_embeddings, mock_llm, mock_llm_chain

def test_initialize_naive_rag(mock_naive_rag):
    naive_rag, mock_pinecone, mock_embeddings, mock_llm, mock_llm_chain = mock_naive_rag
    
    assert naive_rag.top_k == 5
    assert naive_rag.embedding_model == mock_embeddings
    assert naive_rag.llm == mock_llm
    assert naive_rag.llm_chain == mock_llm_chain

def test_retrieve_documents(mock_naive_rag):
    naive_rag, _, mock_embeddings, _, _ = mock_naive_rag
    
    mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
    naive_rag.index.query.return_value = MagicMock(matches=[
        MagicMock(id="doc1", score=0.95, metadata={
            "text": "Sample text",
            "source": "Source1"
        }),
        MagicMock(id="doc2", score=0.90, metadata={
            "text": "Another sample text",
            "source": "Source2"
        })
    ])
    
    documents = naive_rag.retrieve_documents("What is a good workout routine for beginners?")
    assert len(documents) == 2
    assert documents[0]["id"] == "doc1"
    assert documents[1]["id"] == "doc2"

def test_format_context(mock_naive_rag):
    naive_rag, _, _, _, _ = mock_naive_rag
    
    documents = [
        {"id": "doc1", "text": "Sample text", "source": "Source1"},
        {"id": "doc2", "text": "Another sample text", "source": "Source2"}
    ]
    
    context = naive_rag.format_context(documents)
    assert "Document 1 [Source: Source1]:\nSample text\n" in context
    assert "Document 2 [Source: Source2]:\nAnother sample text\n" in context

def test_answer_question(mock_naive_rag):
    naive_rag, _, _, _, mock_llm_chain = mock_naive_rag
    
    mock_llm_chain.run.return_value = "Generated answer"
    
    answer = naive_rag.answer_question("What is a good workout routine for beginners?")
    assert "Generated answer" in answer

def test_answer_question_no_documents(mock_naive_rag):
    naive_rag, _, _, _, _ = mock_naive_rag
    
    with patch.object(naive_rag, 'retrieve_documents', return_value=[]):
        answer = naive_rag.answer_question("What is a good workout routine for beginners?")
        assert answer == "I couldn't find any relevant information to answer your question."
