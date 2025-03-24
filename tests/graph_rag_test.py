import pytest
from unittest.mock import patch, MagicMock
import networkx as nx
from src.rag_model.graph_rag import GraphRAG

@pytest.fixture
def mock_graph_rag():
    with patch('src.rag_model.graph_rag.Pinecone') as MockPinecone, \
         patch('src.rag_model.graph_rag.HuggingFaceEmbeddings') as MockEmbeddings, \
         patch('src.rag_model.graph_rag.OpenAI') as MockOpenAI, \
         patch('src.rag_model.graph_rag.LLMChain') as MockLLMChain:
        
        mock_pinecone = MockPinecone.return_value
        mock_embeddings = MockEmbeddings.return_value
        mock_openai = MockOpenAI.return_value
        mock_llm_chain = MockLLMChain.return_value
        
        graph_rag = GraphRAG()
        
        yield graph_rag, mock_pinecone, mock_embeddings, mock_openai, mock_llm_chain

def test_initialize_graph_rag(mock_graph_rag):
    graph_rag, mock_pinecone, mock_embeddings, mock_openai, mock_llm_chain = mock_graph_rag
    
    assert graph_rag.top_k == 5
    assert graph_rag.graph_path == "fitness_knowledge_graph.gpickle"
    assert graph_rag.embedding_model == mock_embeddings
    assert graph_rag.llm == mock_openai
    assert graph_rag.graph_builder_llm == mock_openai

def test_load_graph(mock_graph_rag):
    graph_rag, _, _, _, _ = mock_graph_rag
    
    with patch('src.rag_model.graph_rag.nx.read_gpickle', return_value=nx.DiGraph()) as mock_read_gpickle:
        graph = graph_rag._load_graph()
        mock_read_gpickle.assert_called_once_with("fitness_knowledge_graph.gpickle")
        assert isinstance(graph, nx.DiGraph)

def test_save_graph(mock_graph_rag):
    graph_rag, _, _, _, _ = mock_graph_rag
    
    with patch('src.rag_model.graph_rag.nx.write_gpickle') as mock_write_gpickle:
        graph_rag._save_graph()
        mock_write_gpickle.assert_called_once_with(graph_rag.graph, "fitness_knowledge_graph.gpickle")

def test_build_knowledge_graph(mock_graph_rag):
    graph_rag, mock_pinecone, mock_embeddings, mock_openai, mock_llm_chain = mock_graph_rag
    
    mock_pinecone.Index.return_value.query.return_value.matches = [
        MagicMock(id="doc1", metadata={"text": "Sample text", "source": "Source1"}),
        MagicMock(id="doc2", metadata={"text": "Another sample text", "source": "Source2"}),
    ]
    mock_llm_chain.run.side_effect = ["entity1, entity2", "relation1", "relation2"]
    
    with patch('src.rag_model.graph_rag.nx.write_gpickle') as mock_write_gpickle:
        graph_rag.build_knowledge_graph()
        mock_write_gpickle.assert_called_once()

def test_retrieve_documents(mock_graph_rag):
    graph_rag, mock_pinecone, mock_embeddings, _, _ = mock_graph_rag
    
    mock_pinecone.Index.return_value.query.return_value.matches = [
        MagicMock(id="doc1", score=0.95, metadata={"text": "Sample text", "source": "Source1"}),
        MagicMock(id="doc2", score=0.90, metadata={"text": "Another sample text", "source": "Source2"}),
    ]
    
    documents = graph_rag.retrieve_documents("sample query")
    assert len(documents) == 2
    assert documents[0]["id"] == "doc1"
    assert documents[1]["id"] == "doc2"

def test_extract_entities_from_query(mock_graph_rag):
    graph_rag, _, _, _, mock_llm_chain = mock_graph_rag
    
    mock_llm_chain.run.return_value = "entity1, entity2"
    
    entities = graph_rag.extract_entities_from_query("sample query")
    assert entities == ["entity1", "entity2"]

def test_get_graph_context(mock_graph_rag):
    graph_rag, _, _, _, _ = mock_graph_rag
    
    graph_rag.graph.add_node("entity1")
    graph_rag.graph.add_node("entity2")
    graph_rag.graph.add_edge("entity1", "entity2", relation="relation1")
    
    context = graph_rag.get_graph_context(["entity1"])
    assert "entity1 relation1 entity2" in context

def test_format_context(mock_graph_rag):
    graph_rag, _, _, _, _ = mock_graph_rag
    
    documents = [
        {"id": "doc1", "text": "Sample text", "source": "Source1"},
        {"id": "doc2", "text": "Another sample text", "source": "Source2"},
    ]
    
    context = graph_rag.format_context(documents)
    assert "Document 1 [Source: Source1]:\nSample text\n" in context
    assert "Document 2 [Source: Source2]:\nAnother sample text\n" in context

def test_answer_question(mock_graph_rag):
    graph_rag, _, _, _, mock_llm_chain = mock_graph_rag
    
    graph_rag.graph.add_node("entity1")
    graph_rag.graph.add_node("entity2")
    graph_rag.graph.add_edge("entity1", "entity2", relation="relation1")
    
    mock_llm_chain.run.side_effect = ["entity1, entity2", "entity1 relation1 entity2", "Generated answer"]
    
    answer = graph_rag.answer_question("sample question")
    assert "Generated answer" in answer
