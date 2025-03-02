import pytest
import sys
sys.path.append('./src')
from data_pipeline.vector_db import query_pinecone
from unittest.mock import patch, MagicMock

@patch("scripts.vectdb_pc.Pinecone")
@patch("scripts.vectdb_pc.HuggingFaceEmbeddings")
def test_query_pinecone(mock_embeddings, mock_pinecone):
    """Test querying Pinecone for relevant chunks."""
    mock_model = MagicMock()
    mock_model.embed_query.return_value = [0.1, 0.2, 0.3]  # Mock embedding vector

    mock_embeddings.return_value = mock_model
    mock_pinecone.return_value.Index.return_value.query.return_value = {
        "matches": [{"metadata": {"text": "Relevant text chunk"}}]
    }

    query_pinecone("How to do push-ups?")
    mock_pinecone.return_value.Index.return_value.query.assert_called_once()
