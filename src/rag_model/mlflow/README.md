# MLflow Integration for RAG Evaluation in PersonalTrainerAI

This documentation explains how to use MLflow to track experiments for RAG evaluation in the PersonalTrainerAI project.

## Overview

MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. In this integration, we use MLflow to track metrics, parameters, and artifacts for different RAG (Retrieval-Augmented Generation) implementations, allowing for systematic comparison and evaluation.

## Directory Structure

```
mlflow_setup/
├── mlflow_rag_metrics.py       # Defines metrics to track in MLflow
├── mlflow_rag_tracker.py       # Core MLflow tracking implementation
├── rag_evaluation_mlflow.py    # Script to run RAG evaluation with MLflow tracking
├── test_mlflow_integration.py  # Test script for MLflow integration
└── README.md                   # This documentation
```

## Installation

1. Install the required packages:

```bash
pip install mlflow pandas scikit-learn matplotlib
```

2. Verify the installation:

```bash
python -c "import mlflow; print(f'MLflow version: {mlflow.__version__}')"
```

## Quick Start

1. Start the MLflow server:

```bash
cd mlflow
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri file:///D:/PersonalTrainerAI/PersonalTrainerAI/src/rag_model/mlflow/mlruns
```

2. Run the test script to verify the integration:

```bash
python test_mlflow_integration.py
```

3. Access the MLflow UI at http://localhost:5000

## Integration with RAG Evaluation

### Step 1: Copy the MLflow Integration Files

Copy the following files to your project:

- `mlflow_rag_metrics.py`
- `mlflow_rag_tracker.py`
- `rag_evaluation_mlflow.py`

Place them in an appropriate directory in your project structure, such as `src/rag_model/mlflow/`.

### Step 2: Modify Your RAG Evaluation Code

Integrate MLflow tracking into your existing RAG evaluation code. Here's how to modify the `advanced_rag_evaluation.py` file:

```python
# Import the MLflow tracker
from src.rag_model.mlflow.mlflow_rag_tracker import MLflowRAGTracker

class AdvancedRAGEvaluator:
    def __init__(self, output_dir="results", test_queries=None):
        # Existing initialization code...
        
        # Initialize MLflow tracker
        self.mlflow_tracker = MLflowRAGTracker(
            experiment_name="rag_evaluation",
            tracking_uri="http://localhost:5000"
        )
    
    def evaluate_implementation(self, rag_instance):
        # Existing evaluation code...
        
        # Extract parameters from the RAG implementation
        parameters = {
            "embedding_model": getattr(rag_instance, "embedding_model", "unknown"),
            "llm_model": getattr(rag_instance, "llm_model", "unknown"),
            "chunk_size": getattr(rag_instance, "chunk_size", 0),
            "chunk_overlap": getattr(rag_instance, "chunk_overlap", 0),
            "retrieval_k": getattr(rag_instance, "retrieval_k", 0),
            "temperature": getattr(rag_instance, "temperature", 0.0)
        }
        
        # Log the results to MLflow
        self.mlflow_tracker.log_rag_evaluation_results(
            results=results,
            implementation_name=rag_instance.__class__.__name__,
            parameters=parameters
        )
        
        return results
```

### Step 3: Run RAG Evaluation with MLflow Tracking

Use the provided script to run RAG evaluation with MLflow tracking:

```bash
python src/rag_model/mlflow/rag_evaluation_mlflow.py --rag-implementations advanced modular raptor --output-dir results --start-mlflow-server
```

Or integrate it into your existing workflow:

```python
from src.rag_model.mlflow.rag_evaluation_mlflow import run_evaluation_with_mlflow
import argparse

# Parse arguments
args = argparse.Namespace()
args.rag_implementations = ["advanced", "modular", "raptor"]
args.output_dir = "results"
args.mlflow_tracking_uri = "http://localhost:5000"
args.mlflow_experiment_name = "rag_evaluation"
args.start_mlflow_server = True
args.test_queries = None

# Run evaluation with MLflow tracking
run_evaluation_with_mlflow(args)
```

## Tracked Metrics

The following metrics are tracked in MLflow:

### RAGAS Metrics
- faithfulness
- answer_relevancy
- context_relevancy
- context_precision

### Custom Fitness Domain Metrics
- fitness_domain_accuracy
- scientific_correctness
- practical_applicability
- safety_consideration

### Retrieval Metrics
- retrieval_precision
- retrieval_recall

### Human Evaluation Metrics
- answer_completeness
- answer_conciseness
- answer_helpfulness

## Tracked Parameters

The following parameters are tracked for each RAG implementation:

- rag_implementation: Type of RAG implementation (advanced, modular, raptor)
- embedding_model: Embedding model used
- llm_model: LLM model used
- chunk_size: Document chunk size
- chunk_overlap: Document chunk overlap
- retrieval_k: Number of documents retrieved
- temperature: LLM temperature

## Visualizing Results

MLflow automatically tracks and visualizes metrics, parameters, and artifacts. To compare different RAG implementations:

1. Access the MLflow UI at http://localhost:5000
2. Navigate to the "rag_evaluation" experiment
3. Select the runs you want to compare
4. Click "Compare" to see a side-by-side comparison of metrics and parameters

The `rag_evaluation_mlflow.py` script also generates comparison visualizations:

- Overall score comparison bar chart
- RAGAS metrics radar chart

These visualizations are saved to the output directory and logged as artifacts in MLflow.

## Advanced Usage

### Custom Metrics

To add custom metrics, modify the `mlflow_rag_metrics.py` file:

```python
# Add new metrics to the appropriate category
RAG_METRICS = {
    # Existing metrics...
    
    # Add new category
    "new_category": [
        "new_metric_1",
        "new_metric_2"
    ]
}

# Update the combined metrics
ALL_METRICS = (
    RAG_METRICS["ragas_metrics"] + 
    RAG_METRICS["custom_metrics"] + 
    RAG_METRICS["retrieval_metrics"] + 
    RAG_METRICS["human_eval_metrics"] +
    RAG_METRICS["new_category"]  # Add the new category
)
```

### Remote MLflow Server

To use a remote MLflow server:

```python
tracker = MLflowRAGTracker(
    experiment_name="rag_evaluation",
    tracking_uri="http://remote-server:5000"
)
```

### Batch Evaluation

To evaluate multiple RAG implementations in batch:

```python
from src.rag_model.mlflow.rag_evaluation_mlflow import run_evaluation_with_mlflow
import argparse

# Parse arguments
args = argparse.Namespace()
args.rag_implementations = ["advanced", "modular", "raptor"]
args.output_dir = "results"
args.mlflow_tracking_uri = "http://localhost:5000"
args.mlflow_experiment_name = "rag_evaluation"
args.start_mlflow_server = True
args.test_queries = "path/to/test_queries.json"

# Run evaluation with MLflow tracking
run_evaluation_with_mlflow(args)
```

## Troubleshooting

### MLflow Server Not Starting

If the MLflow server fails to start:

1. Check if another process is using port 5000:
   ```bash
   lsof -i :5000
   ```

2. Kill the process if necessary:
   ```bash
   kill -9 <PID>
   ```

3. Try a different port:
   ```python
   start_mlflow_server(port=5001)
   ```

### Connection Refused

If you see "Connection refused" errors:

1. Ensure the MLflow server is running
2. Check the tracking URI is correct
3. Wait a few seconds for the server to start before connecting

### Missing Dependencies

If you encounter missing dependencies:

```bash
pip install mlflow pandas scikit-learn matplotlib numpy
```

## Conclusion

This MLflow integration provides a robust framework for tracking and comparing different RAG implementations in the PersonalTrainerAI project. By systematically tracking metrics, parameters, and artifacts, you can make data-driven decisions about which RAG implementation performs best for your specific use case.
