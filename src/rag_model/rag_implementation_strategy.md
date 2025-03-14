# RAG Implementation Strategy for PersonalTrainerAI

## Overview

This document outlines the strategy for implementing and evaluating multiple Retrieval-Augmented Generation (RAG) approaches for the PersonalTrainerAI project. The goal is to create a system that can effectively retrieve relevant fitness information from our knowledge base and generate personalized workout routines and advice.

## Current Implementation Status

- Data pipeline for scraping fitness content from Renaissance Periodization and other sources is complete
- Data preprocessing and chunking is implemented
- Vector embeddings are generated using sentence-transformers/all-mpnet-base-v2
- Pinecone vector database is set up for storing and retrieving embeddings
- Airflow orchestration is in place for the data pipeline

## RAG Architecture Options

Based on our research, we'll implement and evaluate three RAG architectures:

### 1. Naive RAG (Baseline)

- **Description**: The simplest RAG implementation that directly retrieves a fixed number of documents based on query similarity and passes them to the LLM.
- **Components**:
  - Query embedding using the same model as document embeddings
  - Vector similarity search in Pinecone
  - Fixed context window with top-k retrieved documents
  - Direct passage to LLM with minimal prompt engineering

### 2. Advanced RAG

- **Description**: Enhanced RAG implementation with improved retrieval and context processing.
- **Components**:
  - Query expansion using LLM to generate multiple related queries
  - Sentence-window retrieval for better context
  - Re-ranking of retrieved documents using relevance scoring
  - Dynamic context window based on relevance scores
  - Structured prompt engineering for better LLM utilization

### 3. Modular RAG

- **Description**: A flexible, component-based RAG system that can be optimized for different query types.
- **Components**:
  - Query classification to determine query intent (workout planning, exercise technique, nutrition advice, etc.)
  - Specialized retrievers for different query types
  - Hypothetical Document Embedding (HyDE) for complex queries
  - Multi-stage retrieval with feedback loops
  - LLM-based reranking for final document selection
  - Template-based response generation with structured output formats

## Implementation Plan

### Phase 1: Core RAG Components

1. **Query Processing**
   - Implement query embedding using the existing embedding model
   - Create query expansion module using LLM
   - Develop query classification for different fitness-related intents

2. **Retrieval Mechanisms**
   - Implement basic vector similarity search (already available with Pinecone)
   - Develop sentence-window retrieval for better context
   - Create HyDE implementation for complex queries
   - Build multi-query retrieval system

3. **Reranking**
   - Implement relevance scoring based on semantic similarity
   - Create LLM-based reranking module
   - Develop Maximum Marginal Relevance (MMR) for diversity in results

4. **Response Generation**
   - Design prompt templates for different query types
   - Implement structured output formatting for workout plans
   - Create citation and reference tracking

### Phase 2: Architecture Assembly

1. **Naive RAG**
   - Assemble basic components with minimal complexity
   - Focus on direct retrieval and generation

2. **Advanced RAG**
   - Integrate query expansion and reranking
   - Implement sentence-window retrieval
   - Add structured prompt engineering

3. **Modular RAG**
   - Implement query classification and routing
   - Create specialized retrievers for different query types
   - Integrate multi-stage retrieval with feedback loops
   - Develop template-based response generation

## Evaluation Framework

We'll evaluate each RAG architecture using the following metrics:

### Retrieval Metrics

1. **Precision@k**: Percentage of relevant documents among top-k retrieved
2. **Recall@k**: Percentage of all relevant documents that are retrieved in top-k
3. **Mean Reciprocal Rank (MRR)**: Average position of the first relevant document
4. **Normalized Discounted Cumulative Gain (nDCG)**: Measures ranking quality with relevance scores

### Generation Metrics

1. **Answer Relevance**: How well the generated answer addresses the query
2. **Factual Accuracy**: Correctness of information in the generated response
3. **Contextual Precision**: How well the response uses the retrieved information
4. **Hallucination Rate**: Frequency of generated content not supported by retrieved documents

### Fitness Domain-Specific Metrics

1. **Exercise Accuracy**: Correctness of exercise recommendations
2. **Workout Completeness**: Whether the workout plan covers all necessary components
3. **Personalization Quality**: How well the response adapts to user-specific information
4. **Scientific Backing**: Whether recommendations are supported by scientific evidence

## Technology Stack

1. **LLM Options**:
   - OpenAI models (GPT-3.5-turbo, GPT-4)
   - Open-source models (Llama 3, Mistral)
   - Specialized fitness-tuned models (if available)

2. **Vector Database**:
   - Continue using Pinecone (already implemented)

3. **Embedding Models**:
   - Continue using sentence-transformers/all-mpnet-base-v2 (already implemented)
   - Evaluate domain-specific embeddings for fitness content

4. **Framework**:
   - LangChain for RAG pipeline components
   - Evaluation frameworks like RAGAS or TruLens

## Next Steps

1. Implement the core RAG components
2. Assemble the three RAG architectures
3. Develop the evaluation framework
4. Test and compare the RAG implementations
5. Select the best approach or create a hybrid solution
6. Integrate with the existing data pipeline
7. Document the implementation and results
