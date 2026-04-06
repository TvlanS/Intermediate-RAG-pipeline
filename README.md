# Intermediate-RAG-pipeline

A sophisticated Retrieval-Augmented Generation (RAG) pipeline featuring hybrid retrieval (BM25 + vector embeddings), query augmentation, and parent‑child chunking for improved context selection.

## Overview

```mermaid
flowchart TD
    A[PDF Documents] --> B[PyPDFLoader]
    B --> C[Parent‑Child Splitting]

    subgraph "Indexing"
        C --> D[Parent Chunks]
        C --> E[Child Chunks]
        E --> F[Vector Embeddings]
        F --> G[ChromaDB]
        E --> H[BM25 Index]
    end

    I[User Query] --> J[Query Augmentation<br/>DeepSeek LLM]
    J --> K[Alternative Query]

    I --> L{Ensemble Retriever}
    K --> L
    G -.-> L
    H -.-> L

    L --> M[Retrieve Child Chunks]
    M --> N[Ranking & Parent Selection]
    D -.-> N
    N --> O[Top‑k Parent Chunks]
    O --> P[Context Assembly]
    P --> Q[LLM Generation<br/>(planned)]
```

## Key Features

- **Hybrid Retrieval**: Combines BM25 (lexical) and vector similarity for better recall
- **Query Augmentation**: LLM-generated alternative queries increase retrieval robustness
- **Parent‑Child Chunking**: Fine-grained child chunks for retrieval, parent chunks for context coherence
- **Duplicate‑Aware Ranking**: Prioritizes parent chunks with multiple relevant child chunks
- **Persistent Vector Storage**: ChromaDB avoids re‑embedding on each run

## Quick Start

1. Install dependencies:
```bash
pip install langchain langchain-community chromadb sentence-transformers pypdf pyprojroot openai yaml
```

2. Configure API key in `Config/app_config.yml`

3. Run the pipeline:
```python
from utils.vector_embedding_advance import VectorLoader
from utils.LLM_load import llm
from utils.rankingV2 import Ranking
import json

# Load and index documents
app = VectorLoader(["document.pdf"])
result = app.embedding_documents()

# Augment query and retrieve
query = "What is machine learning?"
alternative = llm(query).llm_call()
alt_query = json.loads(alternative)["alternative_query"]

# Get ranked context
context = Ranking(query, alt_query, result["retriever"], result["parent"], k=10).quering()
```

## Documentation

- **[Full Pipeline Details](RAG_PIPELINE.md)** – Complete architectural breakdown and component descriptions
- **Project Structure**:
  - `utils/vector_embedding_advance.py` – Document loading, chunking, embedding, retrieval setup
  - `utils/rankingV2.py` – Query‑time ranking and parent selection
  - `utils/LLM_load.py` – DeepSeek LLM wrapper
  - `utils/config_setup.py` – Configuration management
  - `Config/app_config.yml` – API key and system prompt

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 3000 | Parent chunk token size |
| `chunk_overlap` | 500 | Parent chunk overlap |
| `embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `k_number` | 20 | Number of chunks retrieved by each retriever |
| `embed_weight` / `bm25_weight` | 0.5 | Ensemble retriever weights |

## Next Steps

The pipeline currently stops at context assembly. Future work includes:
1. Implementing final LLM answer generation
2. Adding cross‑encoder re‑ranking
3. Supporting additional document formats
4. Production deployment optimizations

---
*Part of an intermediate RAG implementation for educational and research purposes.* 
