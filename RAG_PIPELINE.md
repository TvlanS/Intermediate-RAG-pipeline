# RAG Pipeline - Intermediate-RAG-pipeline

This document describes the complete Retrieval-Augmented Generation (RAG) pipeline implemented in this project. The pipeline features hybrid retrieval (lexical + vector), query augmentation, and parent‑child chunking for improved context selection.

## Text‑Based Flowchart

```
PDF Documents
     │
     ▼
PyPDFLoader
     │
     ▼
Parent‑Child Splitting
     ├─────────────────┐
     │                 │
     ▼                 ▼
Parent Chunks    Child Chunks
(3000 tokens)    (300 tokens)
     │                 │
     │                 ├──────────────┐
     │                 ▼              ▼
     │         HuggingFace      BM25 Index
     │         Embeddings           │
     │              │               │
     │              ▼               │
     │         ChromaDB◄────────────┘
     │         (Vector DB)           │
     │              │               │
     └──────────────┼───────────────┘
                    │
User Query          │   Ensemble Retriever
     │              │   (BM25 + Vector)
     ├──────────────┼──────────────┐
     │              │              │
     ▼              │              │
Query Augmentation  │              │
(DeepSeek LLM)      │              │
     │              │              │
     ▼              │              │
Alternative Query   │              │
     │              │              │
     └──────────────┼──────────────┘
                    │
                    ▼
           Retrieve Child Chunks
                    │
                    ▼
           Ranking & Parent Selection
                    │
                    ▼
           Top‑k Parent Chunks
                    │
                    ▼
           Context Assembly
                    │
                    ▼
           LLM Generation (planned)
                    │
                    ▼
           Final Answer
```

## Mermaid Diagram

```mermaid
flowchart TD
    %% Data Ingestion & Indexing Phase
    A[PDF Documents] --> B[PyPDFLoader]
    B --> C[Parent‑Child Splitting]

    subgraph "Chunking & Embedding"
        C --> D[Parent Chunks<br/>chunk_size=3000]
        C --> E[Child Chunks<br/>chunk_size=300]
        E --> F[HuggingFace Embeddings<br/>all‑MiniLM‑L6‑v2]
        F --> G[Vector Database<br/>ChromaDB]
        E --> H[BM25 Index]
    end

    %% Query‑Time Phase
    I[User Query] --> J[Query Augmentation<br/>DeepSeek LLM]
    J --> K[Alternative Query]

    I --> L{Ensemble Retriever}
    K --> L

    L --> M[Retrieve Child Chunks<br/>BM25 + Vector]

    M --> N[Ranking & Parent Selection]
    N --> O[Top‑k Parent Chunks]
    O --> P[Context Assembly<br/>with source metadata]
    P --> Q[LLM Generation<br/>(to be implemented)]
    Q --> R[Final Answer]

    %% Stored Data
    G -.-> L
    H -.-> L
    D -.-> O
```

## Detailed Pipeline Steps

### 1. Document Ingestion & Preprocessing

**Component**: `VectorLoader.load_pdf()`
- **Input**: Directory path or list of PDF files
- **Process**: Uses `PyPDFLoader` from LangChain to load PDF documents
- **Output**: List of `Document` objects with page content and metadata

### 2. Two‑Level Chunking

**Component**: `VectorLoader._chunk_documents()`
- **Parent Chunks**: Large chunks (default 3000 tokens) for retaining broader context
- **Child Chunks**: Smaller chunks (default 300 tokens) for precise retrieval
- **Process**:
  1. Split documents into parent chunks using `RecursiveCharacterTextSplitter`
  2. Add parent ID (hash of first 100 chars) to metadata
  3. Prepend source header: `[SOURCE: {title}| PAGE:{page}]`
  4. Further split each parent into child chunks, inheriting the parent ID
- **Output**: Two lists – parent chunks and child chunks

### 3. Vector Embedding & Storage

**Component**: `VectorLoader.embedding_documents()`
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` via `HuggingFaceEmbeddings`
- **Vector Database**: ChromaDB with persistent storage in `VectorDB/{dataset_name}/`
- **Process**:
  1. Generate embeddings for all child chunks
  2. Store in ChromaDB with automatic persistence
  3. If database already exists, load it instead of re‑embedding
- **Output**: Vector store retriever configured with `search_kwargs={"k":20}`

### 4. Lexical Indexing

**Component**: `VectorLoader.get_bm25_retriever()`
- **Index**: BM25 (Okapi BM25) built on child chunks
- **Configuration**: Same `k=20` as vector retriever
- **Output**: BM25 retriever instance

### 5. Hybrid Retrieval Setup

**Component**: `VectorLoader.embedding_documents()` (continued)
- **Ensemble Retriever**: Combines BM25 and vector retrievers with equal weights (0.5 each)
- **Configuration**: Both retrievers return top‑20 child chunks
- **Output**: Single `EnsembleRetriever` that merges results from both retrieval methods

### 6. Query Augmentation

**Component**: `llm.llm_call()` with specialized system prompt
- **LLM Provider**: DeepSeek (via OpenAI‑compatible API)
- **System Prompt**: Instructs model to generate an alternative query in JSON format
- **Process**:
  1. User query sent to DeepSeek with system prompt
  2. Model returns `{"alternative_query": "..."}` JSON
  3. Original query and alternative query are both used for retrieval
- **Purpose**: Increases retrieval recall by searching with multiple query formulations

### 7. Hybrid Retrieval Execution

**Component**: `Ranking.quering()`
- **Input**: Original query and alternative query
- **Process**:
  1. Both queries sent to the ensemble retriever
  2. For each query, retrieve top‑20 child chunks (BM25 + vector)
  3. Combine results from both queries
- **Output**: Combined list of relevant child chunks

### 8. Parent‑Level Ranking & Selection

**Component**: `Ranking.parent_id_select()` and ranking logic
- **Process**:
  1. Extract parent IDs from all retrieved child chunks
  2. Identify duplicate parent IDs (appearing for both queries)
  3. Select the parent with most duplicate appearances as "top parent"
  4. Take top‑k unique parent IDs (default k=5)
  5. Ensure top parent is included if not already in top‑k
- **Rationale**: Parent chunks provide broader context; selecting parents with multiple relevant child chunks increases confidence

### 9. Context Assembly

**Component**: `Ranking.quering()` (final stage)
- **Process**:
  1. Filter parent chunks by selected parent IDs
  2. Format context with explicit source headers already added during chunking
  3. Combine with original query in a prompt template
- **Output**: Formatted string ready for LLM generation

### 10. LLM Generation (Planned)

**Current Status**: The pipeline stops at context assembly. The formatted string includes the query and context but is not sent to an LLM for final answer generation.

**Planned Implementation**: Would involve:
1. Sending assembled context to DeepSeek or another LLM
2. Generating a coherent answer with citations
3. Returning final response to user

## File Structure & Classes

| Component | File | Class | Key Methods |
|-----------|------|-------|-------------|
| Document Loading & Embedding | `utils/vector_embedding_advance.py` | `VectorLoader` | `load_pdf()`, `_chunk_documents()`, `embedding_documents()` |
| Ranking & Context Selection | `utils/rankingV2.py` | `Ranking` | `parent_id_select()`, `quering()` |
| LLM Integration | `utils/LLM_load.py` | `llm` | `llm_call()` |
| Configuration | `utils/config_setup.py` | `Config` | Reads `Config/app_config.yml` |
| Example Usage | `utils/testingV2.ipynb` | – | End‑to‑end demonstration |

## Configuration

Key parameters in `VectorLoader` initialization:
- `chunk_size`: Parent chunk size (default: 3000)
- `chunk_overlap`: Parent chunk overlap (default: 500)
- `embedding_model`: HuggingFace model name (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `k_number`: Number of chunks retrieved by each retriever (default: 20)
- `embed_weight` / `bm25_weight`: Ensemble weights (default: 0.5 each)

## Usage Example

```python
from vector_embedding_advance import VectorLoader
from LLM_load import llm
from rankingV2 import Ranking
import json

# 1. Load and index documents
app = VectorLoader(["document.pdf"])
result = app.embedding_documents()
retriever = result["retriever"]
parent_data = result["parent"]

# 2. Augment query
query = "What is machine learning?"
alternative = llm(query)
query_2 = alternative.llm_call()
query_3 = json.loads(query_2)["alternative_query"]

# 3. Retrieve and rank
output = Ranking(query, query_3, retriever, parent_data, k=10)
context = output.quering()

# 4. (Future) Generate final answer with LLM
# final_answer = llm(context).llm_call()
```

## Advantages of This Architecture

1. **Hybrid Retrieval**: Combines lexical (BM25) and semantic (vector) matching for better recall
2. **Query Augmentation**: Increases retrieval robustness via LLM‑generated alternative queries
3. **Parent‑Child Chunking**: Balances granular retrieval (child chunks) with coherent context (parent chunks)
4. **Duplicate‑Aware Ranking**: Prioritizes parent chunks with multiple relevant child chunks
5. **Persistent Storage**: ChromaDB avoids re‑embedding on each run

## Future Improvements

1. **Implement final LLM generation step**
2. **Add re‑ranking models** (e.g., cross‑encoder) for better precision
3. **Support more document types** (DOCX, HTML, etc.)
4. **Add query‑understanding components** (entity extraction, query expansion)
5. **Implement streaming and caching for production deployment**

---

## Rendering the Diagrams

The Mermaid diagrams in this document are natively supported on GitHub. For local viewing:
1. Use the [Mermaid Live Editor](https://mermaid.live/)
2. Install a Markdown preview extension that supports Mermaid (e.g., VS Code with Markdown Preview Enhanced)
3. Alternatively, copy the Mermaid code blocks to any Mermaid‑compatible viewer

## License & Attribution

This pipeline documentation is part of the Intermediate‑RAG‑pipeline project. Diagrams created with Mermaid.