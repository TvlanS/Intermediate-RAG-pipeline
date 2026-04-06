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
