"""Central configuration for UDA-Hub. Values can be overridden via env vars."""
import os
from pathlib import Path

STARTER_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = STARTER_DIR / "data"

UDAHUB_DB_PATH = Path(os.getenv("UDAHUB_DB_PATH", DATA_DIR / "core" / "udahub.db"))
CULTPASS_DB_PATH = Path(os.getenv("CULTPASS_DB_PATH", DATA_DIR / "external" / "cultpass.db"))

DEFAULT_ACCOUNT_ID = os.getenv("UDAHUB_DEFAULT_ACCOUNT", "cultpass")
GUEST_EXTERNAL_USER_ID = "guest"

# Model provider: "openai" (OpenAI API) or "bedrock" (Amazon Bedrock, using the standard AWS credential chain)
LLM_PROVIDER = os.getenv("UDAHUB_LLM_PROVIDER", "openai").lower()
AWS_REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))

# Embeddings follow the chat provider unless set separately. "sagemaker" uses a SageMaker endpoint
# deployed with deploy/sagemaker_embeddings.py.
EMBEDDING_PROVIDER = os.getenv("UDAHUB_EMBEDDING_PROVIDER", LLM_PROVIDER).lower()

_DEFAULT_CHAT_MODELS = {
    "openai": "gpt-4o-mini",
    # Bedrock ids depend on the region and on which models the account has enabled
    "bedrock": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
}
_DEFAULT_EMBEDDING_MODELS = {
    "openai": "text-embedding-3-small",
    "bedrock": "amazon.titan-embed-text-v2:0",
    # Hugging Face model id the SageMaker endpoint serves
    "sagemaker": "BAAI/bge-small-en-v1.5",
}
if LLM_PROVIDER not in _DEFAULT_CHAT_MODELS:
    raise ValueError(f"UDAHUB_LLM_PROVIDER must be one of {sorted(_DEFAULT_CHAT_MODELS)}, not {LLM_PROVIDER!r}")
if EMBEDDING_PROVIDER not in _DEFAULT_EMBEDDING_MODELS:
    raise ValueError(f"UDAHUB_EMBEDDING_PROVIDER must be one of {sorted(_DEFAULT_EMBEDDING_MODELS)}, "
                     f"not {EMBEDDING_PROVIDER!r}")
CHAT_MODEL = os.getenv("UDAHUB_CHAT_MODEL", _DEFAULT_CHAT_MODELS[LLM_PROVIDER])
EMBEDDING_MODEL = os.getenv("UDAHUB_EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODELS[EMBEDDING_PROVIDER])
SAGEMAKER_ENDPOINT = os.getenv("UDAHUB_SAGEMAKER_ENDPOINT", "udahub-embeddings")

# Decision thresholds
CLASSIFIER_MIN_CONFIDENCE = float(os.getenv("UDAHUB_CLASSIFIER_MIN_CONFIDENCE", 0.5))
RESOLUTION_MIN_CONFIDENCE = float(os.getenv("UDAHUB_RESOLUTION_MIN_CONFIDENCE", 0.7))
# Minimum retrieval score for a KB article to count as relevant
RAG_MIN_SCORE_EMBEDDING = float(os.getenv("UDAHUB_RAG_MIN_SCORE", 0.30))
RAG_MIN_SCORE_KEYWORD = 0.10
RAG_TOP_K = 3

# How much context is fed to the agents
MAX_CONVERSATION_MESSAGES = 12
MAX_PRIOR_TICKETS = 3
MAX_MEMORIES = 10

# Structured event log: one JSON object per agent decision (JSON Lines)
EVENT_LOG_PATH = Path(os.getenv("UDAHUB_EVENT_LOG", STARTER_DIR / "logs" / "events.jsonl"))
