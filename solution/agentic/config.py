"""Central configuration for UDA-Hub. Values can be overridden via env vars."""
import os
from pathlib import Path


def _env(name: str, default=None):
    """A setting from the environment. Empty counts as unset: Docker Compose passes settings left blank in
    service/.env as empty strings, which would otherwise override the defaults below."""
    value = os.getenv(name)
    return value if value not in (None, "") else default


STARTER_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = STARTER_DIR / "data"

UDAHUB_DB_PATH = Path(_env("UDAHUB_DB_PATH", DATA_DIR / "core" / "udahub.db"))
CULTPASS_DB_PATH = Path(_env("CULTPASS_DB_PATH", DATA_DIR / "external" / "cultpass.db"))
# SQLAlchemy URLs (e.g. postgresql+psycopg://...) that replace the SQLite files above when set.
# The microservices in service/ use these; the notebooks and tests keep using the SQLite files.
UDAHUB_DB_URL = _env("UDAHUB_DB_URL") or None
CULTPASS_DB_URL = _env("CULTPASS_DB_URL") or None

# Microservices (see service/README.md). When a URL is set, the matching tools call that service over
# HTTP instead of opening its database, so the agent never touches another service's data directly.
CULTPASS_SERVICE_URL = _env("CULTPASS_SERVICE_URL") or None
TICKET_SERVICE_URL = _env("TICKET_SERVICE_URL") or None
KNOWLEDGE_SERVICE_URL = _env("KNOWLEDGE_SERVICE_URL") or None
# Shared secret the services require on every internal request (header X-Internal-Token)
INTERNAL_API_TOKEN = _env("INTERNAL_API_TOKEN") or None
SERVICE_TIMEOUT_SECONDS = float(_env("UDAHUB_SERVICE_TIMEOUT", 15))

DEFAULT_ACCOUNT_ID = _env("UDAHUB_DEFAULT_ACCOUNT", "cultpass")
GUEST_EXTERNAL_USER_ID = "guest"

# Model provider: "openai" (OpenAI API) or "bedrock" (Amazon Bedrock, using the standard AWS credential chain)
LLM_PROVIDER = _env("UDAHUB_LLM_PROVIDER", "openai").lower()
AWS_REGION = _env("AWS_REGION", _env("AWS_DEFAULT_REGION", "us-east-1"))

# Embeddings follow the chat provider unless set separately. "sagemaker" uses a SageMaker endpoint
# deployed with deploy/sagemaker_embeddings.py.
EMBEDDING_PROVIDER = _env("UDAHUB_EMBEDDING_PROVIDER", LLM_PROVIDER).lower()

_DEFAULT_CHAT_MODELS = {
    "openai": "gpt-4o-mini",
    # Bedrock ids depend on the region and on which models the account has enabled
    "bedrock": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
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
CHAT_MODEL = _env("UDAHUB_CHAT_MODEL", _DEFAULT_CHAT_MODELS[LLM_PROVIDER])
EMBEDDING_MODEL = _env("UDAHUB_EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODELS[EMBEDDING_PROVIDER])
SAGEMAKER_ENDPOINT = _env("UDAHUB_SAGEMAKER_ENDPOINT", "udahub-embeddings")

# Decision thresholds
CLASSIFIER_MIN_CONFIDENCE = float(_env("UDAHUB_CLASSIFIER_MIN_CONFIDENCE", 0.5))
RESOLUTION_MIN_CONFIDENCE = float(_env("UDAHUB_RESOLUTION_MIN_CONFIDENCE", 0.7))
# Minimum retrieval score for a KB article to count as relevant. Each embedding model scores on its own scale:
# Titan's right articles score from 0.26 ("my card was charged twice") and questions no article covers at most
# 0.20 (parking), where OpenAI's are 0.45 and 0.41, so each provider has its own default.
_DEFAULT_RAG_MIN_SCORES = {"openai": 0.30, "bedrock": 0.24, "sagemaker": 0.30}
RAG_MIN_SCORE_EMBEDDING = float(_env("UDAHUB_RAG_MIN_SCORE", _DEFAULT_RAG_MIN_SCORES[EMBEDDING_PROVIDER]))
RAG_MIN_SCORE_KEYWORD = 0.10
RAG_TOP_K = 3

# How much context is fed to the agents
MAX_CONVERSATION_MESSAGES = 12
MAX_PRIOR_TICKETS = 3
MAX_MEMORIES = 10

# Structured event log: one JSON object per agent decision (JSON Lines)
EVENT_LOG_PATH = Path(_env("UDAHUB_EVENT_LOG", STARTER_DIR / "logs" / "events.jsonl"))
