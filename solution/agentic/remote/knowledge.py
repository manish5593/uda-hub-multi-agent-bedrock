"""Remote knowledge-base retrieval (Knowledge service), used by get_retriever() in knowledge_tools.py."""
from urllib.parse import quote

from agentic import config
from agentic.remote.http import call


class RemoteRetriever:
    """Same `search` interface as KnowledgeRetriever; the index lives in the Knowledge service."""

    def __init__(self, account_id: str):
        self.account_id = account_id

    def search(self, query: str, k: int = config.RAG_TOP_K) -> list[dict]:
        result = call("knowledge", "POST", f"/accounts/{quote(self.account_id, safe='')}/search",
                      json={"query": query, "k": k})
        return result["articles"]
