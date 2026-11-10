"""Knowledge service: owns the knowledge-base articles and their embedding index (RAG retrieval).

Only this service needs the embedding model's credentials. The index is built in memory from the
database on first use per account (at startup for the default account), and falls back to keyword
search if embeddings are unavailable, exactly as KnowledgeRetriever does in the notebooks.
"""
import logging
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field

from agentic import config
from agentic.remote.http import serving_locally
from agentic.tools import knowledge_tools as kt
from agentic.tools.db import udahub_engine
from service.common.app import create_app, ping_database, refuse_own_url

refuse_own_url("knowledge", config.KNOWLEDGE_SERVICE_URL)
logger = logging.getLogger("knowledge")


@asynccontextmanager
async def lifespan(app):
    try:
        with serving_locally():
            retriever = kt.get_retriever(config.DEFAULT_ACCOUNT_ID)
        logger.info("Indexed %d articles for %s (%s search)", len(retriever.articles),
                    config.DEFAULT_ACCOUNT_ID, retriever.mode)
    except Exception:
        # Not fatal: the index is built on the first search instead
        logger.exception("Could not build the knowledge index at startup")
    yield


app = create_app("knowledge", "UDA-Hub knowledge service", health_check=lambda: ping_database(udahub_engine()),
                 owns_data=True, lifespan=lifespan)


class SearchIn(BaseModel):
    query: str = Field(max_length=4000)
    k: int = Field(config.RAG_TOP_K, ge=1, le=10)


@app.post("/accounts/{account_id}/search")
def search(account_id: str, body: SearchIn):
    retriever = kt.get_retriever(account_id)
    return {"mode": retriever.mode, "articles": retriever.search(body.query, k=body.k)}


@app.post("/accounts/{account_id}/reindex")
def reindex(account_id: str):
    """Rebuild the account's index after its articles changed."""
    kt._retrievers.pop(account_id, None)
    retriever = kt.get_retriever(account_id)
    return {"mode": retriever.mode, "articles": len(retriever.articles)}
