"""Knowledge service: owns the knowledge-base articles and their embedding index (RAG retrieval).

Only this service needs the embedding model's credentials. The index is built on first use per account
(at startup for the default account). UDAHUB_VECTOR_STORE chooses where the embeddings live:

* pgvector (docker-compose's default): in the knowledge database, so only new or changed articles are
  embedded and every replica shares one index (pgvector_index.py).
* memory (the default outside docker-compose): rebuilt in memory on every start, as in the notebooks.

Either way, search falls back to keywords if the embedding model is unavailable.
"""
import logging
import os
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field

from agentic import config
from agentic.remote.http import serving_locally
from agentic.tools import knowledge_tools as kt
from agentic.tools.db import udahub_engine
from service.common.app import create_app, ping_database, refuse_own_url

refuse_own_url("knowledge", config.KNOWLEDGE_SERVICE_URL)
logger = logging.getLogger("knowledge")

VECTOR_STORE = os.getenv("UDAHUB_VECTOR_STORE", "memory").lower()
if VECTOR_STORE == "pgvector":
    from service.knowledge_service.pgvector_index import PgVectorRetriever
    kt.retriever_class = PgVectorRetriever
elif VECTOR_STORE != "memory":
    raise RuntimeError(f"UDAHUB_VECTOR_STORE must be 'pgvector' or 'memory', not {VECTOR_STORE!r}")


@asynccontextmanager
async def lifespan(app):
    try:
        with serving_locally():
            retriever = kt.get_retriever(config.DEFAULT_ACCOUNT_ID)
        logger.info("Indexed %d articles for %s (%s search, %s index)", len(retriever.articles),
                    config.DEFAULT_ACCOUNT_ID, retriever.mode, retriever.store)
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
    return {"mode": retriever.mode, "store": retriever.store, "articles": len(retriever.articles),
            "embedded": getattr(retriever, "embedded", None)}
