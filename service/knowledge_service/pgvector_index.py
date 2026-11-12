"""The knowledge service's embedding index in Postgres, with pgvector.

The in-memory index (KnowledgeRetriever) embeds every article again each time the service starts, and each
replica holds its own copy. This one keeps the embeddings in the knowledge database, in the table
knowledge_embeddings, so:

* Only new or changed articles are embedded, on startup or on POST /accounts/{id}/reindex. An article is
  "changed" when the hash of its embedded text differs; deleted articles' rows go with them (ON DELETE
  CASCADE), and rows of articles no longer in the account are removed on reindex.
* Every replica of the service searches the same index.
* Embeddings of different models never mix: each row records the model that made it, and a search only
  compares vectors of the configured model. Switching the model re-embeds everything once.

The table is a derived index, not data: it can always be rebuilt from the articles. So the service creates
it itself (like LangGraph's checkpoint tables in the agent's database), and it is outside the Alembic
history of the knowledge database. The vector column has no fixed dimension, so any embedding model fits
(1536 for OpenAI text-embedding-3-small, 1024 for Titan V2). Search is an exact scan, which suits a
knowledge base of this size; an approximate index (HNSW) needs a fixed dimension, so it would be added per
model if the base grew to many thousands of articles.

The extension itself is created by the db-setup job (postgres/init-databases.sh), because it needs a
superuser. If it is missing, the retriever falls back to the in-memory index and logs why.
"""
import hashlib
import logging

from sqlalchemy import text

from agentic import config
from agentic.llm import get_embeddings
from agentic.tools import knowledge_tools as kt
from agentic.tools.db import udahub_engine

logger = logging.getLogger("knowledge")

TABLE = "knowledge_embeddings"
CREATE_TABLE = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    article_id   VARCHAR NOT NULL REFERENCES knowledge (article_id) ON DELETE CASCADE,
    model        VARCHAR NOT NULL,
    account_id   VARCHAR NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    embedding    vector NOT NULL,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (article_id, model)
)"""
CREATE_INDEX = f"CREATE INDEX IF NOT EXISTS ix_{TABLE}_account_model ON {TABLE} (account_id, model)"


def model_name() -> str:
    """Identifies the embedding model, so vectors from different models are never compared."""
    return f"{config.EMBEDDING_PROVIDER}:{config.EMBEDDING_MODEL}"


def content_hash(article: dict) -> str:
    return hashlib.sha256(kt.document_text(article).encode()).hexdigest()


def vector_literal(values) -> str:
    """pgvector's text form, '[0.1,0.2,...]', passed as a parameter and cast in SQL."""
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


def pgvector_available(engine) -> bool:
    with engine.connect() as conn:
        return conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")).first() is not None


class PgVectorRetriever(kt.KnowledgeRetriever):
    """KnowledgeRetriever with the embeddings in Postgres. Scores are cosine similarity, as in memory, so
    the same relevance threshold (UDAHUB_RAG_MIN_SCORE) applies."""

    store = "pgvector"

    def _build_index(self) -> None:
        engine = udahub_engine()
        if engine.dialect.name != "postgresql" or not pgvector_available(engine):
            logger.warning("pgvector is not available in the knowledge database; keeping the index in memory")
            self.store = "memory"
            super()._build_index()
            return
        self._engine = engine
        self._embeddings = get_embeddings()
        self._model = model_name()
        self.embedded = self._sync()

    def _sync(self) -> int:
        """Embed the articles that are new or changed, drop rows of articles that are gone. Returns how many
        articles were embedded."""
        wanted = {a["article_id"]: (a, content_hash(a)) for a in self.articles}
        with self._engine.begin() as conn:
            conn.execute(text(CREATE_TABLE))
            conn.execute(text(CREATE_INDEX))
            # Two replicas starting at once embed the same articles once, not twice
            conn.execute(text("SELECT pg_advisory_xact_lock(hashtext(:key))"),
                         {"key": f"udahub-embed-{self.account_id}"})
            stored = dict(conn.execute(
                text(f"SELECT article_id, content_hash FROM {TABLE} WHERE account_id = :a AND model = :m"),
                {"a": self.account_id, "m": self._model}).all())
            todo = [(a, h) for a, h in wanted.values() if stored.get(a["article_id"]) != h]
            if todo:
                vectors = self._embeddings.embed_documents([kt.document_text(a) for a, _ in todo])
                conn.execute(text(f"""
                    INSERT INTO {TABLE} (article_id, model, account_id, content_hash, embedding)
                    VALUES (:id, :m, :a, :h, CAST(:v AS vector))
                    ON CONFLICT (article_id, model) DO UPDATE
                    SET content_hash = EXCLUDED.content_hash, embedding = EXCLUDED.embedding,
                        account_id = EXCLUDED.account_id, updated_at = now()"""),
                    [{"id": a["article_id"], "m": self._model, "a": self.account_id, "h": h,
                      "v": vector_literal(v)} for (a, h), v in zip(todo, vectors)])
            gone = set(stored) - set(wanted)
            for article_id in gone:
                conn.execute(text(f"DELETE FROM {TABLE} WHERE article_id = :id AND model = :m"),
                             {"id": article_id, "m": self._model})
        logger.info("pgvector index for %s: %d articles, %d embedded now, %d removed",
                    self.account_id, len(wanted), len(todo), len(gone))
        return len(todo)

    def _embedding_search(self, query: str, k: int) -> list[tuple[dict, float]]:
        if self.store != "pgvector":
            return super()._embedding_search(query, k)
        by_id = {a["article_id"]: a for a in self.articles}
        q = vector_literal(self._embeddings.embed_query(query))
        rows = self._nearest(q, k)
        if not rows and self.articles:
            # The query has no threshold, so no rows means the index lost them: the articles were deleted and
            # loaded again (seed --reset), or the table was dropped. Rebuild it once rather than answer from
            # an empty index, which would escalate every question.
            logger.warning("pgvector index for %s is empty; embedding the articles again", self.account_id)
            self.articles = kt.load_articles(self.account_id)
            by_id = {a["article_id"]: a for a in self.articles}
            self.embedded = self._sync()
            rows = self._nearest(q, k)
        return [(by_id[r.article_id], float(r.score)) for r in rows if r.article_id in by_id]

    def _nearest(self, q: str, k: int) -> list:
        with self._engine.connect() as conn:
            return conn.execute(text(f"""
                SELECT article_id, 1 - (embedding <=> CAST(:q AS vector)) AS score
                FROM {TABLE}
                WHERE account_id = :a AND model = :m
                ORDER BY embedding <=> CAST(:q AS vector)
                LIMIT :k"""), {"q": q, "a": self.account_id, "m": self._model, "k": k}).all()
