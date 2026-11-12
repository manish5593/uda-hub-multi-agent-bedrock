"""The knowledge service's pgvector index (service/knowledge_service/pgvector_index.py).

Needs a Postgres with pgvector: set UDAHUB_TEST_POSTGRES_URL to a role that may create databases (CI uses
the pgvector/pgvector image). The fallback to the in-memory index runs on SQLite, without it.
"""
import os
import uuid

import pytest
from langchain_core.embeddings import Embeddings
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from agentic import config
from agentic.tools import db, knowledge_tools as kt
from service.knowledge_service import pgvector_index as pv
from service.migrations.common import upgrade

POSTGRES_ADMIN_URL = os.getenv("UDAHUB_TEST_POSTGRES_URL")
ACCOUNT = "acme"
ARTICLES = {
    "a-refund": ("Refund policy", "billing,refund", "Refunds for duplicate charges are reviewed by billing."),
    "a-qr": ("QR code not scanning", "app,qr", "Turn up the screen brightness and show the reservation id."),
    "a-pass": ("Reset your password", "login,password", "Tap forgot password on the login screen."),
}
VOCAB = ["refund", "charge", "billing", "qr", "scan", "brightness", "password", "login", "forgot"]


class WordEmbeddings(Embeddings):
    """Counts vocabulary words, so similar texts get similar vectors. Records what it embedded."""

    def __init__(self):
        self.documents = []

    def _vector(self, text_: str) -> list[float]:
        words = text_.lower()
        # The constant last dimension keeps vectors non-zero, so cosine distance is always defined
        return [float(words.count(w)) for w in VOCAB] + [0.01]

    def embed_documents(self, texts):
        self.documents.extend(texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, text_):
        return self._vector(text_)


@pytest.fixture
def knowledge_db(monkeypatch):
    """A fresh knowledge database with pgvector and three articles; the service's engine points at it."""
    if not POSTGRES_ADMIN_URL:
        pytest.skip("UDAHUB_TEST_POSTGRES_URL not set")
    name = f"udahub_test_{uuid.uuid4().hex[:12]}"
    admin = create_engine(POSTGRES_ADMIN_URL, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        if not conn.execute(text("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'")).first():
            pytest.skip("pgvector is not installed on the test server")
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    url = make_url(POSTGRES_ADMIN_URL).set(database=name).render_as_string(hide_password=False)
    engine = create_engine(url)
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION vector"))
        upgrade("knowledge", url)
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO accounts (account_id, account_name) VALUES (:a, 'Acme')"), {"a": ACCOUNT})
            for article_id, (title, tags, content) in ARTICLES.items():
                conn.execute(text("INSERT INTO knowledge (article_id, account_id, title, content, tags) "
                                  "VALUES (:id, :a, :t, :c, :g)"),
                             {"id": article_id, "a": ACCOUNT, "t": title, "c": content, "g": tags})
        monkeypatch.setattr(config, "UDAHUB_DB_URL", url)
        embeddings = WordEmbeddings()
        monkeypatch.setattr(pv, "get_embeddings", lambda: embeddings)
        yield engine, embeddings
    finally:
        engine.dispose()
        db._engine.cache_clear()
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


def rows(engine) -> dict:
    with engine.connect() as conn:
        return dict(conn.execute(text("SELECT article_id, model FROM knowledge_embeddings")).all())


def test_search_ranks_by_cosine_similarity(knowledge_db):
    retriever = pv.PgVectorRetriever(ACCOUNT)
    assert (retriever.store, retriever.mode, retriever.embedded) == ("pgvector", "embedding", 3)

    hits = retriever.search("I was charged twice, I want a refund")
    assert hits[0]["article_id"] == "a-refund"
    assert 0 < hits[0]["score"] <= 1
    assert retriever.search("my QR code won't scan")[0]["article_id"] == "a-qr"
    # Below the relevance threshold: nothing, so the agent escalates instead of guessing
    assert retriever.search("is there parking at the venue") == []


def test_only_new_or_changed_articles_are_embedded(knowledge_db):
    engine, embeddings = knowledge_db
    pv.PgVectorRetriever(ACCOUNT)
    assert len(embeddings.documents) == 3

    # A restart embeds nothing
    assert pv.PgVectorRetriever(ACCOUNT).embedded == 0
    assert len(embeddings.documents) == 3

    # An edited article is embedded again; the others are not
    with engine.begin() as conn:
        conn.execute(text("UPDATE knowledge SET content = 'Login help: tap forgot password.' "
                          "WHERE article_id = 'a-pass'"))
    assert pv.PgVectorRetriever(ACCOUNT).embedded == 1
    assert "Login help" in embeddings.documents[-1]


def test_deleted_articles_leave_the_index(knowledge_db):
    engine, _ = knowledge_db
    pv.PgVectorRetriever(ACCOUNT)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM knowledge WHERE article_id = 'a-qr'"))
    assert "a-qr" not in rows(engine)  # ON DELETE CASCADE
    hits = pv.PgVectorRetriever(ACCOUNT).search("my QR code won't scan")
    assert all(h["article_id"] != "a-qr" for h in hits)


def test_index_rebuilds_itself_after_the_articles_are_reloaded(knowledge_db):
    """seed --reset deletes the articles and inserts them again: the cascade empties the index under a
    running service, which must not then answer every question from nothing."""
    engine, embeddings = knowledge_db
    retriever = pv.PgVectorRetriever(ACCOUNT)
    with engine.begin() as conn:
        saved = conn.execute(text("SELECT article_id, account_id, title, content, tags FROM knowledge")).all()
        conn.execute(text("DELETE FROM knowledge"))
        for r in saved:
            conn.execute(text("INSERT INTO knowledge (article_id, account_id, title, content, tags) "
                              "VALUES (:id, :a, :t, :c, :g)"),
                         {"id": r.article_id, "a": r.account_id, "t": r.title, "c": r.content, "g": r.tags})
    assert rows(engine) == {}

    assert retriever.search("I was charged twice, I want a refund")[0]["article_id"] == "a-refund"
    assert retriever.embedded == 3 and len(rows(engine)) == 3


def test_each_model_has_its_own_vectors(knowledge_db, monkeypatch):
    engine, embeddings = knowledge_db
    pv.PgVectorRetriever(ACCOUNT)
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "another-model")
    retriever = pv.PgVectorRetriever(ACCOUNT)
    assert retriever.embedded == 3
    with engine.connect() as conn:
        models = conn.execute(text("SELECT model, count(*) FROM knowledge_embeddings GROUP BY model")).all()
    # The old model's vectors stay (switching back costs nothing) and are never compared with the new ones
    assert sorted(count for _, count in models) == [3, 3]
    assert retriever.search("refund for a duplicate charge")[0]["article_id"] == "a-refund"


def test_falls_back_to_memory_without_pgvector(monkeypatch):
    """On SQLite (the notebooks' databases) the same class keeps its index in memory."""
    embeddings = WordEmbeddings()
    monkeypatch.setattr(kt, "get_embeddings", lambda: embeddings)
    retriever = pv.PgVectorRetriever(config.DEFAULT_ACCOUNT_ID)
    assert (retriever.store, retriever.mode) == ("memory", "embedding")
    assert len(embeddings.documents) == len(retriever.articles) > 0
