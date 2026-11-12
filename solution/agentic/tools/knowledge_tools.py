"""Retrieval-Augmented Generation over the account's knowledge base (udahub `knowledge` table)."""
import logging
import math
import re
from collections import Counter

from langchain_core.documents import Document
from langchain_core.tools import StructuredTool
from langchain_core.vectorstores import InMemoryVectorStore

from agentic import config
from agentic.llm import get_embeddings
from agentic.remote.http import is_remote
from agentic.tools.db import session_scope, udahub_engine
from data.models import udahub

logger = logging.getLogger(__name__)

_STOPWORDS = set(
    "a an the and or of to in on for is are was be it my i me we you your can how do does what when "
    "with this that at from as by not have has why where who will would could should please "
    "work working help need want get use make know tell hi hello thanks".split()
)


def _stem(token: str) -> str:
    """Very light suffix stripping so 'refunds'/'refunded' match 'refund'."""
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def _tokens(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS and len(t) > 1]


def load_articles(account_id: str) -> list[dict]:
    with session_scope(udahub_engine()) as session:
        rows = session.query(udahub.Knowledge).filter_by(account_id=account_id).all()
        return [
            {"article_id": r.article_id, "title": r.title, "content": r.content, "tags": r.tags or ""}
            for r in rows
        ]


def document_text(article: dict) -> str:
    """The text that is embedded for an article."""
    return f"{article['title']}\nTags: {article['tags']}\n{article['content']}"


class KnowledgeRetriever:
    """Semantic search with embeddings, kept in memory; falls back to keyword scoring if embeddings fail.

    Subclasses keep the embeddings elsewhere by overriding _build_index and _embedding_search (the
    knowledge service keeps them in Postgres with pgvector: service/knowledge_service/pgvector_index.py)."""

    # Where the embeddings live, for logs and the knowledge service's answers
    store = "memory"

    def __init__(self, account_id: str):
        self.account_id = account_id
        self.articles = load_articles(account_id)
        self.mode = "keyword"
        self._store = None
        try:
            if self.articles:
                self._build_index()
                self.mode = "embedding"
        except Exception as exc:  # e.g. no API access to the embedding model
            logger.warning("Embedding index unavailable (%s); using keyword retrieval.", exc)
        self._idf = self._build_idf()

    def _build_index(self) -> None:
        docs = [
            Document(page_content=document_text(a), metadata={k: a[k] for k in ("article_id", "title", "tags")})
            for a in self.articles
        ]
        self._store = InMemoryVectorStore(get_embeddings())
        self._store.add_documents(docs)

    def _embedding_search(self, query: str, k: int) -> list[tuple[dict, float]]:
        """The k nearest articles with their cosine similarity, best first."""
        by_id = {a["article_id"]: a for a in self.articles}
        return [
            (by_id[doc.metadata["article_id"]], score)
            for doc, score in self._store.similarity_search_with_score(query, k=k)
        ]

    def _build_idf(self) -> dict:
        df = Counter()
        for a in self.articles:
            df.update(set(_tokens(f"{a['title']} {a['tags']} {a['content']}")))
        n = max(len(self.articles), 1)
        return {t: math.log((n + 1) / (c + 0.5)) for t, c in df.items()}

    def _keyword_search(self, query: str, k: int) -> list[tuple[dict, float]]:
        q = set(_tokens(query))
        if not q:
            return []
        total = sum(self._idf.get(t, 0) for t in q) or 1
        scored = []
        for a in self.articles:
            # Title and tags count double: they're the strongest relevance signal
            head = set(_tokens(f"{a['title']} {a['tags']}"))
            body = set(_tokens(a["content"]))
            score = sum(self._idf.get(t, 0) * (2 if t in head else 1) for t in q if t in head | body)
            scored.append((a, score / (2 * total)))
        return sorted(scored, key=lambda x: x[1], reverse=True)[:k]

    @property
    def min_score(self) -> float:
        return config.RAG_MIN_SCORE_EMBEDDING if self.mode == "embedding" else config.RAG_MIN_SCORE_KEYWORD

    def search(self, query: str, k: int = config.RAG_TOP_K) -> list[dict]:
        """Return the top-k relevant articles (score >= threshold), best first."""
        if self.mode == "embedding":
            try:
                hits = self._embedding_search(query, k)
            except Exception as exc:
                logger.warning("Embedding search failed (%s); using keyword retrieval.", exc)
                hits = self._keyword_search(query, k)
        else:
            hits = self._keyword_search(query, k)
        return [
            {**article, "score": round(float(score), 3)}
            for article, score in hits
            if score >= self.min_score
        ]


_retrievers: dict[str, KnowledgeRetriever] = {}
# The class get_retriever builds; the knowledge service swaps in its pgvector retriever
retriever_class: type[KnowledgeRetriever] = KnowledgeRetriever


def get_retriever(account_id: str) -> KnowledgeRetriever:
    if is_remote("knowledge"):
        # The index lives in the Knowledge service (KNOWLEDGE_SERVICE_URL); see agentic/remote
        from agentic.remote.knowledge import RemoteRetriever
        return RemoteRetriever(account_id)
    if account_id not in _retrievers:
        _retrievers[account_id] = retriever_class(account_id)
    return _retrievers[account_id]


def format_articles(articles: list[dict]) -> str:
    if not articles:
        return "No relevant knowledge base articles found."
    return "\n\n".join(
        f"[Article: {a['title']}] (relevance {a['score']})\n{a['content']}" for a in articles
    )


def build_knowledge_tool(account_id: str) -> StructuredTool:
    def search_knowledge_base(query: str) -> str:
        """Search the support knowledge base for policies and how-to guidance. Use before stating any policy."""
        return format_articles(get_retriever(account_id).search(query))

    return StructuredTool.from_function(search_knowledge_base)
