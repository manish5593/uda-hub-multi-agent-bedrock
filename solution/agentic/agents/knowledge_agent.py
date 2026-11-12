"""Knowledge resolver: answers how-to and policy questions with RAG over the knowledge base."""
from langchain_core.messages import SystemMessage

from agentic.agents.common import add_trace, conversation, customer_context_block, last_user_message, text_of
from agentic.llm import get_llm
from agentic.tools.knowledge_tools import format_articles, get_retriever

KNOWLEDGE_PROMPT = """You are a friendly CultPass customer support agent.
Answer the customer's latest message using ONLY the knowledge base articles below.
- Follow the articles' guidance and adapt their 'Suggested phrasing'; never invent policies, prices or steps.
- If the articles don't answer the question, say so and that you'll pass it to the team. Never send the customer
  to other sources (venue websites, event pages, third parties) that the articles don't mention.
- Personalise using the customer context (name, preferences) and answer in the customer's language ({language}).
- Be concise: short paragraphs or a few bullet points. Don't mention 'articles' or internal notes.

Customer context:
{context}

Knowledge base articles:
{articles}"""


def resolve_with_knowledge(state: dict) -> dict:
    classification = state.get("classification", {})
    customer = state.get("customer") or {}
    query = f"{classification.get('summary', '')} {last_user_message(state)}"
    articles = get_retriever(customer.get("account_id", "cultpass")).search(query)
    titles = [f"{a['title']} ({a['score']})" for a in articles]

    if not articles:
        return {
            "knowledge": [],
            "draft_response": None,
            "resolver": "knowledge_agent",
            "trace": add_trace(state, "[knowledge_agent] no relevant article found", event="retrieve", articles=[]),
        }

    response = get_llm().invoke([
        SystemMessage(content=KNOWLEDGE_PROMPT.format(
            language=classification.get("language", "English"),
            context=customer_context_block(state),
            articles=format_articles(articles),
        )),
        *conversation(state),
    ])
    return {
        "knowledge": articles,
        "draft_response": text_of(response),
        "resolver": "knowledge_agent",
        "trace": add_trace(state, f"[knowledge_agent] retrieved {titles}", event="retrieve",
                           articles=[{"title": a["title"], "score": a["score"]} for a in articles]),
    }
