"""Which experiences fit a customer's stored preferences.

Left to the account agent, recommendations were padded: when nothing new matched Bob's taste for music and
dance, it suggested a sightseeing tour and said it "fits your interests". Embedding similarity can't decide
the match either (Christ the Redeemer scores 0.47 against "music and dance in Rio", the Carnival tour 0.46).

So the match is one narrow, structured question to the model, asked in code around it: only experiences
the customer can book and hasn't booked are candidates, the answer must name a candidate and quote a stored
preference exactly, and anything else is dropped. The agent then only sees the matches, or is told plainly
that nothing matches, so it has nothing to pad with.
"""
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agentic.llm import get_llm

MATCH_PROMPT = """You match cultural experiences to a customer's stored preferences.
For each experience, decide whether it clearly fits one of the preferences: something a person with that
preference would obviously choose. A famous sight is not a match for a music preference, and a museum is not a
match for an outdoor preference. When in doubt, leave it out. Quote the preference exactly as given."""


class Match(BaseModel):
    experience_id: str = Field(description="Id of an experience from the list")
    preference: str = Field(description="The stored preference it fits, quoted exactly")
    why: str = Field(description="A few words on how it fits that preference")


class Matches(BaseModel):
    matches: list[Match] = Field(description="Only experiences that clearly fit a preference; empty if none do")


def stored_preferences(memories: list[str]) -> list[str]:
    """The customer's preferences from their long-term memories ("(preference) Likes ..." entries)."""
    prefix = "(preference)"
    return [m[len(prefix):].strip() for m in memories or [] if m.lower().startswith(prefix)]


def match_preferences(preferences: list[str], experiences: list[dict]) -> list[dict]:
    """The experiences that fit a preference, each with `fits_preference` and `why`. Answers that name an
    experience not in the list, or a preference the customer doesn't have, are dropped."""
    if not preferences or not experiences:
        return []
    listing = "\n".join(f"- {e['experience_id']}: {e['title']}. {e['description']} ({e['location']})"
                        for e in experiences)
    answer: Matches = get_llm().with_structured_output(Matches).invoke([
        SystemMessage(content=MATCH_PROMPT),
        HumanMessage(content="Preferences:\n" + "\n".join(f"- {p}" for p in preferences)
                     + f"\n\nExperiences:\n{listing}"),
    ])
    by_id = {e["experience_id"]: e for e in experiences}
    known = {p.strip().lower(): p for p in preferences}
    results, seen = [], set()
    for m in answer.matches:
        preference = known.get(m.preference.strip().lower())
        if m.experience_id in by_id and preference and m.experience_id not in seen:
            seen.add(m.experience_id)
            results.append({**by_id[m.experience_id], "fits_preference": preference, "why": m.why})
    return results
