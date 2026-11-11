"""Langfuse tracing is optional: off without keys, and when on it groups each ticket's turns into a session."""
from agentic import tracing


def test_off_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    assert tracing.run_config("T-1", {"external_user_id": "u1"}) == {"configurable": {"thread_id": "T-1"}}


def test_on_with_keys(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")

    class FakeHandler:
        pass

    monkeypatch.setattr(tracing, "_callback_handler_class", lambda: FakeHandler)

    config = tracing.run_config("T-1", {"external_user_id": "u1", "email": "a@b.c", "channel": "chat",
                                        "account_id": "cultpass"})

    assert config["configurable"] == {"thread_id": "T-1"}
    assert config["run_name"] == tracing.TRACE_NAME
    assert isinstance(config["callbacks"][0], FakeHandler)
    assert config["metadata"] == {"langfuse_trace_name": tracing.TRACE_NAME,
                                  "langfuse_session_id": "T-1", "langfuse_user_id": "u1",
                                  "langfuse_tags": ["cultpass", "chat"]}
    # The email never reaches Langfuse
    assert "a@b.c" not in str(config["metadata"])


def test_new_handler_per_run(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setattr(tracing, "_callback_handler_class", lambda: object)

    assert tracing.run_config("T-1")["callbacks"][0] is not tracing.run_config("T-1")["callbacks"][0]


def test_off_when_package_missing(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setattr(tracing, "_callback_handler_class", lambda: None)

    assert tracing.run_config("T-1") == {"configurable": {"thread_id": "T-1"}}
