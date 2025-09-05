import os
import sys

from openai import OpenAIError

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import app  # noqa: E402


class ErrorClient:
    class Chat:
        class Completions:
            def create(self, **kwargs):
                raise OpenAIError("boom")

        completions = Completions()

    chat = Chat()


def test_query_rhyme_score_uses_fallback(monkeypatch):
    monkeypatch.setattr(app, "_get_openai_client", lambda: ErrorClient())
    app._CACHE.clear()
    result = app.query_rhyme_score("cat", "hat")
    parsed = app.parse_output(result)
    assert parsed["rhyme type"] == "Perfect"
