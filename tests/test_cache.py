import os
import sys
import json
from types import SimpleNamespace

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import app  # noqa: E402


def test_query_rhyme_score_uses_cache(monkeypatch, tmp_path):
    # Redirect cache file to temporary directory
    monkeypatch.setattr(app, "CACHE_FILE", tmp_path / "cache.json")
    app._CACHE.clear()

    calls = []

    class FakeClient:
        class Chat:
            class Completions:
                def create(self, **kwargs):
                    calls.append(kwargs)
                    return SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(content="cached")
                            )
                        ]
                    )

            completions = Completions()

        chat = Chat()

    monkeypatch.setattr(app, "_get_openai_client", lambda: FakeClient())

    first = app.query_rhyme_score("cat", "hat")
    assert first == "cached"
    assert len(calls) == 1

    second = app.query_rhyme_score("cat", "hat")
    assert second == "cached"
    assert len(calls) == 1  # still one call thanks to cache

    app._save_cache()
    assert app.CACHE_FILE.exists()
    with app.CACHE_FILE.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert "cat|hat|gpt-3.5-turbo|0.7" in data
