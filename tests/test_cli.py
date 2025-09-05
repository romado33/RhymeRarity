import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import cli  # noqa: E402


def test_cli_prints_table(monkeypatch, capsys):
    monkeypatch.setattr(cli, "analyze_rhyme", lambda w1, w2: ("Perfect", "10", "", ""))
    cli.main(["cat", "hat"])
    captured = capsys.readouterr()
    assert "cat" in captured.out
    assert "hat" in captured.out
    assert "Perfect" in captured.out
    assert "10" in captured.out


def test_cli_requires_even_number():
    with pytest.raises(SystemExit):
        cli.main(["cat"])
