import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from app import _sanitize

def test_sanitize_strips_and_replaces_newlines():
    assert _sanitize("  hello\nworld \r ") == "hello world"

def test_sanitize_only_whitespace_returns_empty_string():
    assert _sanitize(" \n\r ") == ""
