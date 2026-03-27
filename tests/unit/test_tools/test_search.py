import json
from unittest.mock import patch

from src.tools.search import run_serper_search


def test_run_serper_search_formats_organic():
    payload = {
        "organic": [
            {
                "title": "T1",
                "link": "https://a.example",
                "snippet": "S1",
            }
        ]
    }
    fake_body = json.dumps(payload).encode("utf-8")

    class FakeResponse:
        def read(self):
            return fake_body

    class FakeCM:
        def __enter__(self):
            return FakeResponse()

        def __exit__(self, *args):
            return None

    with patch("urllib.request.urlopen", return_value=FakeCM()):
        out = run_serper_search("key", "q")

    assert "T1" in out
    assert "https://a.example" in out
    assert "S1" in out
