import json
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError

from src.tools.wikipedia import fetch_wikipedia_summary


def test_fetch_wikipedia_summary_parses_extract():
    payload = {
        "title": "Python",
        "extract": "A programming language.",
        "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Python"}},
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
        out = fetch_wikipedia_summary("Python")

    assert "Python" in out
    assert "programming language" in out


def test_fetch_wikipedia_http404():
    def open_url(*args, **kwargs):
        raise HTTPError("url", 404, "Not Found", hdrs=None, fp=BytesIO(b"{}"))

    with patch("urllib.request.urlopen", side_effect=open_url):
        out = fetch_wikipedia_summary("missing_page")

    assert "No Wikipedia" in out or "404" in out
