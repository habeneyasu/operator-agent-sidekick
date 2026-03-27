import json
from unittest.mock import patch

from src.tools.notification import send_pushover


def test_send_pushover_success():
    fake = json.dumps({"status": 1}).encode("utf-8")

    class FakeResponse:
        def read(self):
            return fake

    class FakeCM:
        def __enter__(self):
            return FakeResponse()

        def __exit__(self, *args):
            return None

    with patch("urllib.request.urlopen", return_value=FakeCM()):
        out = send_pushover("tok", "usr", "hello")

    assert "sent" in out.lower()
