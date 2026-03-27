"""Push notification via Pushover (optional)."""

from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request

from src.config import Settings
from src.tools.spec import SidekickTool

PUSHOVER_URL = "https://api.pushover.net/1/messages.json"


def send_pushover(
    token: str,
    user: str,
    message: str,
    *,
    timeout: float = 15.0,
) -> str:
    payload = json.dumps(
        {"token": token, "user": user, "message": message[:1024]}
    ).encode("utf-8")
    request = urllib.request.Request(
        PUSHOVER_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, ssl.SSLError, TimeoutError, OSError) as exc:
        return f"Pushover error: {exc}"

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return "Pushover error: invalid response"
    if data.get("status") != 1:
        return f"Pushover error: {data}"
    return "Notification sent."


def notification_tool(settings: Settings) -> SidekickTool | None:
    if not settings.enable_notifications:
        return None
    if not settings.pushover_token or not settings.pushover_user:
        return None

    token = settings.pushover_token
    user = settings.pushover_user

    def invoke(message: str) -> str:
        text = message.strip()
        if not text:
            return "Error: empty notification message"
        return send_pushover(token, user, text)

    return SidekickTool(
        name="send_push_notification",
        description="Send a short push notification via Pushover (if enabled). Input: message text.",
        invoke=invoke,
    )
