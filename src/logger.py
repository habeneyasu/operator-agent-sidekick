from __future__ import annotations

import logging
import os
from typing import Optional

_LOGGER: Optional[logging.Logger] = None


def get_logger(name: str = "sidekick") -> logging.Logger:
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER.getChild(name) if name != "sidekick" else _LOGGER

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    _LOGGER = logging.getLogger("sidekick")
    _LOGGER.setLevel(level)
    return _LOGGER.getChild(name) if name != "sidekick" else _LOGGER

