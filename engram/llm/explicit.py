"""Isolated provider configuration for standalone maintenance/evaluation commands.

LiteLLM has process-global configuration, loads .env in development mode, and
can print diagnostics. This context is for standalone commands, never request
handlers or concurrent application tasks. Their only output should be their
own structured, sanitized result.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import TextIOBase


class _Discard(TextIOBase):
    @property
    def encoding(self) -> str:
        return "utf-8"

    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        pass


@contextmanager
def explicit_embedding_runtime():
    """Disable implicit dotenv loading and discard raw SDK diagnostics.

    Restore ambient process settings on exit, including on import/provider
    failure. Import the SDK only here, after setting production mode. Callers
    must enter this context only for an explicitly requested provider operation.
    """
    settings = {"LITELLM_MODE": "PRODUCTION", "LITELLM_LOCAL_MODEL_COST_MAP": "True"}
    previous = {key: os.environ.get(key) for key in settings}
    previous_logging = logging.root.manager.disable
    os.environ.update(settings)
    logging.disable(logging.CRITICAL)
    try:
        with redirect_stdout(_Discard()), redirect_stderr(_Discard()):
            import litellm

            previous_debug = litellm.suppress_debug_info
            previous_verbose = litellm.set_verbose
            litellm.suppress_debug_info = True
            litellm.set_verbose = False
            try:
                yield
            finally:
                litellm.suppress_debug_info = previous_debug
                litellm.set_verbose = previous_verbose
    finally:
        logging.disable(previous_logging)
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
