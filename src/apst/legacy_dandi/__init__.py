"""Self-contained DANDI000688 data and association-profile utilities.

This namespace isolates the Sub-C/Sub-M release dependency closure from any
pre-existing ``apst.dandi`` implementation.
"""

from apst.legacy_dandi.protocol import (
    DEVELOPMENT_SESSIONS,
    FINAL_SESSIONS,
    SOURCE_SESSIONS,
)

__all__ = ["SOURCE_SESSIONS", "DEVELOPMENT_SESSIONS", "FINAL_SESSIONS"]
