"""Abstract base renderer for model-specific context formatting."""

from __future__ import annotations

import abc

from engram.core.models import ConceptNode, IntentAnchor


class ContextRenderer(abc.ABC):
    """Renders selected concepts into a prompt-ready string for a specific LLM."""

    @abc.abstractmethod
    def render(
        self,
        concepts: list[ConceptNode],
        intent: IntentAnchor | None,
        token_budget: int,
        core_memory: str = "",
        worked_examples: list[dict] | None = None,
        usage_stats: dict[str, str] | None = None,
    ) -> str:
        """Render concepts into a formatted context string.

        Optional extensions:
          - core_memory: Mem-α-style always-in-context summary (rendered first).
          - worked_examples: DC-style nearest prior-input/output pairs (rendered
            last, adjacent to the question).
          - usage_stats: maps concept.content → "(used N×, success Y/Z)" suffix.
        """

    @abc.abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a string."""
