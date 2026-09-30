"""Abstract base renderer for model-specific context formatting."""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any

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
          - usage_stats: maps str(concept.id) → "(used N×, success Y/Z)" suffix.
            Content-keyed annotations remain supported for existing callers.
        """

    @abc.abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a string."""

    @staticmethod
    def _usage_suffix(concept: ConceptNode, usage_stats: dict[str, str] | None) -> str:
        """Prefer identity so duplicate text cannot share another memory's stats."""
        if not usage_stats:
            return ""
        annotation = usage_stats.get(str(concept.id), usage_stats.get(concept.content, ""))
        return f" {annotation}" if annotation else ""

    def _render_full(
        self,
        concepts: list[ConceptNode],
        intent: IntentAnchor | None,
        token_budget: int,
        **options: Any,
    ) -> str:
        # Compatibility for third-party renderers implementing the original API.
        return self.render(concepts, intent, token_budget, **options)

    def render_with_selection(
        self,
        concepts: list[ConceptNode],
        intent: IntentAnchor | None,
        token_budget: int,
        **options: Any,
    ) -> tuple[str, list[ConceptNode]]:
        """Pack complete concepts against the final format, returning exact IDs.

        Counts wrappers, separators, tags and optional annotations with this
        renderer's estimator. Oversized candidates are skipped, never a reason
        to stop considering smaller successors. Priority is core memory, intent,
        worked examples, then the supplied candidate order. Oversized core and
        intent text is shortened; examples and concepts remain whole.
        """
        budget = max(0, token_budget)
        if not budget:
            return "", []
        supplied = dict(options)
        selected_options = dict(options)
        if "core_memory" in options:
            selected_options["core_memory"] = ""
        if "worked_examples" in options:
            selected_options["worked_examples"] = None
        selected_intent = None

        def full(items: list[ConceptNode], anchor: IntentAnchor | None) -> str:
            # Internal full renderers do not trim. The large budget keeps
            # original third-party renderer implementations compatible too.
            return self._render_full(items, anchor, 2**63 - 1, **selected_options)

        def fits(text: str) -> bool:
            return self.estimate_tokens(text) <= budget

        def prefix(text: str, render: Callable[[str], str]) -> str:
            if fits(render(text)):
                return text
            low, high = 0, len(text)
            best = ""
            while low <= high:
                middle = (low + high) // 2
                candidate = text[:middle].rstrip() + " …[truncated]" if middle else ""
                if fits(render(candidate)):
                    best = candidate
                    low = middle + 1
                else:
                    high = middle - 1
            return best

        if supplied.get("core_memory"):

            def render_core(value: str) -> str:
                selected_options["core_memory"] = value
                return full([], None)

            selected_options["core_memory"] = prefix(supplied["core_memory"], render_core)

        if intent:
            if fits(full([], intent)):
                selected_intent = intent
            else:
                anchor = intent.model_copy(update={"success_criteria": [], "constraints": []})
                objective = prefix(
                    anchor.objective,
                    lambda value: full([], anchor.model_copy(update={"objective": value})),
                )
                anchor = anchor.model_copy(update={"objective": objective})
                if objective and fits(full([], anchor)):
                    selected_intent = anchor
                    for field in ("success_criteria", "constraints"):
                        for value in getattr(intent, field):
                            trial = selected_intent.model_copy(
                                update={field: [*getattr(selected_intent, field), value]}
                            )
                            if fits(full([], trial)):
                                selected_intent = trial

        if supplied.get("worked_examples"):
            examples = []
            for example in supplied["worked_examples"]:
                selected_options["worked_examples"] = [*examples, example]
                if fits(full([], selected_intent)):
                    examples.append(example)
            selected_options["worked_examples"] = examples or None

        selected: list[ConceptNode] = []
        rendered = full([], selected_intent)
        if not fits(rendered):
            return "", []
        for concept in concepts:
            candidate = full([*selected, concept], selected_intent)
            if fits(candidate):
                selected.append(concept)
                rendered = candidate
        return rendered, selected
