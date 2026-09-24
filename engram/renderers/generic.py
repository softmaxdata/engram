"""Generic plain-text renderer as a fallback for any model."""

from __future__ import annotations

from engram.core.models import ConceptNode, IntentAnchor
from engram.renderers.base import ContextRenderer


class GenericRenderer(ContextRenderer):
    """Fallback renderer — outputs plain structured text."""

    def render(
        self,
        concepts: list[ConceptNode],
        intent: IntentAnchor | None,
        token_budget: int,
        core_memory: str = "",
        worked_examples: list[dict] | None = None,
        usage_stats: dict[str, str] | None = None,
    ) -> str:
        return self.render_with_selection(
            concepts, intent, token_budget,
            core_memory=core_memory, worked_examples=worked_examples, usage_stats=usage_stats,
        )[0]

    def _render_full(
        self,
        concepts: list[ConceptNode],
        intent: IntentAnchor | None,
        token_budget: int,
        core_memory: str = "",
        worked_examples: list[dict] | None = None,
        usage_stats: dict[str, str] | None = None,
    ) -> str:
        sections: list[str] = []

        if core_memory:
            sections.append(f"CORE MEMORY: {core_memory}")
            sections.append("")

        if intent:
            sections.append(f"OBJECTIVE: {intent.objective}")
            if intent.success_criteria:
                sections.append(
                    "SUCCESS CRITERIA: " + "; ".join(intent.success_criteria)
                )
            if intent.constraints:
                sections.append("CONSTRAINTS: " + "; ".join(intent.constraints))
            sections.append("")

        for concept in concepts:
            usage = self._usage_suffix(concept, usage_stats)
            line = f"[{concept.type.value.upper()}] {concept.content}{usage}"
            sections.append(line)

        if worked_examples:
            sections.append("")
            sections.append("WORKED EXAMPLES (verify before copying):")
            for i, ex in enumerate(worked_examples, 1):
                inp = (ex.get("input") or "").strip()
                out = (ex.get("output") or "").strip()
                if inp:
                    sections.append(f"  [{i}] INPUT: {inp}")
                if out:
                    sections.append(f"      BULLETS: {out}")

        return "\n".join(sections)

    def estimate_tokens(self, text: str) -> int:
        return len(text) // 4 + 1 if text else 0
