"""Generic plain-text renderer as a fallback for any model."""

from __future__ import annotations

from engram.core.models import ConceptNode, ConceptType, IntentAnchor
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

        current_tokens = self.estimate_tokens("\n".join(sections))

        for concept in concepts:
            usage = (
                f" {usage_stats[concept.content]}"
                if usage_stats and concept.content in usage_stats else ""
            )
            line = f"[{concept.type.value.upper()}] {concept.content}{usage}"
            line_tokens = self.estimate_tokens(line)
            if current_tokens + line_tokens > token_budget:
                break
            sections.append(line)
            current_tokens += line_tokens

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
        return len(text) // 4 + 1
