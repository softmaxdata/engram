"""Markdown renderer optimized for GPT models."""

from __future__ import annotations

from engram.core.models import ConceptNode, ConceptType, IntentAnchor
from engram.renderers.base import ContextRenderer


class GPTRenderer(ContextRenderer):
    """Renders concept graph into markdown for GPT models."""

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
            sections.append("## Core Memory\n")
            sections.append(core_memory)
            sections.append("")

        if intent:
            sections.append("# Project Context")
            sections.append(f"\n**Objective:** {intent.objective}\n")
            if intent.success_criteria:
                sections.append("**Success Criteria:**")
                for sc in intent.success_criteria:
                    sections.append(f"- {sc}")
                sections.append("")
            if intent.constraints:
                sections.append("**Constraints:**")
                for c in intent.constraints:
                    sections.append(f"- {c}")
                sections.append("")

        groups: list[tuple[str, ConceptType]] = [
            ("Key Decisions", ConceptType.DECISION),
            ("Known Facts", ConceptType.FACT),
            ("Active Constraints", ConceptType.CONSTRAINT),
            ("Entities", ConceptType.ENTITY),
            ("Goals", ConceptType.GOAL),
            ("Procedures", ConceptType.PROCEDURE),
            ("Observations", ConceptType.OBSERVATION),
        ]

        categorized_types = {ct for _, ct in groups}
        current_tokens = self.estimate_tokens("\n".join(sections))

        for heading, concept_type in groups:
            items = [c for c in concepts if c.type == concept_type]
            if not items:
                continue
            block = self._render_group(heading, items, usage_stats)
            block_tokens = self.estimate_tokens(block)
            if current_tokens + block_tokens > token_budget:
                break
            sections.append(block)
            current_tokens += block_tokens

        # Remaining types
        other = [c for c in concepts if c.type not in categorized_types]
        if other:
            block = self._render_group("Other Context", other, usage_stats)
            block_tokens = self.estimate_tokens(block)
            if current_tokens + block_tokens <= token_budget:
                sections.append(block)

        if worked_examples:
            sections.append("\n## Worked Examples\n")
            sections.append(
                "_Nearest prior inputs from this context. Verify before copying — "
                "they're retrieved by semantic similarity._\n"
            )
            for i, ex in enumerate(worked_examples, 1):
                inp = (ex.get("input") or "").strip()
                out = (ex.get("output") or "").strip()
                sections.append(f"### Example {i}")
                if inp:
                    sections.append(f"**Input:** {inp}")
                if out:
                    sections.append(f"**Bullets produced:** {out}")
                sections.append("")

        return "\n".join(sections)

    def estimate_tokens(self, text: str) -> int:
        """Use tiktoken for GPT models when available, fallback to ~4 chars/token."""
        try:
            import tiktoken

            enc = tiktoken.encoding_for_model("gpt-4o")
            return len(enc.encode(text))
        except Exception:
            return len(text) // 4 + 1

    def _render_group(
        self, heading: str, concepts: list[ConceptNode],
        usage_stats: dict[str, str] | None = None,
    ) -> str:
        lines = [f"\n## {heading}\n"]
        for c in concepts:
            tag_str = f" `{', '.join(c.domain_tags)}`" if c.domain_tags else ""
            usage = f" {usage_stats[c.content]}" if usage_stats and c.content in usage_stats else ""
            lines.append(f"- {c.content}{tag_str}{usage}")
        return "\n".join(lines)
