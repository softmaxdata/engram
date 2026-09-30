"""XML-structured renderer optimized for Claude models."""

from __future__ import annotations

from engram.core.models import ConceptNode, ConceptType, IntentAnchor
from engram.renderers.base import ContextRenderer


class ClaudeRenderer(ContextRenderer):
    """Renders concept graph into XML-structured context for Claude."""

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
        sections: list[str] = ["<context>"]

        if core_memory:
            sections.append("  <core_memory>")
            sections.append(f"    {core_memory}")
            sections.append("  </core_memory>")

        if intent:
            sections.append("  <intent>")
            sections.append(f"    <objective>{intent.objective}</objective>")
            if intent.success_criteria:
                sections.append("    <success_criteria>")
                for sc in intent.success_criteria:
                    sections.append(f"      <criterion>{sc}</criterion>")
                sections.append("    </success_criteria>")
            if intent.constraints:
                sections.append("    <constraints>")
                for c in intent.constraints:
                    sections.append(f"      <constraint>{c}</constraint>")
                sections.append("    </constraints>")
            sections.append("  </intent>")

        # Group concepts by type
        decisions = [c for c in concepts if c.type == ConceptType.DECISION]
        facts = [c for c in concepts if c.type == ConceptType.FACT]
        constraints = [c for c in concepts if c.type == ConceptType.CONSTRAINT]
        entities = [c for c in concepts if c.type == ConceptType.ENTITY]
        goals = [c for c in concepts if c.type == ConceptType.GOAL]
        procedures = [c for c in concepts if c.type == ConceptType.PROCEDURE]
        observations = [c for c in concepts if c.type == ConceptType.OBSERVATION]
        other = [
            c
            for c in concepts
            if c.type
            not in {
                ConceptType.DECISION,
                ConceptType.FACT,
                ConceptType.CONSTRAINT,
                ConceptType.ENTITY,
                ConceptType.GOAL,
                ConceptType.PROCEDURE,
                ConceptType.OBSERVATION,
            }
        ]

        for section_name, section_concepts in [
            ("key_decisions", decisions),
            ("relevant_facts", facts),
            ("active_constraints", constraints),
            ("entities", entities),
            ("goals", goals),
            ("procedures", procedures),
            ("observations", observations),
            ("other", other),
        ]:
            if not section_concepts:
                continue
            block = self._render_section(section_name, section_concepts, usage_stats)
            sections.append(block)

        if worked_examples:
            sections.append("  <worked_examples>")
            sections.append(
                "    <!-- Nearest prior inputs from this context. Verify before "
                "copying; they're retrieved by semantic similarity. -->"
            )
            for i, ex in enumerate(worked_examples, 1):
                sections.append(f"    <example index=\"{i}\">")
                inp = (ex.get("input") or "").strip()
                out = (ex.get("output") or "").strip()
                if inp:
                    sections.append(f"      <input>{inp}</input>")
                if out:
                    sections.append(f"      <bullets_produced>{out}</bullets_produced>")
                sections.append("    </example>")
            sections.append("  </worked_examples>")

        sections.append("</context>")
        return "\n".join(sections)

    def estimate_tokens(self, text: str) -> int:
        """Approximate token count: ~4 characters per token for Claude."""
        return len(text) // 4 + 1 if text else 0

    def _render_section(
        self, name: str, concepts: list[ConceptNode],
        usage_stats: dict[str, str] | None = None,
    ) -> str:
        lines = [f"  <{name}>"]
        for c in concepts:
            confidence = f' confidence="{c.confidence:.1f}"' if c.confidence < 1.0 else ""
            tags = f' tags="{",".join(c.domain_tags)}"' if c.domain_tags else ""
            suffix = self._usage_suffix(c, usage_stats)
            lines.append(f"    <concept{confidence}{tags}>{c.content}{suffix}</concept>")
        lines.append(f"  </{name}>")
        return "\n".join(lines)
