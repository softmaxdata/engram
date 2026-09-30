"""Tests for v0.4 features — canonical Reflector, raw input preservation, re-extraction.

Covers the 8 test cases from the v0.4 update prompt:
1. Same raw input from different agents produces identical bullets (canonical model)
2. Raw input is preserved and retrievable after commit
3. Duplicate raw inputs are detected and skipped
4. Re-extraction dry_run returns accurate previews
5. Re-extraction preserves high-value bullets even if new model doesn't reproduce them
6. Re-extraction removes low-value bullets not reproduced by new model
7. Activity records include extraction metadata (model, prompt version, bullet IDs)
8. Changing server config does NOT retroactively affect existing bullets
"""

from __future__ import annotations

import hashlib
import json
import uuid
from unittest.mock import AsyncMock

import pytest

from engram.core.config import IngestionConfig
from engram.core.ingestion import CuratorEngine, IngestionEngine, ReflectorEngine
from engram.core.models import (
    Activity,
    Bullet,
    BulletType,
    ContentType,
    Context,
    DeltaBatch,
    DeltaOpType,
    DeltaOperation,
    DeltaSource,
    IntentAnchor,
    Reflection,
    ReflectionInsight,
    ReExtractionPreview,
    ReExtractionRequest,
    ReExtractionResult,
    SourceType,
)
from engram.core.re_extraction import ReExtractionEngine
from engram.llm.adapter import LLMAdapter
from engram.storage.sqlite import SQLiteBackend


class MockLLMAdapter(LLMAdapter):
    """Mock LLM adapter that returns canned responses for Reflector."""

    def __init__(self):
        self.complete_response = json.dumps({
            "new_insights": [
                {
                    "content": "PaddleOCR is 40% faster than Textract",
                    "insight_type": "fact",
                    "suggested_section": "ocr_tools",
                    "evidence": "Performance benchmark data",
                    "novelty": 0.8,
                },
                {
                    "content": "Use PaddleOCR for field detection",
                    "insight_type": "decision",
                    "suggested_section": "architecture",
                    "evidence": "Based on performance comparison",
                    "novelty": 0.9,
                },
            ],
            "strategies_that_worked": [],
            "failure_modes": [],
            "prediction_errors": [],
            "open_questions": [],
            "confidence": 0.85,
        })

    async def complete(self, prompt, system=None, temperature=0.0, max_tokens=4096, response_format=None, model=None):
        return self.complete_response

    async def embed(self, text):
        h = hash(text)
        return [(h >> i & 0xFF) / 255.0 for i in range(8)]


@pytest.fixture
async def storage(tmp_path):
    db_path = str(tmp_path / "test.db")
    backend = SQLiteBackend(db_path=db_path)
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.fixture
def mock_llm():
    return MockLLMAdapter()


@pytest.fixture
def ingestion_config():
    return IngestionConfig(
        reflector_model="test-canonical-model",
        reflector_prompt_version="v1-test",
        max_reflection_rounds=1,
    )


@pytest.fixture
async def context_id(storage):
    intent = IntentAnchor(objective="Test v0.4 features")
    ctx = Context(name="v0.4 Test", intent=intent)
    await storage.create_context(ctx)
    return ctx.id


@pytest.fixture
def ingestion_engine(storage, mock_llm, ingestion_config):
    return IngestionEngine(storage, mock_llm, ingestion_config=ingestion_config)


class TestCanonicalReflector:
    """Test 1: Same raw input from different agents produces identical bullets."""

    async def test_same_input_different_agents_same_bullets(
        self, storage, mock_llm, ingestion_config, context_id,
    ):
        """When two agents commit the same raw content, the canonical Reflector
        ensures identical extraction (same model, same prompt version)."""
        engine = IngestionEngine(storage, mock_llm, ingestion_config=ingestion_config)
        content = "PaddleOCR is 40% faster than Textract. We decided to use PaddleOCR."

        # Agent 1 (Claude) commits
        batch1 = await engine.commit(
            context_id=context_id,
            agent_id="claude-agent",
            content=content,
            source_model="claude-sonnet-4-5",
        )
        # The second commit with identical content should be deduped
        batch2 = await engine.commit(
            context_id=context_id,
            agent_id="gpt-agent",
            content=content,
            source_model="gpt-4o",
        )

        # Both should return the same delta batch (dedup by hash)
        assert batch1.id == batch2.id

    async def test_reflector_uses_canonical_model(self, mock_llm, ingestion_config):
        """Reflector uses the server-configured canonical model."""
        reflector = ReflectorEngine(mock_llm, config=ingestion_config)
        assert reflector.config is not None
        assert reflector.config.reflector_model == "test-canonical-model"


class TestRawInputPreservation:
    """Test 2: Raw input is preserved and retrievable after commit."""

    async def test_raw_input_stored_in_activity(
        self, storage, mock_llm, ingestion_config, context_id,
    ):
        """After commit, the raw input is preserved in the activity ledger."""
        engine = IngestionEngine(storage, mock_llm, ingestion_config=ingestion_config)
        raw_content = "We discovered that PaddleOCR handles tables well."

        await engine.commit(
            context_id=context_id,
            agent_id="test-agent",
            content=raw_content,
        )

        # Retrieve activities with raw input
        activities = await storage.get_activities_with_raw_input(str(context_id))
        assert len(activities) >= 1

        activity = activities[0]
        assert activity.raw_input == raw_content
        assert activity.raw_input_hash is not None
        assert activity.raw_input_hash == hashlib.sha256(raw_content.encode()).hexdigest()

    async def test_raw_input_retrievable_by_hash(
        self, storage, mock_llm, ingestion_config, context_id,
    ):
        """Raw input can be looked up by its SHA-256 hash."""
        engine = IngestionEngine(storage, mock_llm, ingestion_config=ingestion_config)
        raw_content = "Unique content for hash test."

        await engine.commit(
            context_id=context_id,
            agent_id="test-agent",
            content=raw_content,
        )

        input_hash = hashlib.sha256(raw_content.encode()).hexdigest()
        found = await storage.get_raw_input_by_hash(str(context_id), input_hash)
        assert found is not None
        assert found.raw_input == raw_content


class TestDuplicateDetection:
    """Test 3: Duplicate raw inputs are detected and skipped."""

    async def test_exact_duplicate_returns_existing_batch(
        self, storage, mock_llm, ingestion_config, context_id,
    ):
        """Committing the exact same content twice returns the same batch ID."""
        engine = IngestionEngine(storage, mock_llm, ingestion_config=ingestion_config)
        content = "This is duplicate content for dedup testing."

        batch1 = await engine.commit(
            context_id=context_id,
            agent_id="agent-1",
            content=content,
        )
        batch2 = await engine.commit(
            context_id=context_id,
            agent_id="agent-2",
            content=content,
        )

        assert batch1.id == batch2.id

    async def test_different_content_not_deduped(
        self, storage, mock_llm, ingestion_config, context_id,
    ):
        """Different content produces different batches."""
        engine = IngestionEngine(storage, mock_llm, ingestion_config=ingestion_config)

        batch1 = await engine.commit(
            context_id=context_id,
            agent_id="agent-1",
            content="First unique content.",
        )
        batch2 = await engine.commit(
            context_id=context_id,
            agent_id="agent-1",
            content="Second unique content.",
        )

        assert batch1.id != batch2.id


class TestReExtractionPreview:
    """Test 4: Re-extraction dry_run returns accurate previews."""

    async def test_dry_run_returns_preview(
        self, storage, mock_llm, ingestion_config, context_id,
    ):
        """dry_run=True returns a ReExtractionPreview with correct counts."""
        engine = IngestionEngine(storage, mock_llm, ingestion_config=ingestion_config)
        await engine.commit(
            context_id=context_id,
            agent_id="test-agent",
            content="Some content that was originally processed.",
        )

        reflector = ReflectorEngine(mock_llm, config=ingestion_config)
        curator = CuratorEngine(storage, mock_llm)
        re_engine = ReExtractionEngine(reflector, curator, storage)

        request = ReExtractionRequest(
            reflector_model="new-better-model",
            dry_run=True,
        )
        result = await re_engine.re_extract(str(context_id), request)

        assert isinstance(result, ReExtractionPreview)
        assert result.activities_to_process >= 1
        assert result.estimated_input_tokens > 0

    async def test_empty_context_returns_empty_preview(
        self, storage, mock_llm, ingestion_config, context_id,
    ):
        """Re-extraction on context with no raw inputs returns empty preview."""
        reflector = ReflectorEngine(mock_llm, config=ingestion_config)
        curator = CuratorEngine(storage, mock_llm)
        re_engine = ReExtractionEngine(reflector, curator, storage)

        request = ReExtractionRequest(
            reflector_model="new-model",
            dry_run=True,
        )
        result = await re_engine.re_extract(str(context_id), request)

        assert isinstance(result, ReExtractionPreview)
        assert result.activities_to_process == 0


class TestReExtractionHighValuePreservation:
    """Test 5: Re-extraction preserves high-value bullets even if new model doesn't reproduce them."""

    async def test_high_value_bullets_preserved(
        self, storage, mock_llm, ingestion_config, context_id,
    ):
        """Bullets with high salience and hit_rate are kept even if re-extraction
        doesn't reproduce them. The reinforcement signal trumps model opinion."""
        ctx_id = str(context_id)

        # Create a high-value bullet manually
        high_value_bullet = Bullet(
            content="Critical finding: Always validate OCR output against ground truth",
            section="best_practices",
            bullet_type=BulletType.PRINCIPLE,
            salience=0.9,
            confidence=0.9,
            hit_count=10,
            recall_count=12,
        )
        high_value_bullet = await storage.add_bullet(ctx_id, high_value_bullet)

        # Create a low-value bullet
        low_value_bullet = Bullet(
            content="Tried using tesseract v3 briefly",
            section="notes",
            bullet_type=BulletType.FACT,
            salience=0.1,
            confidence=0.3,
            hit_count=0,
            recall_count=5,
        )
        low_value_bullet = await storage.add_bullet(ctx_id, low_value_bullet)

        # Mock curator's curate_re_extraction
        curator = CuratorEngine(storage, mock_llm)
        new_reflection = Reflection(
            new_insights=[
                ReflectionInsight(
                    content="A totally new insight from better model",
                    insight_type="fact",
                    suggested_section="general",
                    novelty=0.9,
                ),
            ],
        )

        batch = await curator.curate_re_extraction(
            new_reflection=new_reflection,
            old_bullets=[high_value_bullet, low_value_bullet],
            context_id=ctx_id,
        )

        # Check: high-value bullet should NOT have a REMOVE operation
        remove_ops = [
            op for op in batch.operations
            if op.op_type == DeltaOpType.REMOVE_BULLET
        ]
        removed_ids = [op.target_id for op in remove_ops]
        assert high_value_bullet.id not in removed_ids, (
            "High-value bullet should be preserved even if re-extraction doesn't reproduce it"
        )


class TestReExtractionLowValueRemoval:
    """Test 6: Re-extraction removes low-value bullets not reproduced by new model."""

    async def test_low_value_unmatched_bullets_removed(
        self, storage, mock_llm, ingestion_config, context_id,
    ):
        """Bullets with low salience and low hit_rate that aren't reproduced
        by the new model should be removed."""
        ctx_id = str(context_id)

        # Create a low-value bullet
        low_value_bullet = Bullet(
            content="Some unimportant note from old extraction",
            section="notes",
            bullet_type=BulletType.FACT,
            salience=0.1,
            confidence=0.3,
            hit_count=0,
            recall_count=5,
        )
        low_value_bullet = await storage.add_bullet(ctx_id, low_value_bullet)

        curator = CuratorEngine(storage, mock_llm)
        new_reflection = Reflection(
            new_insights=[
                ReflectionInsight(
                    content="Completely different insight from new model",
                    insight_type="fact",
                    suggested_section="general",
                    novelty=0.9,
                ),
            ],
        )

        batch = await curator.curate_re_extraction(
            new_reflection=new_reflection,
            old_bullets=[low_value_bullet],
            context_id=ctx_id,
        )

        # Low-value bullet should be removed
        remove_ops = [
            op for op in batch.operations
            if op.op_type == DeltaOpType.REMOVE_BULLET
        ]
        removed_ids = [op.target_id for op in remove_ops]
        assert low_value_bullet.id in removed_ids, (
            "Low-value bullet not reproduced by re-extraction should be removed"
        )


class TestExtractionMetadata:
    """Test 7: Activity records include extraction metadata."""

    async def test_activity_has_extraction_model(
        self, storage, mock_llm, ingestion_config, context_id,
    ):
        """Activity records include the extraction model used."""
        engine = IngestionEngine(storage, mock_llm, ingestion_config=ingestion_config)
        await engine.commit(
            context_id=context_id,
            agent_id="test-agent",
            content="Content for metadata test.",
            source_model="claude-sonnet-4-5",
        )

        activities = await storage.get_activities_with_raw_input(str(context_id))
        assert len(activities) >= 1

        activity = activities[0]
        assert activity.extraction_model == "test-canonical-model"
        assert activity.extraction_prompt_version == "v1-test"
        assert activity.source_agent_model == "claude-sonnet-4-5"

    async def test_activity_has_bullet_ids_produced(
        self, storage, mock_llm, ingestion_config, context_id,
    ):
        """Activity records track which bullet IDs were produced."""
        engine = IngestionEngine(storage, mock_llm, ingestion_config=ingestion_config)
        await engine.commit(
            context_id=context_id,
            agent_id="test-agent",
            content="Content that produces bullets.",
        )

        activities = await storage.get_activities_with_raw_input(str(context_id))
        assert len(activities) >= 1

        activity = activities[0]
        assert isinstance(activity.bullet_ids_produced, list)
        # The mock LLM produces 2 insights, so we should have bullet IDs
        assert len(activity.bullet_ids_produced) >= 1

    async def test_activity_has_content_type(
        self, storage, mock_llm, ingestion_config, context_id,
    ):
        """Activity records include the content type."""
        engine = IngestionEngine(storage, mock_llm, ingestion_config=ingestion_config)
        await engine.commit(
            context_id=context_id,
            agent_id="test-agent",
            content="Tool output from extraction pipeline.",
            content_type=ContentType.TOOL_OUTPUT,
        )

        activities = await storage.get_activities_with_raw_input(str(context_id))
        assert len(activities) >= 1
        assert activities[0].content_type == "tool_output"


class TestConfigIsolation:
    """Test 8: Changing server config does NOT retroactively affect existing bullets."""

    async def test_config_change_does_not_affect_existing_bullets(
        self, storage, mock_llm, context_id,
    ):
        """When the server config changes, existing bullets are untouched.
        Only new commits use the new config."""
        # Commit with original config
        config_v1 = IngestionConfig(
            reflector_model="model-v1",
            reflector_prompt_version="v1",
        )
        engine = IngestionEngine(storage, mock_llm, ingestion_config=config_v1)
        batch1 = await engine.commit(
            context_id=context_id,
            agent_id="test-agent",
            content="Content processed with model v1.",
        )

        # Count bullets after first commit
        bullets_after_v1 = await storage.list_bullets(str(context_id))
        bullets_v1_ids = {b.id for b in bullets_after_v1}

        # Change config — this simulates updating the server config
        config_v2 = IngestionConfig(
            reflector_model="model-v2",
            reflector_prompt_version="v2",
        )
        engine.ingestion_config = config_v2
        engine.reflector.config = config_v2

        # New commit uses new config
        batch2 = await engine.commit(
            context_id=context_id,
            agent_id="test-agent",
            content="New content processed with model v2.",
        )

        # Verify original bullets still exist unchanged
        for bid in bullets_v1_ids:
            bullet = await storage.get_bullet(bid)
            assert bullet is not None, f"Bullet {bid} from v1 should still exist"

        # Verify activities show different extraction models
        activities = await storage.get_activities_with_raw_input(str(context_id))
        extraction_models = [a.extraction_model for a in activities]
        assert "model-v1" in extraction_models
        assert "model-v2" in extraction_models


class TestIngestionConfigModel:
    """Unit tests for the IngestionConfig model itself."""

    def test_default_values(self):
        config = IngestionConfig()
        assert config.reflector_model == "claude-haiku-4-5"
        assert config.reflector_prompt_version == "v1"
        assert config.max_reflection_rounds == 2
        assert config.curator_dedup_threshold == 0.92
        assert config.embedding_model == "text-embedding-3-small"

    def test_custom_values(self):
        config = IngestionConfig(
            reflector_model="claude-sonnet-4-20250514",
            max_reflection_rounds=3,
        )
        assert config.reflector_model == "claude-sonnet-4-20250514"
        assert config.max_reflection_rounds == 3


class TestActivityComputeHash:
    """Test the Activity.compute_hash static method."""

    def test_compute_hash_deterministic(self):
        content = "Test content for hashing."
        h1 = Activity.compute_hash(content)
        h2 = Activity.compute_hash(content)
        assert h1 == h2

    def test_compute_hash_different_content(self):
        h1 = Activity.compute_hash("Content A")
        h2 = Activity.compute_hash("Content B")
        assert h1 != h2

    def test_compute_hash_matches_sha256(self):
        content = "Hello Engram"
        expected = hashlib.sha256(content.encode()).hexdigest()
        assert Activity.compute_hash(content) == expected


class TestStorageRawInputMethods:
    """Test the new storage methods for raw input queries."""

    async def test_get_activities_with_raw_input_filters(self, storage, context_id):
        """get_activities_with_raw_input only returns records with raw_input."""
        ctx_id = str(context_id)

        # Add an activity WITH raw input
        a1 = Activity(
            agent_id="agent1",
            action_type="fact_learned",
            summary="Has raw input",
            raw_input="Some raw content",
            raw_input_hash=Activity.compute_hash("Some raw content"),
        )
        await storage.add_activity(context_id, a1)

        # Add an activity WITHOUT raw input
        a2 = Activity(
            agent_id="agent2",
            action_type="fact_learned",
            summary="No raw input",
        )
        await storage.add_activity(context_id, a2)

        results = await storage.get_activities_with_raw_input(ctx_id)
        assert len(results) == 1
        assert results[0].raw_input == "Some raw content"

    async def test_get_raw_input_by_hash_returns_match(self, storage, context_id):
        """get_raw_input_by_hash returns matching activity."""
        ctx_id = str(context_id)
        content = "Content for hash lookup"
        content_hash = Activity.compute_hash(content)

        a = Activity(
            agent_id="agent1",
            action_type="fact_learned",
            summary="Test",
            raw_input=content,
            raw_input_hash=content_hash,
        )
        await storage.add_activity(context_id, a)

        found = await storage.get_raw_input_by_hash(ctx_id, content_hash)
        assert found is not None
        assert found.raw_input == content

    async def test_get_raw_input_by_hash_returns_none(self, storage, context_id):
        """get_raw_input_by_hash returns None for non-matching hash."""
        result = await storage.get_raw_input_by_hash(
            str(context_id), "nonexistent_hash"
        )
        assert result is None

    async def test_get_bullets_by_ids(self, storage, context_id):
        """get_bullets_by_ids returns matching bullets."""
        ctx_id = str(context_id)
        b1 = Bullet(content="Bullet A", bullet_type=BulletType.FACT)
        b2 = Bullet(content="Bullet B", bullet_type=BulletType.FACT)
        b1 = await storage.add_bullet(ctx_id, b1)
        b2 = await storage.add_bullet(ctx_id, b2)

        result = await storage.get_bullets_by_ids(ctx_id, [b1.id, b2.id])
        assert len(result) == 2
        result_ids = {b.id for b in result}
        assert b1.id in result_ids
        assert b2.id in result_ids

    async def test_get_bullets_by_ids_empty(self, storage, context_id):
        """get_bullets_by_ids with empty list returns empty list."""
        result = await storage.get_bullets_by_ids(str(context_id), [])
        assert result == []
