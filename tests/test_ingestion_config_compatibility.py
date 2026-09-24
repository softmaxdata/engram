"""Configured ingestion behavior reaches providers without breaking old adapters."""

from unittest.mock import AsyncMock

import pytest
from test_storage_transactions import storage as transaction_storage

from engram.core.config import IngestionConfig
from engram.core.ingestion import CuratorEngine, ReflectorEngine
from engram.core.models import Bullet, DeltaOpType, Reflection, ReflectionInsight

storage = transaction_storage


@pytest.mark.parametrize("override", [None, "explicit-model"])
async def test_reflector_routes_configured_and_explicit_models(override):
    llm = AsyncMock()
    llm.complete.return_value = "{}"
    config = IngestionConfig(reflector_model="configured-model")
    await ReflectorEngine(llm, config).reflect("raw input", model_override=override)
    assert llm.complete.await_args.kwargs["model"] == (override or "configured-model")


async def test_legacy_adapter_completion_signature_still_works():
    class LegacyAdapter:
        async def complete(self, prompt, system=None, temperature=0, response_format=None):
            return '{"confidence": 0.75}'

    result = await ReflectorEngine(LegacyAdapter()).reflect("legacy input")
    assert result.confidence == 0.75


@pytest.mark.parametrize(
    "threshold,expected", [(0.85, DeltaOpType.UPDATE_BULLET), (0.95, DeltaOpType.ADD_BULLET)]
)
async def test_configured_curator_threshold_changes_actual_dedup(storage, threshold, expected):
    backend, context_id = storage
    await backend.add_bullet(
        context_id, Bullet(id="stored", content="stored knowledge", embedding=[1.0] + [0.0] * 1535)
    )
    llm = AsyncMock()
    llm.embed.return_value = [0.9, 0.19**0.5] + [0.0] * 1534
    config = IngestionConfig(curator_dedup_threshold=threshold)
    if hasattr(config, "validity_gate_enabled"):
        config.validity_gate_enabled = False
    curator = CuratorEngine(backend, llm, ingestion_config=config)
    result = await curator.curate(
        context_id,
        Reflection(
            new_insights=[
                ReflectionInsight(content="a longer paraphrase of stored knowledge", novelty=0.9),
            ]
        ),
    )
    assert len(result.operations) == 1
    assert result.operations[0].op_type == expected


async def test_per_call_model_does_not_change_adapter_default(monkeypatch):
    import sys
    from types import SimpleNamespace

    from engram.llm.adapter import LiteLLMAdapter, complete_with_model

    completion = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))]
        )
    )
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(acompletion=completion))
    adapter = LiteLLMAdapter(model="adapter-default")
    await complete_with_model(adapter, model="one-call-only", prompt="first")
    await adapter.complete(prompt="second")
    assert [call.kwargs["model"] for call in completion.await_args_list] == [
        "one-call-only",
        "adapter-default",
    ]
    assert adapter.model == "adapter-default"


async def test_provider_type_error_is_not_retried_as_legacy_adapter():
    from engram.llm.adapter import complete_with_model

    llm = AsyncMock()
    llm.complete.side_effect = TypeError("provider response malformed")
    with pytest.raises(TypeError, match="provider response malformed"):
        await complete_with_model(llm, model="configured", prompt="input")
    llm.complete.assert_awaited_once()
