"""Exercise legacy public SDK methods against real REST routes and SQLite."""

import httpx
import pytest

from engram.core.config import Settings
from engram.sdk.client import Engram
from engram.server import app as app_module
from engram.storage.sqlite import SQLiteBackend


class OfflineProvider:
    async def embed(self, text):
        return [1.0] + [0.0] * 1535

    async def complete(self, *args, **kwargs):
        return '{"new_insights": [], "confidence": 1.0}'


@pytest.mark.parametrize("id_form", ["uuid", "string"])
async def test_legacy_sdk_roundtrip_preserves_request_and_response_contracts(
    tmp_path, monkeypatch, id_form
):
    settings = Settings(_env_file=None, auth_enabled=False)
    monkeypatch.setattr(app_module, "get_settings", lambda: settings)
    monkeypatch.setattr("engram.core.config.get_settings", lambda: settings)
    monkeypatch.setattr(
        app_module, "_create_storage", lambda: SQLiteBackend(str(tmp_path / "sdk.db"))
    )
    monkeypatch.setattr(app_module, "_create_llm", OfflineProvider)
    app = app_module.create_app()
    async with app.router.lifespan_context(app):
        async with Engram(url="http://test") as sdk:
            await sdk._client.aclose()
            sdk._client = httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            )
            context = await sdk.create_context(
                "SDK compatibility", {"objective": "Keep supported clients working"}
            )
            context_id = context.id if id_form == "uuid" else str(context.id)
            assert (await sdk.get_context(context_id)).id == context.id
            assert str(context.id) in {item["id"] for item in await sdk.list_contexts()}
            added = await sdk.add_bullet(context_id, "The existing SDK contract remains supported.")
            bullet_id = added["bullet_id"]
            assert (await sdk.get_bullet(context_id, bullet_id))["content"].startswith(
                "The existing SDK"
            )
            assert bullet_id in {item["id"] for item in await sdk.list_bullets(context_id)}
            decision = await sdk.record_decision(
                context_id, "Keep API field names", "Preserve existing integrations"
            )
            assert (await sdk.get_bullet(context_id, decision["concept_id"]))[
                "bullet_type"
            ] == "decision"
            materialized = await sdk.materialize(
                context_id, query="existing SDK", token_budget=2000
            )
            assert isinstance(materialized["rendered_text"], str)
            assert materialized["materialization_id"]
            assert bullet_id in materialized["bullets_included"]
            assert isinstance(await sdk.recall(context_id, "existing SDK"), str)
            committed = await sdk.commit(
                context_id, "legacy-sdk", "No new insights in this fixture."
            )
            assert {"delta_batch_id", "activity_id", "bullets_added"} <= committed.keys()
            assert (await sdk.archive_bullet(context_id, bullet_id))["archived"] is True
            assert bullet_id in {item["id"] for item in await sdk.list_archived_bullets(context_id)}
            assert (await sdk.restore_bullet(context_id, bullet_id))["restored"] is True
            assert "capacity" in await sdk.get_lifecycle(context_id)
            assert "delta_batches" in await sdk.sync(context_id, since="2020-01-01T00:00:00")
            assert isinstance(await sdk.get_activity(context_id), list)
            assert "reflector_model" in await sdk.get_ingestion_config()
            assert (await sdk.update_ingestion_config(curator_dedup_threshold=0.9))[
                "updated"
            ] is True
            assert (await sdk.purge_context(context_id))["purged"] is True
