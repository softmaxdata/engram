"""Regression tests for supported REST request shapes and server resource cleanup."""

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from engram.core.models import Context, DeltaBatch, IntentAnchor
from engram.server.routes.lifecycle import router
from engram.storage.sqlite import SQLiteBackend


@pytest.mark.parametrize(
    "since", ["2020-01-01T00:00:00", "2020-01-01T00:00:00Z", "2019-12-31T19:00:00-05:00"]
)
async def test_sync_accepts_iso_timestamps_with_and_without_timezone(tmp_path, since):
    storage = SQLiteBackend(str(tmp_path / "sync.db"))
    app = FastAPI()
    app.state.storage = storage
    app.include_router(router, prefix="/contexts")

    @app.middleware("http")
    async def local_user(request, call_next):
        request.state.user_id = "default"
        return await call_next(request)

    try:
        await storage.initialize()
        context = await storage.create_context(
            Context(name="Sync", intent=IntentAnchor(objective="Test"))
        )
        batch = DeltaBatch(context_id=str(context.id), timestamp=datetime(2021, 1, 1, tzinfo=UTC))
        await storage.save_delta_batch(batch)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            response = await client.post(f"/contexts/{context.id}/sync", params={"since": since})
        assert response.status_code == 200, response.text
        assert [item["id"] for item in response.json()["delta_batches"]] == [batch.id]
    finally:
        await storage.close()


async def test_sync_invalid_timestamp_returns_existing_400(tmp_path):
    storage = SQLiteBackend(str(tmp_path / "sync.db"))
    app = FastAPI()
    app.state.storage = storage
    app.include_router(router, prefix="/contexts")

    @app.middleware("http")
    async def local_user(request, call_next):
        request.state.user_id = "default"
        return await call_next(request)

    try:
        await storage.initialize()
        context = await storage.create_context(
            Context(name="Sync", intent=IntentAnchor(objective="Test"))
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            response = await client.post(
                f"/contexts/{context.id}/sync", params={"since": "invalid"}
            )
        assert response.status_code == 400
    finally:
        await storage.close()


@pytest.mark.parametrize("body", [{"max_reflection_rounds": "not-a-number"}, [], None])
async def test_invalid_ingestion_config_returns_validation_error_without_mutation(
    body, monkeypatch
):
    from engram.core.config import IngestionConfig, Settings
    from engram.server.routes.lifecycle import config_router

    monkeypatch.setattr("engram.core.config.get_settings", lambda: Settings(_env_file=None))
    app = FastAPI()
    app.state.ingestion_config = IngestionConfig()
    before = app.state.ingestion_config.model_dump()
    app.include_router(config_router)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        response = await client.put(
            "/config/ingestion",
            content=json.dumps(body),
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == 422, response.text
    assert app.state.ingestion_config.model_dump() == before


async def test_ingestion_config_updates_future_commit_and_reextraction_engines(monkeypatch):
    from engram.core.config import IngestionConfig, Settings
    from engram.server.routes.lifecycle import config_router

    monkeypatch.setattr("engram.core.config.get_settings", lambda: Settings(_env_file=None))
    app = FastAPI()
    config = IngestionConfig()
    app.state.ingestion_config = config
    app.state.ingestion = SimpleNamespace(
        ingestion_config=config,
        reflector=SimpleNamespace(config=config),
        curator=SimpleNamespace(ingestion_config=config),
    )
    app.state.re_extraction = SimpleNamespace(
        reflector=SimpleNamespace(config=config),
        curator=SimpleNamespace(ingestion_config=config),
    )
    app.include_router(config_router)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.put("/config/ingestion", json={"curator_dedup_threshold": 0.8})
    assert response.status_code == 200
    updated = app.state.ingestion_config
    assert updated.curator_dedup_threshold == 0.8
    assert app.state.ingestion.ingestion_config is updated
    assert app.state.ingestion.reflector.config is updated
    assert app.state.ingestion.curator.ingestion_config is updated
    assert app.state.re_extraction.reflector.config is updated
    assert app.state.re_extraction.curator.ingestion_config is updated
