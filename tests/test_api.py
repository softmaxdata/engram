"""Tests for the FastAPI REST API (v0.3)."""

import uuid

import pytest
from fastapi.testclient import TestClient

from engram.server.app import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
def client(app):
    """Synchronous test client for basic route testing."""
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.5.0"


class TestContextEndpoints:
    def test_create_context(self, client):
        resp = client.post(
            "/contexts",
            json={
                "name": "Test Project",
                "description": "A test context",
                "owner": "tester",
                "intent": {
                    "objective": "Build something great",
                    "success_criteria": ["It works"],
                    "constraints": ["Under budget"],
                },
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Test Project"
        assert "id" in data

    def test_list_contexts(self, client):
        for name in ["Project A", "Project B"]:
            client.post(
                "/contexts",
                json={
                    "name": name,
                    "intent": {"objective": f"Objective for {name}"},
                },
            )
        resp = client.get("/contexts")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 2
        # v0.2: check bullet_count and schema_count in list
        assert "bullet_count" in data[0]
        assert "schema_count" in data[0]

    def test_get_context(self, client):
        create_resp = client.post(
            "/contexts",
            json={
                "name": "Fetch Me",
                "intent": {"objective": "Test retrieval"},
            },
        )
        ctx_id = create_resp.json()["id"]
        resp = client.get(f"/contexts/{ctx_id}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "Fetch Me"

    def test_get_context_not_found(self, client):
        fake_id = str(uuid.uuid4())
        resp = client.get(f"/contexts/{fake_id}")
        assert resp.status_code == 404


class TestConceptEndpoints:
    def _create_context(self, client) -> str:
        resp = client.post(
            "/contexts",
            json={
                "name": "Concept Test",
                "intent": {"objective": "Test concepts"},
            },
        )
        return resp.json()["id"]

    def test_add_concept(self, client):
        ctx_id = self._create_context(client)
        resp = client.post(
            f"/contexts/{ctx_id}/concepts",
            json={
                "type": "fact",
                "content": "The sky is blue",
                "salience": 0.7,
                "confidence": 0.99,
                "domain_tags": ["weather"],
            },
        )
        assert resp.status_code == 201
        assert "concept_id" in resp.json()

    def test_list_concepts(self, client):
        ctx_id = self._create_context(client)
        client.post(
            f"/contexts/{ctx_id}/concepts",
            json={"type": "fact", "content": "Fact 1"},
        )
        client.post(
            f"/contexts/{ctx_id}/concepts",
            json={"type": "decision", "content": "Decision 1"},
        )
        resp = client.get(f"/contexts/{ctx_id}/concepts")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_invalidate_concept(self, client):
        ctx_id = self._create_context(client)
        add_resp = client.post(
            f"/contexts/{ctx_id}/concepts",
            json={"type": "fact", "content": "Will be invalidated"},
        )
        concept_id = add_resp.json()["concept_id"]

        del_resp = client.delete(f"/contexts/{ctx_id}/concepts/{concept_id}")
        assert del_resp.status_code == 200
        assert del_resp.json()["status"] == "invalidated"


class TestBulletEndpoints:
    def _create_context(self, client) -> str:
        resp = client.post(
            "/contexts",
            json={
                "name": "Bullet Test",
                "intent": {"objective": "Test bullets"},
            },
        )
        return resp.json()["id"]

    def test_add_bullet(self, client):
        ctx_id = self._create_context(client)
        resp = client.post(
            f"/contexts/{ctx_id}/bullets",
            json={
                "content": "PaddleOCR is fast",
                "section": "ocr",
                "bullet_type": "fact",
                "salience": 0.8,
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "bullet_id" in data
        assert "delta_batch_id" in data

    def test_list_bullets(self, client):
        ctx_id = self._create_context(client)
        for content in ["Fact 1", "Fact 2", "Fact 3"]:
            client.post(
                f"/contexts/{ctx_id}/bullets",
                json={"content": content, "section": "general"},
            )
        resp = client.get(f"/contexts/{ctx_id}/bullets")
        assert resp.status_code == 200
        assert len(resp.json()) == 3

    def test_get_bullet(self, client):
        ctx_id = self._create_context(client)
        add_resp = client.post(
            f"/contexts/{ctx_id}/bullets",
            json={"content": "Retrieve me", "section": "test"},
        )
        bullet_id = add_resp.json()["bullet_id"]

        resp = client.get(f"/contexts/{ctx_id}/bullets/{bullet_id}")
        assert resp.status_code == 200
        assert resp.json()["content"] == "Retrieve me"

    def test_list_bullets_by_section(self, client):
        ctx_id = self._create_context(client)
        client.post(f"/contexts/{ctx_id}/bullets", json={"content": "OCR fact", "section": "ocr"})
        client.post(f"/contexts/{ctx_id}/bullets", json={"content": "API fact", "section": "api"})

        resp = client.get(f"/contexts/{ctx_id}/bullets", params={"section": "ocr"})
        assert resp.status_code == 200
        assert len(resp.json()) == 1


class TestDeltaEndpoints:
    def _create_context_with_bullets(self, client) -> str:
        ctx_id = client.post(
            "/contexts",
            json={"name": "Delta Test", "intent": {"objective": "Test deltas"}},
        ).json()["id"]
        # Adding a bullet creates a delta batch
        client.post(
            f"/contexts/{ctx_id}/bullets",
            json={"content": "Creates a delta", "section": "test"},
        )
        return ctx_id

    def test_list_deltas(self, client):
        ctx_id = self._create_context_with_bullets(client)
        resp = client.get(f"/contexts/{ctx_id}/deltas")
        assert resp.status_code == 200
        assert len(resp.json()) >= 1

    def test_get_delta(self, client):
        ctx_id = self._create_context_with_bullets(client)
        deltas = client.get(f"/contexts/{ctx_id}/deltas").json()
        delta_id = deltas[0]["id"]

        resp = client.get(f"/contexts/{ctx_id}/deltas/{delta_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == delta_id

    def test_rollback_delta(self, client):
        ctx_id = self._create_context_with_bullets(client)
        deltas = client.get(f"/contexts/{ctx_id}/deltas").json()
        delta_id = deltas[0]["id"]

        resp = client.post(f"/contexts/{ctx_id}/deltas/{delta_id}/rollback")
        assert resp.status_code == 200
        assert resp.json()["status"] == "rolled_back"


class TestSchemaEndpoints:
    def test_list_schemas_empty(self, client):
        ctx_id = client.post(
            "/contexts",
            json={"name": "Schema Test", "intent": {"objective": "Test schemas"}},
        ).json()["id"]

        resp = client.get(f"/contexts/{ctx_id}/schemas")
        assert resp.status_code == 200
        assert resp.json() == []


class TestHealthEndpointForContext:
    def test_context_health(self, client):
        ctx_id = client.post(
            "/contexts",
            json={"name": "Health Test", "intent": {"objective": "Test health"}},
        ).json()["id"]

        # Add some bullets
        for i in range(3):
            client.post(
                f"/contexts/{ctx_id}/bullets",
                json={"content": f"Fact {i}", "section": "general"},
            )

        resp = client.get(f"/contexts/{ctx_id}/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_bullets"] == 3
        assert data["active_bullets"] == 3
        assert data["archived_bullets"] == 0
        assert data["schema_count"] == 0
        assert "avg_salience" in data
        assert "avg_effective_salience" in data
        assert "top_sections" in data


class TestConsolidateEndpoint:
    def test_consolidate_empty_context(self, client):
        ctx_id = client.post(
            "/contexts",
            json={"name": "Consolidate Test", "intent": {"objective": "Test consolidation"}},
        ).json()["id"]

        resp = client.post(
            f"/contexts/{ctx_id}/consolidate",
            json={},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "decayed" in data
        assert "deduplicated" in data
        assert "schemas_formed" in data
        assert "archived" in data
        assert "promoted" in data
        assert data["duration_ms"] >= 0


class TestMaterializeEndpoints:
    def _create_context_with_concepts(self, client) -> str:
        ctx_id = client.post(
            "/contexts",
            json={
                "name": "Materialize Test",
                "intent": {
                    "objective": "Test materialization",
                    "success_criteria": ["Renders correctly"],
                },
            },
        ).json()["id"]

        for content in ["OCR is fast", "Use PaddleOCR", "Client needs HIPAA"]:
            client.post(
                f"/contexts/{ctx_id}/concepts",
                json={"type": "fact", "content": content},
            )
        return ctx_id

    def test_materialize(self, client):
        ctx_id = self._create_context_with_concepts(client)
        resp = client.post(
            f"/contexts/{ctx_id}/materialize",
            json={"token_budget": 4000, "target_model": "claude"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "rendered_text" in data
        assert "materialization_id" in data
        assert data["token_count"] > 0

    def test_materialize_gpt(self, client):
        ctx_id = self._create_context_with_concepts(client)
        resp = client.post(
            f"/contexts/{ctx_id}/materialize",
            json={"token_budget": 4000, "target_model": "gpt-4o"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "# Project Context" in data["rendered_text"]

    def test_recall(self, client):
        ctx_id = self._create_context_with_concepts(client)
        resp = client.post(
            f"/contexts/{ctx_id}/recall",
            json={"query": "What OCR tool?", "token_budget": 2000, "target_model": "claude"},
        )
        assert resp.status_code == 200
        assert "context" in resp.json()

    def test_materialize_not_found(self, client):
        fake_id = str(uuid.uuid4())
        resp = client.post(
            f"/contexts/{fake_id}/materialize",
            json={"token_budget": 1000},
        )
        assert resp.status_code == 404


class TestLifecycleEndpoints:
    """v0.3 lifecycle management endpoints."""

    def _create_context_with_bullet(self, client):
        """Helper: create context + add a bullet."""
        resp = client.post(
            "/contexts",
            json={
                "name": "Lifecycle Test",
                "intent": {"objective": "Test lifecycle"},
            },
        )
        ctx_id = resp.json()["id"]
        resp = client.post(
            f"/contexts/{ctx_id}/bullets",
            json={"content": "Test bullet for lifecycle", "section": "general"},
        )
        bullet_id = resp.json()["bullet_id"]
        return ctx_id, bullet_id

    def test_get_lifecycle(self, client):
        resp = client.post(
            "/contexts",
            json={
                "name": "Lifecycle Status Test",
                "intent": {"objective": "Test"},
            },
        )
        ctx_id = resp.json()["id"]
        resp = client.get(f"/contexts/{ctx_id}/lifecycle")
        assert resp.status_code == 200
        data = resp.json()
        assert "capacity" in data
        assert "lifecycle_config" in data
        assert data["capacity"]["pressure_level"] == "normal"

    def test_archive_and_restore_bullet(self, client):
        ctx_id, bullet_id = self._create_context_with_bullet(client)

        # Archive
        resp = client.post(
            f"/contexts/{ctx_id}/bullets/{bullet_id}/archive",
            json={"reason": "test_archive"},
        )
        assert resp.status_code == 200
        assert resp.json()["archived"] is True

        # List archived
        resp = client.get(f"/contexts/{ctx_id}/archived-bullets")
        assert resp.status_code == 200
        archived = resp.json()
        assert len(archived) >= 1
        assert any(b["id"] == bullet_id for b in archived)

        # Restore
        resp = client.post(f"/contexts/{ctx_id}/bullets/{bullet_id}/restore")
        assert resp.status_code == 200
        assert resp.json()["restored"] is True

    def test_archive_nonexistent_bullet(self, client):
        resp = client.post(
            "/contexts",
            json={
                "name": "Archive Test",
                "intent": {"objective": "Test"},
            },
        )
        ctx_id = resp.json()["id"]
        resp = client.post(
            f"/contexts/{ctx_id}/bullets/nonexistent/archive",
            json={"reason": "test"},
        )
        assert resp.status_code == 404

    def test_purge_context(self, client):
        resp = client.post(
            "/contexts",
            json={
                "name": "Purge Test",
                "intent": {"objective": "Test purge"},
            },
        )
        ctx_id = resp.json()["id"]

        resp = client.delete(f"/contexts/{ctx_id}/purge")
        assert resp.status_code == 200
        assert resp.json()["purged"] is True

        # Context should be gone
        resp = client.get(f"/contexts/{ctx_id}")
        assert resp.status_code == 404

    def test_purge_user(self, client):
        resp = client.post(
            "/contexts",
            json={
                "name": "User Purge Test",
                "owner": "test-user-purge",
                "intent": {"objective": "Test user purge"},
            },
        )
        assert resp.status_code == 201

        resp = client.delete("/users/test-user-purge/purge")
        assert resp.status_code == 200
        assert resp.json()["purged_contexts"] >= 1

    def test_health_includes_capacity(self, client):
        ctx_id, _ = self._create_context_with_bullet(client)
        resp = client.get(f"/contexts/{ctx_id}/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "capacity" in data
        assert data["capacity"]["active_bullet_count"] >= 1

    def test_sync_endpoint(self, client):
        ctx_id, _ = self._create_context_with_bullet(client)
        resp = client.post(f"/contexts/{ctx_id}/sync")
        assert resp.status_code == 200
        data = resp.json()
        assert "delta_batches" in data
