"""Schema mutations are reversible without guessing historical state."""

import pytest
from test_storage_transactions import storage as transaction_storage

from engram.core.delta import DeltaEngine
from engram.core.models import (
    Bullet,
    Context,
    DeltaBatch,
    DeltaOperation,
    DeltaOpType,
    IntentAnchor,
    SchemaNode,
)

storage = transaction_storage


@pytest.mark.parametrize("explicit_id", [None, "chosen-schema"])
async def test_added_schema_persists_target_and_rolls_back_without_dangling_links(
    storage, explicit_id
):
    backend, context_id = storage
    engine = DeltaEngine(backend)
    batch = await engine.apply_batch(
        DeltaBatch(
            context_id=context_id,
            operations=[
                DeltaOperation(
                    op_type=DeltaOpType.ADD_SCHEMA,
                    target_id=explicit_id,
                    content="new schema",
                    reasoning="description",
                ),
            ],
        )
    )
    schema_id = batch.operations[0].target_id
    assert schema_id is not None
    if explicit_id:
        assert schema_id == explicit_id
    assert (await backend.get_schema(schema_id)).description == "description"
    assert (await backend.get_delta_batch(batch.id)).operations[0].target_id == schema_id
    await backend.add_bullet(
        context_id, Bullet(id="linked", content="retained memory", schema_id=schema_id)
    )
    assert await engine.rollback_batch(batch.id)
    assert await backend.get_schema(schema_id) is None
    assert (await backend.get_bullet("linked")).schema_id is None
    assert (await backend.get_bullet("linked")).content == "retained memory"


async def test_schema_update_restores_description_without_changing_other_fields(storage):
    backend, context_id = storage
    before = SchemaNode(id="existing", name="name", description="before", instance_count=3)
    await backend.add_schema(context_id, before)
    engine = DeltaEngine(backend)
    batch = await engine.apply_batch(
        DeltaBatch(
            context_id=context_id,
            operations=[
                DeltaOperation(
                    op_type=DeltaOpType.UPDATE_SCHEMA, target_id=before.id, content="after"
                ),
            ],
        )
    )
    assert (await backend.get_schema(before.id)).description == "after"
    assert await engine.rollback_batch(batch.id)
    restored = await backend.get_schema(before.id)
    assert (restored.description, restored.name, restored.instance_count) == ("before", "name", 3)


@pytest.mark.parametrize("kind", [DeltaOpType.ADD_SCHEMA, DeltaOpType.UPDATE_SCHEMA])
async def test_legacy_schema_without_snapshot_declines_entire_rollback(storage, kind):
    backend, context_id = storage
    # Old ADD_SCHEMA ignored caller target_id and generated an unrecorded ID.
    # This target may therefore identify an unrelated, pre-existing schema.
    await backend.add_schema(
        context_id, SchemaNode(id="unrelated", name="keep", description="keep")
    )
    await backend.add_bullet(context_id, Bullet(id="keep-bullet", content="keep"))
    batch = DeltaBatch(
        context_id=context_id,
        operations=[
            DeltaOperation(op_type=kind, target_id="unrelated", content="legacy unknown state"),
            DeltaOperation(op_type=DeltaOpType.ADD_BULLET, target_id="keep-bullet", content="keep"),
        ],
    )
    await backend.save_delta_batch(batch)
    assert not await DeltaEngine(backend).rollback_batch(batch.id)
    assert (await backend.get_bullet("keep-bullet")).is_active
    assert (await backend.get_schema("unrelated")).description == "keep"


async def test_schema_rollback_rejects_foreign_target_atomically(storage):
    backend, context_id = storage
    foreign = await backend.create_context(
        Context(name="foreign", intent=IntentAnchor(objective="private"))
    )
    await backend.add_schema(
        str(foreign.id), SchemaNode(id="foreign-schema", name="private", description="private")
    )
    await backend.add_bullet(context_id, Bullet(id="keep", content="keep"))
    op = DeltaOperation(
        op_type=DeltaOpType.ADD_SCHEMA,
        target_id="foreign-schema",
        content="private",
        previous_state={"schema_created": True},
    )
    batch = DeltaBatch(
        context_id=context_id,
        operations=[
            op,
            DeltaOperation(op_type=DeltaOpType.ADD_BULLET, target_id="keep", content="keep"),
        ],
    )
    await backend.save_delta_batch(batch)
    with pytest.raises(ValueError, match="context"):
        await DeltaEngine(backend).rollback_batch(batch.id)
    assert (await backend.get_bullet("keep")).is_active
    assert await backend.get_schema("foreign-schema") is not None


async def test_schema_rollback_failure_restores_schema_and_bullet_links(storage, monkeypatch):
    backend, context_id = storage
    engine = DeltaEngine(backend)
    await backend.add_schema(context_id, SchemaNode(id="old", name="old", description="before"))
    batch = await engine.apply_batch(
        DeltaBatch(
            context_id=context_id,
            operations=[
                DeltaOperation(op_type=DeltaOpType.UPDATE_SCHEMA, target_id="old", content="after"),
                DeltaOperation(op_type=DeltaOpType.ADD_SCHEMA, content="new"),
            ],
        )
    )
    # Locate by name, independent of ordering/legacy missing delta target.
    new_id = next(s.id for s in await backend.list_schemas(context_id) if s.name == "new")
    await backend.add_bullet(context_id, Bullet(id="linked", content="keep", schema_id=new_id))

    async def fail(self, schema):
        raise RuntimeError("schema update unavailable")

    monkeypatch.setattr(type(backend), "update_schema", fail)
    with pytest.raises(RuntimeError, match="schema update"):
        await engine.rollback_batch(batch.id)
    assert await backend.get_schema(new_id) is not None
    assert (await backend.get_bullet("linked")).schema_id == new_id
    assert (await backend.get_schema("old")).description == "after"
