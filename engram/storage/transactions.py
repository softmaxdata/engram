"""Explicit, task-owned transaction handles shared by the storage backends."""

from __future__ import annotations

import asyncio
import functools
import inspect
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any


@dataclass
class TransactionScope:
    context_id: str
    owner: asyncio.Task
    active: bool = True

    def check(self) -> None:
        if not self.active:
            raise RuntimeError("Transaction handle has expired")
        if self.owner is not asyncio.current_task():
            raise RuntimeError("Transaction handle belongs to another task")


class TaskGate:
    """A reentrant task-owned gate; ownership is never inherited by child tasks."""

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.owner: asyncio.Task | None = None

    @asynccontextmanager
    async def hold(self):
        task = asyncio.current_task()
        if self.owner is task:
            yield
            return
        async with self.lock:
            self.owner = task
            try:
                yield
            finally:
                self.owner = None


def guarded_backend(cls):
    """Guard every public coroutine, including edition-specific auth methods.

    Transactions themselves are async context managers and implement their own
    guard. Private SQL helpers run only inside a guarded public operation.
    """
    for name, method in tuple(vars(cls).items()):
        if name.startswith("_") or not inspect.iscoroutinefunction(method):
            continue

        def wrap(fn, method_name):
            @functools.wraps(fn)
            async def guarded(self, *args, **kwargs):
                async with self._operation(method_name):
                    return await fn(self, *args, **kwargs)

            return guarded

        setattr(cls, name, wrap(method, name))
    return cls


async def finish(awaitable):
    """Finish connection cleanup before releasing ownership, even if cancelled."""
    task = asyncio.ensure_future(awaitable)
    cancellation = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancellation = exc
    result = task.result()
    if cancellation is not None:
        raise cancellation
    return result


class SQLiteTransactionConnection:
    def __init__(self, connection, scope: TransactionScope):
        self.connection = connection
        self.scope = scope

    async def execute(self, *args, **kwargs):
        self.scope.check()
        return await self.connection.execute(*args, **kwargs)

    async def executemany(self, *args, **kwargs):
        self.scope.check()
        return await self.connection.executemany(*args, **kwargs)

    async def commit(self):
        self.scope.check()  # only the outer transaction may commit


class PostgresTransactionConnection:
    """The pool-shaped executor expected by existing PostgreSQL methods."""

    def __init__(self, connection, scope: TransactionScope):
        self.connection = connection
        self.scope = scope

    async def execute(self, *args, **kwargs):
        self.scope.check()
        return await self.connection.execute(*args, **kwargs)

    async def executemany(self, *args, **kwargs):
        self.scope.check()
        return await self.connection.executemany(*args, **kwargs)

    async def fetch(self, *args, **kwargs):
        self.scope.check()
        return await self.connection.fetch(*args, **kwargs)

    async def fetchrow(self, *args, **kwargs):
        self.scope.check()
        return await self.connection.fetchrow(*args, **kwargs)

    async def fetchval(self, *args, **kwargs):
        self.scope.check()
        return await self.connection.fetchval(*args, **kwargs)

    @asynccontextmanager
    async def acquire(self):
        self.scope.check()
        yield self

    def transaction(self, **kwargs: Any):
        self.scope.check()
        return self.connection.transaction(**kwargs)
