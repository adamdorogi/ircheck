import asyncio
import importlib
import inspect
from collections.abc import Callable, Iterable, Mapping
from typing import cast

from ..config import PythonTarget
from .http import normalize_ids


class PythonAdapter:
    def __init__(self, target: PythonTarget) -> None:
        module, _, dotted_name = target.callable.partition(":")
        try:
            callback: object = importlib.import_module(module)
            for name in dotted_name.split("."):
                callback = getattr(callback, name)
        except Exception as error:
            raise ValueError(f"cannot import {target.callable!r}: {error}") from error
        if not callable(callback):
            raise ValueError(f"{target.callable!r} is not callable")
        self.callback = cast(Callable[[str, int], object], callback)

    async def search(self, query: str, k: int) -> list[str]:
        if inspect.iscoroutinefunction(self.callback):
            result = await self.callback(query, k)
        else:
            result = await asyncio.to_thread(self.callback, query, k)
            if inspect.isawaitable(result):
                result = await result
        if not isinstance(result, Iterable) or isinstance(result, (str, bytes, Mapping)):
            raise ValueError("Python target must return an iterable of document IDs")
        return normalize_ids(result)
