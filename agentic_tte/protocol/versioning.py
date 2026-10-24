"""Protocol versioning: Rehearse and Redesign make child versions, diffed against the parent."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from agentic_tte.protocol.models import Provenance, TargetTrialProtocol

_IGNORED = {"version", "parent_version", "status", "provenance"}


class FieldChange(BaseModel):
    path: str
    before: Any = None
    after: Any = None


def new_version(
    parent: TargetTrialProtocol, *, created_by: str, notes: str | None = None, **changes: Any
) -> TargetTrialProtocol:
    """Return a child protocol with ``changes`` applied; the parent is not modified."""
    data = parent.model_dump()
    data.update(changes)
    data.update(
        version=parent.version + 1,
        parent_version=parent.version,
        status="draft",
        provenance=Provenance(created_by=created_by, notes=notes).model_dump(),
    )
    return TargetTrialProtocol.model_validate(data)


def diff(a: TargetTrialProtocol, b: TargetTrialProtocol) -> list[FieldChange]:
    """Field-level changes from ``a`` to ``b``. List items are keyed by ``id``/``label``/``name``
    where available, so reordering criteria does not show up as a change."""
    changes: list[FieldChange] = []
    _walk(
        a.model_dump(mode="json", exclude=_IGNORED),
        b.model_dump(mode="json", exclude=_IGNORED),
        "",
        changes,
    )
    return changes


def _key(item: Any, index: int) -> str:
    if isinstance(item, dict):
        for k in ("id", "label", "name"):
            if item.get(k) is not None:
                return f"{k}={item[k]}"
    return str(index)


def _walk(a: Any, b: Any, path: str, out: list[FieldChange]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            _walk(a.get(k), b.get(k), f"{path}.{k}" if path else k, out)
    elif isinstance(a, list) and isinstance(b, list):
        ka = {_key(x, i): x for i, x in enumerate(a)}
        kb = {_key(x, i): x for i, x in enumerate(b)}
        for k in list(ka) + [k for k in kb if k not in ka]:
            _walk(ka.get(k), kb.get(k), f"{path}[{k}]", out)
    elif a != b:
        out.append(FieldChange(path=path, before=a, after=b))
