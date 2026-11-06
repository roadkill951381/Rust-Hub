"""Progress reporting with cancellation, shared by the pipeline and backends."""

from __future__ import annotations

from collections.abc import Callable


class Cancelled(Exception):
    """Raised inside a job when its client cancelled it."""


#: ``report(fraction, stage)``; raises :class:`Cancelled` when the job is cancelled.
Report = Callable[[float, str], None]


def span(report: Report, start: float, end: float) -> Report:
    """A reporter that maps 0..1 onto ``start..end`` of ``report``."""

    def sub(fraction: float, stage: str) -> None:
        report(start + (end - start) * min(max(fraction, 0.0), 1.0), stage)

    return sub
