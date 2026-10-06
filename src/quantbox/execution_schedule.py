"""Moved to :mod:`quantbox.engine.schedule` — the scheduled book is part of the engine seam (docs/adr/0008).

Kept so ``from quantbox.execution_schedule import schedule_book`` still works.
"""

from quantbox.engine.schedule import ScheduledBook, schedule_book

__all__ = ["ScheduledBook", "schedule_book"]
