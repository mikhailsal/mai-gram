"""Summary file store for daily, weekly, and monthly memory layers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


@dataclass(frozen=True)
class StoredSummary:
    """One stored summary document."""

    summary_type: str
    period: str
    content: str


class SummaryStore:
    """File-based summary storage."""

    def __init__(self, data_dir: str | Path = "./data") -> None:
        self._data_dir = Path(data_dir)

    @property
    def data_dir(self) -> Path:
        """Expose the configured base data directory."""
        return self._data_dir

    def save_daily(self, companion_id: str, target_date: date, content: str) -> Path:
        """Save a daily summary."""
        filename = f"{target_date.isoformat()}.md"
        return self._write_file(self._summary_dir(companion_id, "daily"), filename, content)

    def save_weekly(self, companion_id: str, period: str, content: str) -> Path:
        """Save a weekly summary with period format 'YYYY-Www'."""
        return self._write_file(self._summary_dir(companion_id, "weekly"), f"{period}.md", content)

    def save_monthly(self, companion_id: str, period: str, content: str) -> Path:
        """Save a monthly summary with period format 'YYYY-MM'."""
        return self._write_file(self._summary_dir(companion_id, "monthly"), f"{period}.md", content)

    def get_all_summaries(self, companion_id: str) -> list[StoredSummary]:
        """Return all summaries in monthly -> weekly -> daily chronological order."""
        monthly = self._load_summaries(companion_id, "monthly")
        weekly = self._load_summaries(companion_id, "weekly")
        daily = self._load_summaries(companion_id, "daily")
        return [*monthly, *weekly, *daily]

    def delete_daily(self, companion_id: str, target_date: date) -> bool:
        """Delete a daily summary file."""
        file_path = self._summary_dir(companion_id, "daily") / f"{target_date.isoformat()}.md"
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def delete_weekly(self, companion_id: str, period: str) -> bool:
        """Delete a weekly summary file."""
        file_path = self._summary_dir(companion_id, "weekly") / f"{period}.md"
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def list_dailies(self, companion_id: str) -> list[date]:
        """List all daily summary dates in ascending order."""
        directory = self._summary_dir(companion_id, "daily")
        if not directory.exists():
            return []

        parsed: list[date] = []
        for path in directory.glob("*.md"):
            try:
                parsed.append(date.fromisoformat(path.stem))
            except ValueError:
                continue
        parsed.sort()
        return parsed

    def list_weeklies(self, companion_id: str) -> list[str]:
        """List weekly summary periods (YYYY-Www) in ascending order."""
        directory = self._summary_dir(companion_id, "weekly")
        if not directory.exists():
            return []
        return sorted(path.stem for path in directory.glob("*.md"))

    def list_monthlies(self, companion_id: str) -> list[str]:
        """List monthly summary periods (YYYY-MM) in ascending order."""
        directory = self._summary_dir(companion_id, "monthly")
        if not directory.exists():
            return []
        return sorted(path.stem for path in directory.glob("*.md"))

    def _write_file(self, directory: Path, filename: str, content: str) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        path.write_text(content, encoding="utf-8")
        return path

    def _summary_dir(self, companion_id: str, summary_type: str) -> Path:
        return self._data_dir / companion_id / "summaries" / summary_type

    def _load_summaries(self, companion_id: str, summary_type: str) -> list[StoredSummary]:
        directory = self._summary_dir(companion_id, summary_type)
        if not directory.exists():
            return []

        summaries: list[StoredSummary] = []
        for path in sorted(directory.glob("*.md"), key=lambda item: self._sort_key(summary_type, item.stem)):
            content = path.read_text(encoding="utf-8")
            summaries.append(
                StoredSummary(summary_type=summary_type, period=path.stem, content=content)
            )
        return summaries

    def _sort_key(self, summary_type: str, period: str) -> tuple[int, int, int]:
        if summary_type == "monthly":
            try:
                dt = datetime.strptime(period, "%Y-%m")
                return (dt.year, dt.month, 0)
            except ValueError:
                return (9999, 99, 0)
        if summary_type == "weekly":
            try:
                year_raw, week_raw = period.split("-W")
                year = int(year_raw)
                week = int(week_raw)
                return (year, week, 0)
            except ValueError:
                return (9999, 99, 0)
        try:
            dt = datetime.strptime(period, "%Y-%m-%d")
            return (dt.year, dt.month, dt.day)
        except ValueError:
            return (9999, 99, 99)
