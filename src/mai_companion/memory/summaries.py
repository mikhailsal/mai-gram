"""Summary file store for daily, weekly, and monthly memory layers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class StoredSummary:
    """One stored summary document."""

    summary_type: str
    period: str
    content: str


@dataclass(frozen=True)
class SummaryVersion:
    """A historical version of a summary."""

    summary_type: str
    period: str
    version_id: str  # Timestamp-based ID like "v1_2024-02-24T10:30:00"
    content: str
    timestamp: datetime


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

    def save_version(
        self,
        companion_id: str,
        summary_type: str,
        period: str,
        content: str,
    ) -> Path:
        """Save a version of a summary before re-consolidation.

        Versions are stored in a .versions subdirectory with timestamp-based names.
        """
        versions_dir = self._summary_dir(companion_id, summary_type) / ".versions"
        versions_dir.mkdir(parents=True, exist_ok=True)

        # Count existing versions for this period
        existing = list(versions_dir.glob(f"{period}_v*.md"))
        version_num = len(existing) + 1
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")

        filename = f"{period}_v{version_num}_{timestamp}.md"
        path = versions_dir / filename
        path.write_text(content, encoding="utf-8")
        return path

    def list_versions(
        self,
        companion_id: str,
        summary_type: str,
        period: str,
    ) -> list[SummaryVersion]:
        """List all versions of a specific summary."""
        versions_dir = self._summary_dir(companion_id, summary_type) / ".versions"
        if not versions_dir.exists():
            return []

        versions: list[SummaryVersion] = []
        for path in sorted(versions_dir.glob(f"{period}_v*.md")):
            content = path.read_text(encoding="utf-8")
            # Parse version info from filename: {period}_v{num}_{timestamp}.md
            stem = path.stem
            parts = stem.split("_v", 1)
            if len(parts) < 2:
                continue
            version_part = parts[1]  # e.g., "1_2024-02-24T10-30-00"
            version_id = f"v{version_part}"

            # Extract timestamp from version_part
            ts_parts = version_part.split("_", 1)
            if len(ts_parts) >= 2:
                ts_str = ts_parts[1].replace("-", ":", 2)  # Fix time separators
                # Format: 2024-02-24T10:30:00
                ts_str = ts_str[:10] + "T" + ts_str[11:].replace("-", ":")
                try:
                    timestamp = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
                except ValueError:
                    timestamp = datetime.now(timezone.utc)
            else:
                timestamp = datetime.now(timezone.utc)

            versions.append(SummaryVersion(
                summary_type=summary_type,
                period=period,
                version_id=version_id,
                content=content,
                timestamp=timestamp,
            ))

        return sorted(versions, key=lambda v: v.version_id)

    def get_all_summaries_with_versions(
        self,
        companion_id: str,
    ) -> dict[str, list[tuple[StoredSummary, list[SummaryVersion]]]]:
        """Get all summaries organized by type with their version history.

        Returns a dict with keys 'daily', 'weekly', 'monthly', each containing
        a list of (current_summary, versions) tuples.
        """
        result: dict[str, list[tuple[StoredSummary, list[SummaryVersion]]]] = {
            "daily": [],
            "weekly": [],
            "monthly": [],
        }

        for summary_type in ["daily", "weekly", "monthly"]:
            summaries = self._load_summaries(companion_id, summary_type)
            for summary in summaries:
                versions = self.list_versions(companion_id, summary_type, summary.period)
                result[summary_type].append((summary, versions))

        return result

    def get_daily(self, companion_id: str, target_date: date) -> StoredSummary | None:
        """Get a specific daily summary."""
        path = self._summary_dir(companion_id, "daily") / f"{target_date.isoformat()}.md"
        if not path.exists():
            return None
        content = path.read_text(encoding="utf-8")
        return StoredSummary(summary_type="daily", period=target_date.isoformat(), content=content)

    def get_weekly(self, companion_id: str, period: str) -> StoredSummary | None:
        """Get a specific weekly summary."""
        path = self._summary_dir(companion_id, "weekly") / f"{period}.md"
        if not path.exists():
            return None
        content = path.read_text(encoding="utf-8")
        return StoredSummary(summary_type="weekly", period=period, content=content)

    def get_monthly(self, companion_id: str, period: str) -> StoredSummary | None:
        """Get a specific monthly summary."""
        path = self._summary_dir(companion_id, "monthly") / f"{period}.md"
        if not path.exists():
            return None
        content = path.read_text(encoding="utf-8")
        return StoredSummary(summary_type="monthly", period=period, content=content)

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
