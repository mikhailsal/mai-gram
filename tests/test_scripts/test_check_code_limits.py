from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "check_code_limits.py"
    spec = importlib.util.spec_from_file_location("check_code_limits", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_report_only_mode_returns_zero_for_violations(tmp_path, capsys) -> None:
    module = _load_module()
    source_path = tmp_path / "sample.py"
    source_path.write_text(
        "def oversized():\n    value = 1\n    value += 1\n    return value\n",
        encoding="utf-8",
    )

    exit_code = module.main(
        [
            str(source_path),
            "--max-file-lines",
            "2",
            "--max-function-lines",
            "2",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Report-only mode" in captured.out
    assert "sample.py" in captured.out
    assert "oversized" in captured.out


def test_enforcement_mode_returns_one_for_violations(tmp_path, capsys) -> None:
    module = _load_module()
    source_path = tmp_path / "sample.py"
    source_path.write_text(
        "def oversized():\n    total = 0\n    total += 1\n    return total\n",
        encoding="utf-8",
    )

    exit_code = module.main(
        [
            str(source_path),
            "--max-function-lines",
            "2",
            "--enforce",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Enforcement mode" in captured.out


def test_excluded_paths_are_skipped(tmp_path, capsys) -> None:
    module = _load_module()
    source_path = tmp_path / "excluded.py"
    source_path.write_text(
        "def oversized():\n    counter = 0\n    counter += 1\n    return counter\n",
        encoding="utf-8",
    )

    exit_code = module.main(
        [
            str(source_path),
            "--max-function-lines",
            "2",
            "--exclude",
            str(source_path),
            "--enforce",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Function violations: none" in captured.out
