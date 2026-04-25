#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

DEFAULT_PATHS = ("src/mai_gram",)
DEFAULT_EXCLUDES = ("src/mai_gram/db/migrations.py",)
DEFAULT_MAX_FILE_LINES = 500
DEFAULT_MAX_FUNCTION_LINES = 60


@dataclass(frozen=True)
class FileViolation:
    path: str
    line_count: int


@dataclass(frozen=True)
class FunctionViolation:
    path: str
    qualname: str
    start_line: int
    line_count: int


class _FunctionVisitor(ast.NodeVisitor):
    def __init__(self, path_label: str, max_lines: int) -> None:
        self._path_label = path_label
        self._max_lines = max_lines
        self._scope_stack: list[str] = []
        self.violations: list[FunctionViolation] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record_function(node)

    def _record_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        end_line = node.end_lineno or node.lineno
        line_count = end_line - node.lineno + 1
        qualname = ".".join([*self._scope_stack, node.name])
        if line_count > self._max_lines:
            self.violations.append(
                FunctionViolation(
                    path=self._path_label,
                    qualname=qualname,
                    start_line=node.lineno,
                    line_count=line_count,
                )
            )

        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()


def _path_labels(path: Path, workspace_root: Path) -> set[str]:
    labels = {path.resolve().as_posix()}
    with suppress(ValueError):
        labels.add(path.resolve().relative_to(workspace_root.resolve()).as_posix())
    return labels


def _should_exclude(path: Path, workspace_root: Path, excluded: set[str]) -> bool:
    return bool(_path_labels(path, workspace_root) & excluded)


def _iter_python_files(
    roots: Iterable[Path], workspace_root: Path, excluded: set[str]
) -> list[Path]:
    files: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            candidate_files = [root] if root.suffix == ".py" else []
        else:
            candidate_files = sorted(root.rglob("*.py"))
        for candidate in candidate_files:
            if not _should_exclude(candidate, workspace_root, excluded):
                files.add(candidate.resolve())
    return sorted(files)


def _path_label(path: Path, workspace_root: Path) -> str:
    try:
        return path.resolve().relative_to(workspace_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _count_lines(source: str) -> int:
    return len(source.splitlines())


def analyze_paths(
    *,
    paths: Iterable[str],
    max_file_lines: int,
    max_function_lines: int,
    excludes: Iterable[str],
    workspace_root: Path | None = None,
) -> tuple[list[FileViolation], list[FunctionViolation]]:
    root = (workspace_root or Path.cwd()).resolve()
    excluded = {str(Path(item).resolve().as_posix()) for item in excludes}
    excluded.update(Path(item).as_posix() for item in excludes)
    python_files = _iter_python_files((Path(item) for item in paths), root, excluded)

    file_violations: list[FileViolation] = []
    function_violations: list[FunctionViolation] = []
    for path in python_files:
        source = path.read_text(encoding="utf-8")
        path_label = _path_label(path, root)
        line_count = _count_lines(source)
        if line_count > max_file_lines:
            file_violations.append(FileViolation(path=path_label, line_count=line_count))

        module = ast.parse(source, filename=path_label)
        visitor = _FunctionVisitor(path_label, max_function_lines)
        visitor.visit(module)
        function_violations.extend(visitor.violations)

    file_violations.sort(key=lambda item: (-item.line_count, item.path))
    function_violations.sort(key=lambda item: (-item.line_count, item.path, item.start_line))
    return file_violations, function_violations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit Python file and function sizes. Violations are reported without failing "
            "unless --enforce is set."
        )
    )
    parser.add_argument("paths", nargs="*", default=list(DEFAULT_PATHS))
    parser.add_argument("--max-file-lines", type=int, default=DEFAULT_MAX_FILE_LINES)
    parser.add_argument("--max-function-lines", type=int, default=DEFAULT_MAX_FUNCTION_LINES)
    parser.add_argument("--exclude", action="append", default=list(DEFAULT_EXCLUDES))
    parser.add_argument("--enforce", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    file_violations, function_violations = analyze_paths(
        paths=args.paths,
        max_file_lines=args.max_file_lines,
        max_function_lines=args.max_function_lines,
        excludes=args.exclude,
    )

    print(
        "Code size audit: "
        f"file limit={args.max_file_lines}, function limit={args.max_function_lines}"
    )
    if args.exclude:
        print("Excluded paths:")
        for excluded_path in args.exclude:
            print(f"  - {excluded_path}")

    if file_violations:
        print("\nFile violations:")
        for violation in file_violations:
            print(f"  - {violation.path}: {violation.line_count} lines")
    else:
        print("\nFile violations: none")

    if function_violations:
        print("\nFunction violations:")
        for violation in function_violations:
            print(
                f"  - {violation.path}:{violation.start_line} "
                f"{violation.qualname}: {violation.line_count} lines"
            )
    else:
        print("\nFunction violations: none")

    has_violations = bool(file_violations or function_violations)
    if has_violations and not args.enforce:
        print("\nReport-only mode: size violations do not fail the command yet.")
        return 0
    if has_violations:
        print("\nEnforcement mode: failing because size violations were found.")
        return 1

    print("\nNo size violations found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
