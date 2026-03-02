"""Smoke-test runner для rfdetr-tool CLI.

Запуск: rfdetr-tool test [category=cli|linter|typecheck]
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class TestCase:
    """Описание одного smoke-теста."""

    name: str
    cmd: list[str]
    expect_exit: int
    category: str
    expect_in_output: list[str] = field(default_factory=list)
    expect_not_in_output: list[str] = field(default_factory=list)
    post_check: Callable[[], bool] | None = None
    depends_on: str | None = None


@dataclass
class TestResult:
    """Результат выполнения одного теста."""

    case: TestCase
    passed: bool
    actual_exit: int
    stdout: str
    stderr: str
    duration_ms: int
    failure_reason: str


# ANSI colors
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _project_root() -> Path:
    """Корень проекта — директория с pyproject.toml."""
    p = Path(__file__).resolve().parent.parent
    if (p / "pyproject.toml").exists():
        return p
    return Path.cwd()


def _build_test_catalog(output_dir: Path) -> list[TestCase]:
    """Каталог всех smoke-тестов."""
    cfg_output = str(output_dir / "config.yaml")

    return [
        # --- CLI ---
        TestCase(
            name="help",
            cmd=["rfdetr-tool"],
            expect_exit=0,
            expect_in_output=["rfdetr-tool"],
            category="cli",
        ),
        TestCase(
            name="cfg_stdout",
            cmd=["rfdetr-tool", "cfg", "variant=base"],
            expect_exit=0,
            expect_in_output=["variant: base"],
            category="cli",
        ),
        TestCase(
            name="cfg_file",
            cmd=["rfdetr-tool", "cfg", "variant=large", f"output={cfg_output}"],
            expect_exit=0,
            post_check=lambda: Path(cfg_output).exists(),
            category="cli",
        ),
        TestCase(
            name="train_no_args",
            cmd=["rfdetr-tool", "train"],
            expect_exit=1,
            expect_in_output=["Field required"],
            category="cli",
        ),
        TestCase(
            name="train_unknown_param",
            cmd=["rfdetr-tool", "train", "data=x", "unknown_param=y"],
            expect_exit=1,
            expect_in_output=["Неизвестный параметр"],
            category="cli",
        ),
        TestCase(
            name="train_invalid_variant",
            cmd=["rfdetr-tool", "train", "data=x", "variant=invalid"],
            expect_exit=1,
            category="cli",
        ),
        TestCase(
            name="train_missing_cfg",
            cmd=["rfdetr-tool", "train", "cfg=nonexistent.yaml", "data=x"],
            expect_exit=1,
            expect_in_output=["Файл"],
            category="cli",
        ),
        TestCase(
            name="train_gpus_invalid",
            cmd=["rfdetr-tool", "train", "data=x", "gpus=0"],
            expect_exit=1,
            category="cli",
        ),
        TestCase(
            name="bad_command",
            cmd=["rfdetr-tool", "badcommand"],
            expect_exit=1,
            expect_in_output=["Неизвестная команда"],
            category="cli",
        ),
        TestCase(
            name="train_cfg_override",
            cmd=[
                "rfdetr-tool",
                "train",
                f"cfg={cfg_output}",
                "data=./ds",
                "epochs=5",
            ],
            expect_exit=1,
            expect_not_in_output=["Ошибка валидации"],
            depends_on="cfg_file",
            category="cli",
        ),
        # --- Linter ---
        TestCase(
            name="ruff_check",
            cmd=["uv", "run", "ruff", "check", "rfdetr_tooling/"],
            expect_exit=0,
            category="linter",
        ),
        TestCase(
            name="ruff_format",
            cmd=["uv", "run", "ruff", "format", "--check", "rfdetr_tooling/"],
            expect_exit=0,
            category="linter",
        ),
        # --- Typecheck ---
        TestCase(
            name="mypy",
            cmd=["uv", "run", "mypy", "rfdetr_tooling/"],
            expect_exit=0,
            category="typecheck",
        ),
    ]


def _run_test(case: TestCase, project_root: Path) -> TestResult:
    """Запуск одного теста."""
    t0 = time.monotonic()
    proc = subprocess.run(  # noqa: S603
        case.cmd,
        capture_output=True,
        text=True,
        cwd=project_root,
        check=False,
    )
    duration_ms = int((time.monotonic() - t0) * 1000)

    combined = proc.stdout + proc.stderr
    passed = True
    failure_reason = ""

    if proc.returncode != case.expect_exit:
        passed = False
        failure_reason = f"expected exit={case.expect_exit}, got {proc.returncode}"

    if passed:
        for substr in case.expect_in_output:
            if substr not in combined:
                passed = False
                failure_reason = f"output missing: {substr!r}"
                break

    if passed:
        for substr in case.expect_not_in_output:
            if substr in combined:
                passed = False
                failure_reason = f"output contains unexpected: {substr!r}"
                break

    if passed and case.post_check is not None and not case.post_check():
        passed = False
        failure_reason = "post_check failed"

    return TestResult(
        case=case,
        passed=passed,
        actual_exit=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        duration_ms=duration_ms,
        failure_reason=failure_reason,
    )


def _format_line(result: TestResult, status: str, color: str, note: str) -> str:
    """Форматирование строки результата."""
    cat = f"[{result.case.category}]"
    name = result.case.name
    ms = f"({result.duration_ms}ms)"
    suffix = f"   {note}" if note else ""
    return f"  {cat:<14} {color}{status}{_RESET}  {name:<26} {_DIM}{ms}{_RESET}{suffix}"


def _write_log(log_path: Path, results: list[TestResult]) -> None:
    """Сохранение полного лога в файл."""
    lines: list[str] = []
    for r in results:
        lines.append(f"{'=' * 60}")
        lines.append(f"TEST: {r.case.name} ({r.case.category})")
        lines.append(f"CMD:  {' '.join(r.case.cmd)}")
        lines.append(f"EXIT: {r.actual_exit} (expected {r.case.expect_exit})")
        lines.append(f"PASS: {r.passed}")
        if r.failure_reason:
            lines.append(f"FAIL: {r.failure_reason}")
        lines.append(f"TIME: {r.duration_ms}ms")
        if r.stdout.strip():
            lines.append(f"\n--- stdout ---\n{r.stdout.rstrip()}")
        if r.stderr.strip():
            lines.append(f"\n--- stderr ---\n{r.stderr.rstrip()}")
        lines.append("")
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _execute_cases(
    cases: list[TestCase],
    project_root: Path,
) -> list[TestResult]:
    """Последовательный запуск тестов с учётом зависимостей."""
    results: list[TestResult] = []
    result_by_name: dict[str, TestResult] = {}

    for case in cases:
        if case.depends_on is not None:
            dep = result_by_name.get(case.depends_on)
            if dep is None or not dep.passed:
                skip_result = TestResult(
                    case=case,
                    passed=False,
                    actual_exit=-1,
                    stdout="",
                    stderr="",
                    duration_ms=0,
                    failure_reason=f"prerequisite {case.depends_on} failed",
                )
                results.append(skip_result)
                result_by_name[case.name] = skip_result
                line = _format_line(
                    skip_result,
                    "SKIP",
                    _YELLOW,
                    f"(prerequisite {case.depends_on} failed)",
                )
                sys.stdout.write(line + "\n")
                continue

        result = _run_test(case, project_root)
        results.append(result)
        result_by_name[case.name] = result

        if result.passed:
            line = _format_line(result, "PASS", _GREEN, "")
        else:
            line = _format_line(result, "FAIL", _RED, result.failure_reason)

        sys.stdout.write(line + "\n")

    return results


def _print_summary(
    results: list[TestResult],
    log_path: Path,
    project_root: Path,
    total_s: float,
) -> None:
    """Вывод итогов тестирования."""
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed and r.actual_exit != -1)
    skipped = sum(1 for r in results if r.actual_exit == -1)

    sys.stdout.write(f"{'─' * 50}\n")
    parts: list[str] = []
    if passed:
        parts.append(f"{_GREEN}{passed} passed{_RESET}")
    if failed:
        parts.append(f"{_RED}{failed} failed{_RESET}")
    if skipped:
        parts.append(f"{_YELLOW}{skipped} skipped{_RESET}")
    summary = ", ".join(parts)
    sys.stdout.write(f"  {_BOLD}Результат:{_RESET} {summary}  |  {total_s:.1f}s\n")
    sys.stdout.write(f"  Лог: {log_path.relative_to(project_root)}\n")


def run_tests(overrides: dict[str, str]) -> None:
    """Запуск smoke-тестов. Вызывается из CLI."""
    project_root = _project_root()
    output_dir = project_root / ".test_output"
    output_dir.mkdir(exist_ok=True)

    category_filter = overrides.get("category")

    cases = _build_test_catalog(output_dir)
    if category_filter:
        cases = [c for c in cases if c.category == category_filter]
        if not cases:
            sys.stderr.write(
                f"{_RED}Нет тестов для категории: {category_filter!r}{_RESET}\n"
            )
            sys.exit(1)

    t0 = time.monotonic()
    results = _execute_cases(cases, project_root)
    total_s = time.monotonic() - t0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
    log_path = output_dir / f"test_run_{timestamp}.log"
    _write_log(log_path, results)

    _print_summary(results, log_path, project_root, total_s)

    has_failures = any(not r.passed and r.actual_exit != -1 for r in results)
    sys.exit(1 if has_failures else 0)
