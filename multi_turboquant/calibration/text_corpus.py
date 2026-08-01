# SPDX-License-Identifier: MIT
"""Deterministic, offline text for calibration smoke tests and first runs."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

from .godzilla_triattention import MAX_CALIBRATION_TOKENS


_CORPUS_MARKER = "Multi-TurboQuant generic calibration corpus"
CALIBRATION_CORPUS_SCHEMA_VERSION = 1
_CORPUS_END_MARKER = (
    f"# End of {_CORPUS_MARKER} schema {CALIBRATION_CORPUS_SCHEMA_VERSION}"
)
_SUBJECTS = (
    "a systems engineer",
    "a research librarian",
    "a field technician",
    "a careful reviewer",
    "a data analyst",
    "a software maintainer",
    "a laboratory assistant",
    "an operations planner",
)
_ACTIONS = (
    "compares the observations",
    "checks the assumptions",
    "summarizes the evidence",
    "records the measurements",
    "tests the boundary conditions",
    "explains the tradeoffs",
    "reconstructs the sequence",
    "verifies the result",
)
_CONTEXTS = (
    "before changing the configuration",
    "while preserving the original data",
    "under a strict memory budget",
    "with both common and unusual inputs",
    "without relying on network access",
    "and reports uncertainty explicitly",
    "using a reproducible local procedure",
    "before the final handoff",
)
_QUESTIONS = (
    "Which observation would change the conclusion?",
    "What should be measured again before deployment?",
    "How can the result be reproduced independently?",
    "Where is the strongest source of uncertainty?",
    "Which resource becomes the limiting factor first?",
    "What evidence distinguishes correlation from cause?",
    "How should an incomplete input be reported?",
    "Which fallback preserves the most information?",
)


def _corpus_block(index: int) -> str:
    subject = _SUBJECTS[index % len(_SUBJECTS)]
    action = _ACTIONS[(index * 3 + 1) % len(_ACTIONS)]
    context = _CONTEXTS[(index * 5 + 2) % len(_CONTEXTS)]
    question = _QUESTIONS[(index * 7 + 3) % len(_QUESTIONS)]
    left = (index * 37) % 997
    right = (index * 91 + 17) % 1237
    return (
        f"Passage {index:05d}. {subject.capitalize()} {action} {context}. "
        "The report separates confirmed facts, estimates, and open questions so that a later "
        "reader can audit each decision. Short clauses alternate with longer explanations; "
        "names, dates, units, punctuation, and numeric values remain unambiguous.\n"
        f"Question: {question}\n"
        "Answer: Recheck the relevant input, state the applicable limit, and retain the "
        "measurement that supports the decision.\n"
        f"Structured sample: {{\"passage\": {index}, \"left\": {left}, "
        f"\"right\": {right}, \"accepted\": true}}\n"
        "Notation sample: x -> y; 0 <= ratio <= 1; memory = capacity - reserved; "
        "paths/use-forward/slashes and paths\\use\\backslashes are both represented."
    )


def _reusable_corpus_report(
    output: Path,
    *,
    header: str,
    requested_tokens: int,
) -> dict[str, object] | None:
    """Recognize only a complete corpus generated for the same request."""
    if not output.is_file():
        return None
    existing_text = output.read_text(encoding="utf-8", errors="replace")
    if not (
        existing_text.startswith(header)
        and existing_text.endswith(f"{_CORPUS_END_MARKER}\n")
    ):
        return None
    return {
        "path": str(output),
        "requested_tokens": requested_tokens,
        "characters": len(existing_text),
        "reused": True,
        "generic": True,
        "schema": CALIBRATION_CORPUS_SCHEMA_VERSION,
    }


def _publish_without_hardlinks(temporary_path: Path, output: Path) -> None:
    """Publish without clobbering on filesystems that do not support hard links."""
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as destination, temporary_path.open("rb") as source:
            shutil.copyfileobj(source, destination)
            destination.flush()
            os.fsync(destination.fileno())
        published = True
    finally:
        if not published:
            output.unlink(missing_ok=True)


def _wait_for_concurrent_corpus(
    output: Path,
    *,
    header: str,
    requested_tokens: int,
    timeout: float = 2.0,
) -> dict[str, object] | None:
    """Wait briefly only when a competing writer is publishing our corpus."""
    deadline = time.monotonic() + timeout
    while True:
        report = _reusable_corpus_report(
            output,
            header=header,
            requested_tokens=requested_tokens,
        )
        if report is not None:
            return report
        try:
            with output.open("r", encoding="utf-8", errors="replace") as existing:
                prefix = existing.read(len(header))
        except (OSError, ValueError):
            return None
        # An exclusive fallback file can be briefly empty or contain a prefix
        # of the expected header. Anything else is unrelated and is never reused.
        if prefix and not (header.startswith(prefix) or prefix.startswith(header)):
            return None
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.01)


def generate_calibration_text(
    output_path: str | Path,
    *,
    target_tokens: int = 2048,
) -> dict[str, object]:
    """Create a deterministic generic corpus without downloading or executing code.

    Character count is intentionally generous because exact tokenization depends on the
    selected model. The upstream calibrator remains responsible for truncating to its
    requested maximum length.
    """
    if (
        isinstance(target_tokens, bool)
        or not isinstance(target_tokens, int)
        or not 128 <= target_tokens <= MAX_CALIBRATION_TOKENS
    ):
        raise ValueError(
            f"target_tokens must be between 128 and {MAX_CALIBRATION_TOKENS}"
        )
    requested_tokens = target_tokens
    output = Path(output_path).expanduser().resolve()
    if output.suffix.lower() != ".txt":
        raise ValueError("Generated calibration text must use a .txt filename")

    header = (
        f"# {_CORPUS_MARKER}\n"
        f"# Schema: {CALIBRATION_CORPUS_SCHEMA_VERSION}\n"
        f"# Requested calibration tokens: {requested_tokens}\n"
        "# Generic starter data; use a representative domain corpus for quality qualification.\n"
    )
    reusable_report = _reusable_corpus_report(
        output,
        header=header,
        requested_tokens=requested_tokens,
    )
    if reusable_report is not None:
        return reusable_report
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing calibration file: {output}")

    target_characters = max(4096, requested_tokens * 6)
    blocks: list[str] = []
    characters = len(header)
    index = 1
    while characters < target_characters:
        block = _corpus_block(index)
        blocks.append(block)
        characters += len(block) + 2
        index += 1
    text = header + "\n" + "\n\n".join(blocks) + f"\n{_CORPUS_END_MARKER}\n"

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        try:
            try:
                # A same-directory hard link publishes the fully written temporary file
                # atomically and, unlike replace(), can never clobber a competing file.
                os.link(temporary_path, output)
            except OSError as exc:
                if isinstance(exc, FileExistsError):
                    raise
                # exFAT, some network shares, and restricted Windows volumes may not
                # support hard links. Exclusive creation retains the no-clobber rule;
                # the end marker prevents a partial file from being treated as reusable.
                _publish_without_hardlinks(temporary_path, output)
        except FileExistsError as exc:
            reusable_report = _wait_for_concurrent_corpus(
                output,
                header=header,
                requested_tokens=requested_tokens,
            )
            if reusable_report is not None:
                return reusable_report
            raise FileExistsError(
                f"Refusing to overwrite an existing calibration file: {output}"
            ) from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return {
        "path": str(output),
        "requested_tokens": requested_tokens,
        "characters": len(text),
        "reused": False,
        "generic": True,
        "schema": CALIBRATION_CORPUS_SCHEMA_VERSION,
    }
