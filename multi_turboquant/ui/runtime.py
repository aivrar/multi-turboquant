# SPDX-License-Identifier: MIT
"""Managed local processes and background environment jobs for the UI."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Mapping, Sequence


MAX_CONCURRENT_GODZILLA_CALIBRATIONS = 1


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _split_env_wrapper(command: Sequence[str]) -> tuple[list[str], dict[str, str]]:
    argv = [str(item) for item in command]
    environment: dict[str, str] = {}
    if argv[:1] != ["env"]:
        return argv, environment
    index = 1
    while index < len(argv) and "=" in argv[index]:
        key, value = argv[index].split("=", 1)
        if not key or not key.replace("_", "a").isalnum():
            break
        environment[key] = value
        index += 1
    return argv[index:], environment


class ManagedProcess:
    """Own exactly one child process launched without a shell."""

    def __init__(self, *, log_lines: int = 400):
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._command: list[str] = []
        self._started_at: float | None = None
        self._log: deque[str] = deque(maxlen=log_lines)
        self._collector: threading.Thread | None = None

    def _collect_output(self, process: subprocess.Popen[str]) -> None:
        stream = process.stdout
        if stream is None:
            return
        try:
            for line in stream:
                with self._lock:
                    self._log.append(line.rstrip())
        finally:
            stream.close()
            with self._lock:
                if self._collector is threading.current_thread():
                    self._collector = None

    def start(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        argv, wrapper_environment = _split_env_wrapper(command)
        if not argv:
            raise ValueError("Launch command is empty")
        executable = argv[0]
        if any(separator in executable for separator in ("/", "\\")):
            executable_path = Path(executable).expanduser().resolve()
            if not executable_path.is_file():
                raise ValueError(f"llama-server binary does not exist: {executable_path}")
            argv[0] = str(executable_path)
        elif shutil.which(executable) is None:
            raise ValueError(f"llama-server binary was not found on PATH: {executable}")

        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("A llama-server process is already running")
            previous_collector = self._collector
        if previous_collector is not None:
            previous_collector.join(timeout=2)
            if previous_collector.is_alive():
                raise RuntimeError("The previous llama-server output stream is still draining")

        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("A llama-server process is already running")
            child_environment = dict(os.environ)
            child_environment.update(wrapper_environment)
            if environment:
                child_environment.update(
                    {str(key): str(value) for key, value in environment.items()}
                )
            creationflags = 0
            if os.name == "nt":
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self._log.clear()
            self._process = subprocess.Popen(
                argv,
                cwd=str(Path(cwd).resolve()) if cwd else None,
                env=child_environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                shell=False,
                creationflags=creationflags,
            )
            self._command = argv
            self._started_at = time.time()
            self._collector = threading.Thread(
                target=self._collect_output,
                args=(self._process,),
                daemon=True,
            )
            self._collector.start()
            return self.status()

    def stop(self, *, timeout: float = 10.0) -> dict[str, object]:
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        with self._lock:
            collector = self._collector
        if collector is not None and collector is not threading.current_thread():
            collector.join(timeout=max(1.0, min(timeout, 5.0)))
        return self.status()

    def status(self) -> dict[str, object]:
        with self._lock:
            process = self._process
            returncode = process.poll() if process is not None else None
            running = process is not None and returncode is None
            return {
                "running": running,
                "pid": process.pid if running and process is not None else None,
                "returncode": returncode,
                "command": list(self._command),
                "started_at": self._started_at,
                "log": list(self._log),
            }

    def shutdown(self) -> None:
        self.stop(timeout=3)


class EnvironmentJobManager:
    """Run explicit dependency environment creation in background threads."""

    def __init__(self):
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, object]] = {}

    def _snapshot(self, job: Mapping[str, object]) -> dict[str, object]:
        return {
            key: list(value) if isinstance(value, deque) else value
            for key, value in job.items()
            if not key.startswith("_")
        }

    def start_create(
        self,
        profile_id: str,
        *,
        root: str | Path,
        python: str | None = None,
        cuda_toolkit: str | Path | None = None,
        local_source: str | Path | None = None,
        build_from_source: bool = False,
        max_jobs: int | None = None,
        recreate: bool = False,
    ) -> dict[str, object]:
        from ..optimizations.environments import plan_environment

        plan = plan_environment(
            profile_id,
            root=root,
            python=python,
            cuda_toolkit=cuda_toolkit,
            local_source=local_source,
            build_from_source=build_from_source,
            max_jobs=max_jobs,
        )
        if not plan.ready:
            errors = "; ".join(issue.message for issue in plan.issues if issue.severity == "error")
            raise ValueError(errors or "Environment plan is not ready")
        job_id = uuid.uuid4().hex
        job: dict[str, object] = {
            "id": job_id,
            "profile": profile_id,
            "status": "queued",
            "build_from_source": build_from_source,
            "max_jobs": getattr(plan, "max_jobs", max_jobs),
            "recreate": recreate,
            "cuda_toolkit_root": (
                str(plan.cuda_toolkit_root) if plan.cuda_toolkit_root is not None else None
            ),
            "local_source": (str(plan.local_source) if plan.local_source is not None else None),
            "target": str(plan.target),
            "created_at": time.time(),
            "finished_at": None,
            "report": None,
            "diagnostics": None,
            "backup": None,
            "error": None,
            "log": deque(maxlen=500),
        }
        with self._lock:
            if any(
                item["profile"] == profile_id and item["status"] in {"queued", "running"}
                for item in self._jobs.values()
            ):
                raise RuntimeError(f"An environment job for {profile_id} is already running")
            self._jobs[job_id] = job
        threading.Thread(
            target=self._run_create,
            args=(job_id, plan, recreate),
            daemon=True,
        ).start()
        return self.get(job_id)

    def _run_create(self, job_id: str, plan: object, recreate: bool) -> None:
        from ..optimizations.environments import (
            check_environment,
            diagnose_environment,
            synchronize_environment,
        )

        with self._lock:
            job = self._jobs[job_id]
            job["status"] = "running"

        def runner(argv: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            result = subprocess.run(
                list(argv),
                cwd=kwargs.get("cwd"),
                env=kwargs.get("env"),
                check=False,
                capture_output=True,
                text=True,
            )
            with self._lock:
                log = self._jobs[job_id]["log"]
                assert isinstance(log, deque)
                log.extend((result.stdout or "").splitlines())
                log.extend((result.stderr or "").splitlines())
            return result

        try:
            backup = synchronize_environment(  # type: ignore[arg-type]
                plan,
                recreate=recreate,
                runner=runner,
            )
            report = check_environment(plan)  # type: ignore[arg-type]
        except Exception as exc:
            try:
                diagnostics = diagnose_environment(plan)  # type: ignore[arg-type]
            except Exception as diagnostic_exc:
                diagnostics = {"error": f"Diagnostic collection failed: {diagnostic_exc}"}
            with self._lock:
                job = self._jobs[job_id]
                job["status"] = "failed"
                job["error"] = str(exc)
                job["diagnostics"] = diagnostics
                job["finished_at"] = time.time()
        else:
            with self._lock:
                job = self._jobs[job_id]
                job["status"] = "completed"
                job["report"] = dict(report)
                job["backup"] = str(backup) if backup is not None else None
                job["finished_at"] = time.time()

    def get(self, job_id: str) -> dict[str, object]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"Unknown job: {job_id}")
            return self._snapshot(self._jobs[job_id])

    def list(self) -> list[dict[str, object]]:
        with self._lock:
            return [
                self._snapshot(item)
                for item in sorted(
                    self._jobs.values(),
                    key=lambda value: float(value["created_at"]),
                    reverse=True,
                )
            ]


class GodzillaCalibrationJobManager:
    """Run an explicitly confirmed Godzilla TriAttention preparation job."""

    max_concurrent_jobs = MAX_CONCURRENT_GODZILLA_CALIBRATIONS

    def __init__(self):
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, object]] = {}

    def _snapshot(self, job: Mapping[str, object]) -> dict[str, object]:
        return {
            key: list(value) if isinstance(value, deque) else value
            for key, value in job.items()
            if not key.startswith("_")
        }

    def start(
        self,
        checkout: str | Path,
        gguf: str | Path,
        *,
        output: str | Path | None = None,
        python: str | Path | None = None,
        calibrator: str | Path | None = None,
        calibration_input: str | Path | None = None,
        official_stats_input: str | Path | None = None,
        domvox_calibrator: str | Path | None = None,
        domvox_accept_lossy: bool = False,
        allow_long_calibration: bool = False,
        hf_model: str | None = None,
        n_tokens: int = 2048,
        device: str = "cuda",
        mode: str = "official_python",
        attention_implementation: str = "sdpa",
        tokenizer_backend: str = "transformers",
        dependency_override: bool = False,
        python_discovery: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        from ..calibration.godzilla_triattention import load_huggingface_model_metadata
        from ..integration.godzilla_workspace import plan_godzilla_triattention

        plan = plan_godzilla_triattention(
            checkout,
            gguf,
            output=output,
            python=python,
            calibrator=calibrator,
            calibration_input=calibration_input,
            official_stats_input=official_stats_input,
            domvox_calibrator=domvox_calibrator,
            domvox_accept_lossy=domvox_accept_lossy,
            allow_long_calibration=allow_long_calibration,
            hf_model=hf_model,
            n_tokens=n_tokens,
            device=device,
            mode=mode,
            attention_implementation=attention_implementation,
            tokenizer_backend=tokenizer_backend,
            verify_dependencies=True,
            dependency_override=dependency_override,
            python_discovery=python_discovery,
            model_metadata_loader=lambda model_id: load_huggingface_model_metadata(
                model_id, trust_remote_code=False
            ),
        )
        if not plan.ready:
            errors = "; ".join(issue.message for issue in plan.issues if issue.severity == "error")
            raise ValueError(errors or "Godzilla calibration plan is not ready")

        job_id = uuid.uuid4().hex
        job: dict[str, object] = {
            "id": job_id,
            "type": "godzilla-triattention",
            "status": "queued",
            "checkout": str(plan.checkout),
            "model": str(plan.gguf),
            "output": str(plan.output),
            "mode": plan.mode,
            "python": str(plan.python) if plan.python is not None else None,
            "command": list(plan.command),
            "dependency_preflight": plan.dependency_validation,
            "tokenizer_backend": getattr(plan, "tokenizer_backend", "transformers"),
            "process_limit": MAX_CONCURRENT_GODZILLA_CALIBRATIONS,
            "created_at": time.time(),
            "finished_at": None,
            "report": None,
            "runtime_preflight": None,
            "diagnostics": None,
            "diagnostics_path": None,
            "error": None,
            "log": deque(maxlen=500),
        }
        with self._lock:
            if any(item["status"] in {"queued", "running"} for item in self._jobs.values()):
                raise RuntimeError(
                    "A Godzilla calibration is already queued or running. The local UI limits "
                    "calibration to one process at a time to avoid overlapping model loads."
                )
            self._jobs[job_id] = job
        threading.Thread(target=self._run, args=(job_id, plan), daemon=True).start()
        return self.get(job_id)

    def _run(self, job_id: str, plan: object) -> None:
        from ..calibration.godzilla_triattention import inspect_calibration_python
        from ..integration.godzilla_workspace import (
            collect_godzilla_calibration_diagnostics,
            run_godzilla_triattention,
        )
        from ..optimizations.environments import redact_diagnostic_text

        with self._lock:
            self._jobs[job_id]["status"] = "running"

        last_result: subprocess.CompletedProcess[str] | None = None

        def runner(argv: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            nonlocal last_result
            result = subprocess.run(
                list(argv),
                cwd=kwargs.get("cwd"),
                env=kwargs.get("env"),
                check=False,
                capture_output=True,
                text=True,
            )
            with self._lock:
                log = self._jobs[job_id]["log"]
                assert isinstance(log, deque)
                log.extend(
                    redact_diagnostic_text(line) for line in (result.stdout or "").splitlines()
                )
                log.extend(
                    redact_diagnostic_text(line) for line in (result.stderr or "").splitlines()
                )
            last_result = result
            return result

        try:
            output = getattr(plan, "output")
            mode = getattr(plan, "mode")
            python = getattr(plan, "python")
            if not output.is_file() and mode in {"official_python", "official_convert", "domvox"}:
                if python is None or not python.is_file():
                    raise RuntimeError(
                        "The selected calibration Python disappeared before the job started. "
                        "Repair or reselect the TriAttention environment."
                    )
                runtime_preflight = inspect_calibration_python(
                    python,
                    device=getattr(plan, "device"),
                    tokenizer_backend=getattr(plan, "tokenizer_backend", "transformers"),
                    attention_implementation=getattr(
                        plan,
                        "attention_implementation",
                        "sdpa",
                    )
                    if mode == "official_python"
                    else "sdpa",
                )
                with self._lock:
                    self._jobs[job_id]["runtime_preflight"] = runtime_preflight
                if not runtime_preflight["valid"]:
                    details = "; ".join(str(item) for item in runtime_preflight["issues"])
                    raise RuntimeError(
                        "The selected calibration Python failed the final dependency preflight: "
                        f"{details}. Dependency overrides cannot start calibration; repair or "
                        "select a compatible environment."
                    )
            report = run_godzilla_triattention(plan, runner=runner)  # type: ignore[arg-type]
        except Exception as exc:
            stdout = last_result.stdout if last_result is not None else ""
            stderr = last_result.stderr if last_result is not None else ""
            returncode = last_result.returncode if last_result is not None else None
            try:
                diagnostics = collect_godzilla_calibration_diagnostics(
                    plan,  # type: ignore[arg-type]
                    failure=exc,
                    returncode=returncode,
                    stdout=stdout or "",
                    stderr=stderr or "",
                )
            except Exception as diagnostic_exc:
                diagnostics = {
                    "schema": 1,
                    "collection_error": redact_diagnostic_text(
                        f"{type(diagnostic_exc).__name__}: {diagnostic_exc}"
                    ),
                    "original_error": redact_diagnostic_text(str(exc)),
                }
            diagnostics_path = getattr(plan, "output").with_name(
                getattr(plan, "output").name + f".{job_id[:12]}.diagnostics.json"
            )
            diagnostics_write_error = None
            try:
                _write_json_atomic(diagnostics_path, diagnostics)
            except (OSError, TypeError, ValueError) as write_exc:
                diagnostics_write_error = redact_diagnostic_text(
                    f"{type(write_exc).__name__}: {write_exc}"
                )
            with self._lock:
                job = self._jobs[job_id]
                job["status"] = "failed"
                job["error"] = redact_diagnostic_text(str(exc))
                job["diagnostics"] = diagnostics
                job["diagnostics_path"] = (
                    str(diagnostics_path) if diagnostics_write_error is None else None
                )
                if diagnostics_write_error is not None:
                    job["diagnostics_write_error"] = diagnostics_write_error
                job["finished_at"] = time.time()
        else:
            with self._lock:
                job = self._jobs[job_id]
                job["status"] = "completed"
                job["report"] = dict(report)
                job["finished_at"] = time.time()

    def get(self, job_id: str) -> dict[str, object]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"Unknown job: {job_id}")
            return self._snapshot(self._jobs[job_id])

    def list(self) -> list[dict[str, object]]:
        with self._lock:
            return [
                self._snapshot(item)
                for item in sorted(
                    self._jobs.values(),
                    key=lambda value: float(value["created_at"]),
                    reverse=True,
                )
            ]
