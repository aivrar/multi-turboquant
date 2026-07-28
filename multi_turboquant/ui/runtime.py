# SPDX-License-Identifier: MIT
"""Managed local processes and background environment jobs for the UI."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Mapping, Sequence


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

    def _collect_output(self, process: subprocess.Popen[str]) -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            with self._lock:
                self._log.append(line.rstrip())

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
            child_environment = dict(os.environ)
            child_environment.update(wrapper_environment)
            if environment:
                child_environment.update({str(key): str(value) for key, value in environment.items()})
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
            threading.Thread(
                target=self._collect_output,
                args=(self._process,),
                daemon=True,
            ).start()
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
    ) -> dict[str, object]:
        from ..optimizations.environments import plan_environment

        plan = plan_environment(
            profile_id,
            root=root,
            python=python,
            cuda_toolkit=cuda_toolkit,
            local_source=local_source,
            build_from_source=build_from_source,
        )
        if not plan.ready:
            errors = "; ".join(
                issue.message for issue in plan.issues if issue.severity == "error"
            )
            raise ValueError(errors or "Environment plan is not ready")
        job_id = uuid.uuid4().hex
        job: dict[str, object] = {
            "id": job_id,
            "profile": profile_id,
            "status": "queued",
            "build_from_source": build_from_source,
            "cuda_toolkit_root": (
                str(plan.cuda_toolkit_root) if plan.cuda_toolkit_root is not None else None
            ),
            "local_source": (
                str(plan.local_source) if plan.local_source is not None else None
            ),
            "target": str(plan.target),
            "created_at": time.time(),
            "finished_at": None,
            "report": None,
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
        threading.Thread(target=self._run_create, args=(job_id, plan), daemon=True).start()
        return self.get(job_id)

    def _run_create(self, job_id: str, plan: object) -> None:
        from ..optimizations.environments import check_environment, synchronize_environment

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
            synchronize_environment(plan, runner=runner)  # type: ignore[arg-type]
            report = check_environment(plan)  # type: ignore[arg-type]
        except Exception as exc:
            with self._lock:
                job = self._jobs[job_id]
                job["status"] = "failed"
                job["error"] = str(exc)
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


class GodzillaCalibrationJobManager:
    """Run an explicitly confirmed Godzilla TriAttention preparation job."""

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
        hf_model: str | None = None,
        n_tokens: int = 2048,
        device: str = "cuda",
    ) -> dict[str, object]:
        from ..integration.godzilla_workspace import plan_godzilla_triattention

        plan = plan_godzilla_triattention(
            checkout,
            gguf,
            output=output,
            python=python,
            calibrator=calibrator,
            hf_model=hf_model,
            n_tokens=n_tokens,
            device=device,
        )
        if not plan.ready:
            errors = "; ".join(
                issue.message for issue in plan.issues if issue.severity == "error"
            )
            raise ValueError(errors or "Godzilla calibration plan is not ready")

        job_id = uuid.uuid4().hex
        job: dict[str, object] = {
            "id": job_id,
            "type": "godzilla-triattention",
            "status": "queued",
            "checkout": str(plan.checkout),
            "model": str(plan.gguf),
            "output": str(plan.output),
            "created_at": time.time(),
            "finished_at": None,
            "report": None,
            "error": None,
            "log": deque(maxlen=500),
        }
        with self._lock:
            if any(
                item["output"] == str(plan.output)
                and item["status"] in {"queued", "running"}
                for item in self._jobs.values()
            ):
                raise RuntimeError(
                    f"A Godzilla calibration job for {plan.output} is already running"
                )
            self._jobs[job_id] = job
        threading.Thread(target=self._run, args=(job_id, plan), daemon=True).start()
        return self.get(job_id)

    def _run(self, job_id: str, plan: object) -> None:
        from ..integration.godzilla_workspace import run_godzilla_triattention

        with self._lock:
            self._jobs[job_id]["status"] = "running"

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
            report = run_godzilla_triattention(plan, runner=runner)  # type: ignore[arg-type]
        except Exception as exc:
            with self._lock:
                job = self._jobs[job_id]
                job["status"] = "failed"
                job["error"] = str(exc)
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
