"""
app/web_server.py
FastAPI backend and static frontend host for the MusicGen dashboard.
"""

import glob
import os
import subprocess
import sys
import threading
import uuid
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

APP_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
SRC_DIR = os.path.abspath(os.path.join(ROOT_DIR, "src"))
WEB_DIR = os.path.abspath(os.path.join(APP_DIR, "web"))
WEB_LOG_DIR = os.path.abspath(os.path.join(ROOT_DIR, "outputs", "web_logs"))
OUTPUTS_DIR = os.path.abspath(os.path.join(ROOT_DIR, "outputs"))
os.makedirs(WEB_LOG_DIR, exist_ok=True)

sys.path.insert(0, SRC_DIR)

from config import MODES, get_config, midi_dirs_for_mode  # noqa: E402


class TrainOptions(BaseModel):
    resume: bool = False
    epochs: Optional[int] = Field(default=None, ge=1)
    batch_size: Optional[int] = Field(default=None, ge=1)
    lr: Optional[float] = Field(default=None, gt=0)


class GenerateOptions(BaseModel):
    num_tokens: Optional[int] = Field(default=None, ge=1)
    temperature: Optional[float] = Field(default=None, gt=0)
    top_k: Optional[int] = Field(default=None, ge=0)
    top_p: Optional[float] = Field(default=None, gt=0, le=1)
    bpm: Optional[int] = Field(default=None, ge=30, le=300)
    seed_length: Optional[int] = Field(default=None, ge=1)
    seed_file: Optional[str] = None
    output_filename: Optional[str] = None


class JobRequest(BaseModel):
    mode: Literal["hiphop", "retro", "mixed"] = "mixed"
    action: Literal["preprocess", "train", "generate", "full_pipeline"]
    train: TrainOptions = Field(default_factory=TrainOptions)
    generate: GenerateOptions = Field(default_factory=GenerateOptions)


class JobSummary(BaseModel):
    id: str
    action: str
    mode: str
    status: str
    created_at: str
    started_at: Optional[str]
    finished_at: Optional[str]


app = FastAPI(title="MusicGen Dashboard API", version="1.0.0")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()
MAX_LOG_LINES = 1200


def _iso_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _build_steps(req: JobRequest) -> List[List[str]]:
    def preprocess_cmd() -> List[str]:
        return [
            sys.executable,
            os.path.join(SRC_DIR, "preprocess.py"),
            "--mode",
            req.mode,
        ]

    def train_cmd() -> List[str]:
        cmd = [sys.executable, os.path.join(SRC_DIR, "train.py"), "--mode", req.mode]
        if req.train.resume:
            cmd.append("--resume")
        if req.train.epochs is not None:
            cmd.extend(["--epochs", str(req.train.epochs)])
        if req.train.batch_size is not None:
            cmd.extend(["--batch_size", str(req.train.batch_size)])
        if req.train.lr is not None:
            cmd.extend(["--lr", str(req.train.lr)])
        return cmd

    def generate_cmd() -> List[str]:
        cmd = [sys.executable, os.path.join(SRC_DIR, "generate.py"), "--mode", req.mode]
        if req.generate.num_tokens is not None:
            cmd.extend(["--num_tokens", str(req.generate.num_tokens)])
        if req.generate.temperature is not None:
            cmd.extend(["--temperature", str(req.generate.temperature)])
        if req.generate.top_k is not None:
            cmd.extend(["--top_k", str(req.generate.top_k)])
        if req.generate.top_p is not None:
            cmd.extend(["--top_p", str(req.generate.top_p)])
        if req.generate.bpm is not None:
            cmd.extend(["--bpm", str(req.generate.bpm)])
        if req.generate.seed_length is not None:
            cmd.extend(["--seed_length", str(req.generate.seed_length)])
        if req.generate.seed_file:
            cmd.extend(["--seed_file", req.generate.seed_file])
        if req.generate.output_filename:
            cmd.extend(["--output_filename", req.generate.output_filename])
        return cmd

    if req.action == "preprocess":
        return [preprocess_cmd()]
    if req.action == "train":
        return [train_cmd()]
    if req.action == "generate":
        return [generate_cmd()]
    return [preprocess_cmd(), train_cmd(), generate_cmd()]


def _job_running_locked() -> bool:
    return any(job["status"] == "running" for job in _jobs.values())


def _append_job_log(job: Dict[str, Any], line: str) -> None:
    safe = line.replace("\r", "")
    if not safe.endswith("\n"):
        safe += "\n"
    job["logs"].append(safe)
    with open(job["log_path"], "a", encoding="utf-8") as f:
        f.write(safe)


def _run_command(job: Dict[str, Any], cmd: List[str], step_name: str) -> int:
    _append_job_log(job, f"\n$ {' '.join(cmd)}")
    _append_job_log(job, f"[dashboard] step={step_name} mode={job['mode']} started")

    proc = subprocess.Popen(
        cmd,
        cwd=ROOT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        universal_newlines=True,
    )
    job["pid"] = proc.pid

    assert proc.stdout is not None
    for line in proc.stdout:
        _append_job_log(job, line)

    rc = proc.wait()
    _append_job_log(job, f"[dashboard] step={step_name} finished rc={rc}")
    return rc


def _worker(job_id: str, steps: List[List[str]]) -> None:
    with _jobs_lock:
        job = _jobs[job_id]
        job["status"] = "running"
        job["started_at"] = _iso_now()
        job["current_step"] = 0

    step_names = [
        "preprocess"
        if "preprocess.py" in s[1]
        else "train"
        if "train.py" in s[1]
        else "generate"
        for s in steps
    ]

    try:
        for idx, cmd in enumerate(steps):
            with _jobs_lock:
                job = _jobs[job_id]
                job["current_step"] = idx + 1
                job["total_steps"] = len(steps)
            rc = _run_command(job, cmd, step_names[idx])
            if rc != 0:
                with _jobs_lock:
                    job = _jobs[job_id]
                    job["status"] = "failed"
                    job["finished_at"] = _iso_now()
                    job["return_code"] = rc
                return

        with _jobs_lock:
            job = _jobs[job_id]
            job["status"] = "succeeded"
            job["finished_at"] = _iso_now()
            job["return_code"] = 0
    except Exception as exc:  # pragma: no cover - defensive path
        with _jobs_lock:
            job = _jobs[job_id]
            job["status"] = "failed"
            job["finished_at"] = _iso_now()
            job["error"] = str(exc)
        _append_job_log(job, f"[dashboard] worker error: {exc}")


def _project_status(mode: str) -> Dict[str, Any]:
    cfg = get_config(mode)
    midi_dirs = midi_dirs_for_mode(mode)

    midi_files: List[str] = []
    for mdir in midi_dirs:
        midi_files.extend(glob.glob(os.path.join(mdir, "*.mid")))
        midi_files.extend(glob.glob(os.path.join(mdir, "*.midi")))

    return {
        "mode": mode,
        "midi_count": len(midi_files),
        "preprocessed": os.path.exists(cfg.processed_path)
        and os.path.exists(cfg.vocab_path),
        "trained": os.path.exists(cfg.best_checkpoint),
        "processed_path": cfg.processed_path,
        "vocab_path": cfg.vocab_path,
        "checkpoint": cfg.best_checkpoint,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "time": _iso_now()}


@app.get("/api/meta")
def meta() -> Dict[str, Any]:
    with _jobs_lock:
        running = _job_running_locked()
    return {
        "modes": list(MODES),
        "actions": ["preprocess", "train", "generate", "full_pipeline"],
        "running": running,
    }


@app.get("/api/status")
def status(
    mode: Literal["hiphop", "retro", "mixed"] = Query(default="mixed"),
) -> Dict[str, Any]:
    return _project_status(mode)


@app.post("/api/jobs", response_model=JobSummary)
def create_job(req: JobRequest) -> JobSummary:
    if req.generate.seed_file and not os.path.exists(req.generate.seed_file):
        raise HTTPException(status_code=400, detail="Seed file does not exist.")

    steps = _build_steps(req)

    with _jobs_lock:
        if _job_running_locked():
            raise HTTPException(
                status_code=409,
                detail="A job is already running. Wait for it to finish before starting another.",
            )

        job_id = uuid.uuid4().hex[:12]
        created = _iso_now()
        log_path = os.path.join(WEB_LOG_DIR, f"{job_id}.log")
        job = {
            "id": job_id,
            "action": req.action,
            "mode": req.mode,
            "status": "queued",
            "created_at": created,
            "started_at": None,
            "finished_at": None,
            "current_step": 0,
            "total_steps": len(steps),
            "logs": deque(maxlen=MAX_LOG_LINES),
            "log_path": log_path,
            "pid": None,
            "return_code": None,
            "error": None,
        }
        _jobs[job_id] = job

    thread = threading.Thread(target=_worker, args=(job_id, steps), daemon=True)
    thread.start()

    return JobSummary(
        id=job_id,
        action=req.action,
        mode=req.mode,
        status="queued",
        created_at=created,
        started_at=None,
        finished_at=None,
    )


@app.get("/api/jobs", response_model=List[JobSummary])
def list_jobs(limit: int = Query(default=20, ge=1, le=200)) -> List[JobSummary]:
    with _jobs_lock:
        items = sorted(_jobs.values(), key=lambda j: j["created_at"], reverse=True)[
            :limit
        ]
    return [
        JobSummary(
            id=j["id"],
            action=j["action"],
            mode=j["mode"],
            status=j["status"],
            created_at=j["created_at"],
            started_at=j["started_at"],
            finished_at=j["finished_at"],
        )
        for j in items
    ]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    with _jobs_lock:
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        job = _jobs[job_id]
        return {
            "id": job["id"],
            "action": job["action"],
            "mode": job["mode"],
            "status": job["status"],
            "created_at": job["created_at"],
            "started_at": job["started_at"],
            "finished_at": job["finished_at"],
            "current_step": job["current_step"],
            "total_steps": job["total_steps"],
            "pid": job["pid"],
            "return_code": job["return_code"],
            "error": job["error"],
            "log_path": job["log_path"],
        }


@app.get("/api/jobs/{job_id}/logs")
def get_job_logs(
    job_id: str, tail: int = Query(default=200, ge=1, le=2000)
) -> Dict[str, Any]:
    with _jobs_lock:
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        job = _jobs[job_id]
        lines = list(job["logs"])[-tail:]
    return {"id": job_id, "lines": lines}


@app.get("/api/outputs")
def list_outputs() -> Dict[str, List[str]]:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    entries: List[tuple] = []
    for pattern in ("*.mid", "*.midi"):
        for path in glob.glob(os.path.join(OUTPUTS_DIR, pattern)):
            if os.path.isfile(path):
                entries.append((os.path.getmtime(path), os.path.basename(path)))

    entries.sort(key=lambda item: item[0], reverse=True)
    files = [name for _, name in entries]
    return {"files": files}


@app.get("/api/outputs/{filename}")
def get_output_file(filename: str) -> FileResponse:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # Only allow file names from the outputs root; reject traversal attempts.
    if filename != os.path.basename(filename):
        raise HTTPException(status_code=400, detail="Invalid filename")

    requested_path = os.path.abspath(os.path.join(OUTPUTS_DIR, filename))
    if os.path.commonpath([OUTPUTS_DIR, requested_path]) != OUTPUTS_DIR:
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not os.path.isfile(requested_path):
        raise HTTPException(status_code=404, detail="Output file not found")

    if not requested_path.lower().endswith((".mid", ".midi")):
        raise HTTPException(status_code=400, detail="Only MIDI files are supported")

    return FileResponse(requested_path, media_type="audio/midi", filename=filename)


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run MusicGen dashboard API server")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="TCP port to bind")
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload (development only)"
    )
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
