from __future__ import annotations

import argparse
import json
import mimetypes
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .dependencies import format_missing_dependency_error, missing_runtime_packages

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
JOB_TTL_SECONDS = 12 * 60 * 60
STATIC_DIR = Path(__file__).parent / "static"
WORK_ROOT = Path(tempfile.gettempdir()) / "demucs_web_jobs"


@dataclass
class Job:
    id: str
    directory: Path
    upload_path: Path
    original_name: str
    status: str = "queued"
    stage: str = "Waiting..."
    progress: float = 0.0
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def vocals_path(self) -> Path:
        return self.directory / "vocals.wav"

    @property
    def instrumental_path(self) -> Path:
        return self.directory / "instrumental.wav"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "stage": self.stage,
            "progress": round(self.progress, 3),
            "error": self.error,
            "originalName": self.original_name,
            "downloads": {
                "vocals": self.status == "complete" and self.vocals_path.exists(),
                "instrumental": self.status == "complete" and self.instrumental_path.exists(),
            },
        }


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        WORK_ROOT.mkdir(parents=True, exist_ok=True)

    def create(self, filename: str, payload: bytes) -> Job:
        self.cleanup_expired()
        job_id = uuid.uuid4().hex
        directory = WORK_ROOT / job_id
        directory.mkdir(parents=True, exist_ok=False)
        upload_path = directory / f"upload{Path(filename).suffix.lower()}"
        upload_path.write_bytes(payload)
        job = Job(id=job_id, directory=directory, upload_path=upload_path, original_name=filename)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **changes) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at = time.time()

    def delete(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if not job:
            return False
        shutil.rmtree(job.directory, ignore_errors=True)
        return True

    def cleanup_expired(self) -> None:
        cutoff = time.time() - JOB_TTL_SECONDS
        with self._lock:
            expired = [job_id for job_id, job in self._jobs.items() if job.updated_at < cutoff]
        for job_id in expired:
            self.delete(job_id)


STORE = JobStore()


class DemucsAppHandler(BaseHTTPRequestHandler):
    server_version = "DemucsApp/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/":
            self._send_static(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        elif path.startswith("/static/"):
            static_path = _static_path(_strip_prefix(path, "/static/"))
            if static_path is None:
                self._send_error(HTTPStatus.NOT_FOUND, "Not found.")
            else:
                self._send_static(static_path)
        elif path == "/api/health":
            self._send_json({"ok": True})
        elif path.startswith("/api/jobs/") and "/download/" in path:
            self._handle_download(path)
        elif path.startswith("/api/jobs/"):
            job_id = _strip_prefix(path, "/api/jobs/").strip("/")
            job = STORE.get(job_id)
            if not job:
                self._send_error(HTTPStatus.NOT_FOUND, "Job not found.")
                return
            self._send_json(job.to_dict())
        else:
            self._send_error(HTTPStatus.NOT_FOUND, "Not found.")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/jobs":
            self._send_error(HTTPStatus.NOT_FOUND, "Not found.")
            return

        content_length = int(self.headers.get("Content-Length") or "0")
        if content_length <= 0:
            self._send_error(HTTPStatus.BAD_REQUEST, "No audio file was uploaded.")
            return
        if content_length > MAX_UPLOAD_BYTES:
            self._send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Audio file is too large.")
            return

        try:
            filename, payload = self._parse_upload(content_length)
        except ValueError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return

        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            self._send_error(HTTPStatus.BAD_REQUEST, "Use an mp3, wav, flac, or m4a audio file.")
            return

        job = STORE.create(filename=_safe_filename(filename), payload=payload)
        threading.Thread(target=_run_job, args=(job.id,), daemon=True).start()
        self._send_json(job.to_dict(), status=HTTPStatus.ACCEPTED)

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/jobs/"):
            self._send_error(HTTPStatus.NOT_FOUND, "Not found.")
            return
        job_id = _strip_prefix(parsed.path, "/api/jobs/").strip("/")
        deleted = STORE.delete(job_id)
        if deleted:
            self._send_json({"ok": True})
        else:
            self._send_error(HTTPStatus.NOT_FOUND, "Job not found.")

    def log_message(self, format: str, *args) -> None:
        return

    def _parse_upload(self, content_length: int) -> tuple[str, bytes]:
        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("multipart/form-data"):
            raise ValueError("Expected a multipart audio upload.")
        body = self.rfile.read(content_length)
        message = BytesParser(policy=policy.default).parsebytes(
            f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + body
        )
        for part in message.iter_parts():
            disposition = part.get("Content-Disposition", "")
            if "form-data" not in disposition:
                continue
            params = dict(part.get_params(header="content-disposition", failobj=[])[1:])
            if params.get("name") != "audio":
                continue
            filename = params.get("filename") or "upload"
            payload = part.get_payload(decode=True) or b""
            if not payload:
                raise ValueError("The uploaded audio file was empty.")
            return filename, payload
        raise ValueError("No audio file field named 'audio' was found.")

    def _handle_download(self, path: str) -> None:
        prefix = "/api/jobs/"
        job_id, _, stem = _strip_prefix(path, prefix).partition("/download/")
        job = STORE.get(job_id)
        if not job:
            self._send_error(HTTPStatus.NOT_FOUND, "Job not found.")
            return
        if job.status != "complete":
            self._send_error(HTTPStatus.CONFLICT, "Files are not ready yet.")
            return
        if stem == "vocals":
            file_path = job.vocals_path
            filename = "vocals.wav"
        elif stem == "instrumental":
            file_path = job.instrumental_path
            filename = "instrumental.wav"
        else:
            self._send_error(HTTPStatus.NOT_FOUND, "Unknown download.")
            return
        if not file_path.exists():
            self._send_error(HTTPStatus.NOT_FOUND, "Output file not found.")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(file_path.stat().st_size))
        self.end_headers()
        with file_path.open("rb") as handle:
            shutil.copyfileobj(handle, self.wfile)

    def _send_static(self, file_path: Path, content_type: str | None = None) -> None:
        if not file_path.exists() or not file_path.is_file():
            self._send_error(HTTPStatus.NOT_FOUND, "Not found.")
            return
        guessed_type = content_type or mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        payload = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", guessed_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status=status)


def _run_job(job_id: str) -> None:
    job = STORE.get(job_id)
    if not job:
        return

    def update(stage: str, progress: float) -> None:
        STORE.update(job_id, status="processing", stage=stage, progress=max(0.0, min(progress, 1.0)))

    try:
        missing = missing_runtime_packages()
        if missing:
            STORE.update(
                job_id,
                status="failed",
                stage="Processing failed.",
                error=format_missing_dependency_error(missing),
                progress=1.0,
            )
            return

        from .pipeline import separate_and_enhance

        STORE.update(job_id, status="processing", stage="Separating audio...", progress=0.03)
        separate_and_enhance(job.upload_path, job.directory, progress=update)
        STORE.update(job_id, status="complete", stage="Ready to download.", progress=1.0)
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required package"
        STORE.update(
            job_id,
            status="failed",
            stage="Processing failed.",
            error=format_missing_dependency_error(missing),
            progress=1.0,
        )
    except Exception as exc:
        STORE.update(job_id, status="failed", stage="Processing failed.", error=str(exc), progress=1.0)
    finally:
        try:
            job.upload_path.unlink(missing_ok=True)
        except OSError:
            pass


def _safe_filename(filename: str) -> str:
    name = Path(filename).name.replace("\x00", "")
    if not name:
        return "upload"
    return name


def _strip_prefix(value: str, prefix: str) -> str:
    if value.startswith(prefix):
        return value[len(prefix):]
    return value


def _static_path(fragment: str) -> Path | None:
    root = STATIC_DIR.resolve()
    candidate = (root / fragment).resolve()
    if candidate == root or root in candidate.parents:
        return candidate
    return None


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    missing = missing_runtime_packages()
    if missing:
        print(format_missing_dependency_error(missing), flush=True)
    httpd = ThreadingHTTPServer((host, port), DemucsAppHandler)
    print(f"Demucs web app running at http://{host}:{port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Demucs vocal remover web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args()
    serve(args.host, args.port)
