"""Start PitchProbe locally with the pinned private DeBERTa risk checkpoint.

Run from the repo root:
    python .\start_staging_backend.py

The Hugging Face read token is requested with hidden input and exists only in
this process tree's environment. It is never written to .env.local or a file.
This launches the normal start_backend.py launcher and stops if the backend
falls back to NLI instead of loading the requested checkpoint.
"""

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from getpass import getpass
from pathlib import Path
from queue import Empty, Queue
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
MODEL_ID = "divstu/pitchprobe-deberta-claims-verified-v2"
MODEL_REVISION = "5771ca4cdbc6afc05bdb0baec92409c7f02e9be6"
LOCAL_MODEL_DIR = ROOT / "models" / "pitchprobe-deberta-claims-verified-v2"
MIN_MODEL_WEIGHTS_BYTES = 700_000_000
LOAD_TIMEOUT_SECONDS = 1800  # allow a full large-checkpoint download on slower Windows networks
POLL_SECONDS = 2


def validate_private_checkpoint(token: str) -> None:
    """Check token identity, private repo access, pinned revision and labels."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    identity = api.whoami()
    username = identity.get("name", "unknown")
    print(f"[STAGING] Authenticated Hugging Face account: {username}", flush=True)
    print(f"[STAGING] Checking access to private model: {MODEL_ID}", flush=True)

    # First test repository access independently of the pinned SHA. This makes
    # wrong-account/permission errors distinguishable from a missing revision.
    try:
        latest_info = api.model_info(MODEL_ID, token=token)
    except Exception as exc:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", "unknown")
        body = getattr(response, "text", "") or ""
        body = " ".join(body.split())[:700]
        raise RuntimeError(
            f"Account '{username}' could not read {MODEL_ID}. "
            f"HTTP status: {status}. Hub response: {body or str(exc)[:400]}. "
            "Create a READ token while signed into the Hugging Face account "
            "'divstu' (the model owner), or explicitly grant this account "
            "read access to that private model. Do not paste the token into "
            "chat or a command-line argument."
        ) from exc

    if not latest_info.private:
        raise RuntimeError(
            f"{MODEL_ID} is not private. Stop and verify the intended repo."
        )

    try:
        info = api.model_info(
            MODEL_ID,
            revision=MODEL_REVISION,
            token=token,
        )
    except Exception as exc:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", "unknown")
        body = getattr(response, "text", "") or ""
        body = " ".join(body.split())[:700]
        raise RuntimeError(
            f"Repository access worked for account '{username}', but pinned "
            f"revision {MODEL_REVISION} could not be read. HTTP status: {status}. "
            f"Hub response: {body or str(exc)[:400]}. Check the revision SHA "
            "and try again; the model is not activated."
        ) from exc

    if info.sha != MODEL_REVISION:
        raise RuntimeError(
            "Hub did not return the requested model revision. "
            f"Expected {MODEL_REVISION}, received {info.sha}."
        )

    config_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename="config.json",
        revision=MODEL_REVISION,
        token=token,
    )
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    id2label = {
        int(key): str(value).strip().upper().replace("-", "_").replace(" ", "_")
        for key, value in config.get("id2label", {}).items()
    }
    if set(id2label) != {0, 1, 2} or set(id2label.values()) != {
        "LOW", "MODERATE", "HIGH"
    }:
        raise RuntimeError(
            "The pinned model has an unexpected output-label mapping: "
            f"{id2label}. Refusing to start staging."
        )
    print(
        f"[STAGING] Private model access verified: {MODEL_ID}@{MODEL_REVISION}",
        flush=True,
    )


def read_child_output(process: subprocess.Popen, port_queue: Queue) -> None:
    """Forward launcher logs and capture its selected local port."""
    assert process.stdout is not None
    pattern = re.compile(
        r"\[SUCCESS\] PitchProbe backend is healthy at "
        r"http://127\.0\.0\.1:(\d+)"
    )
    for line in iter(process.stdout.readline, ""):
        print(line, end="", flush=True)
        match = pattern.search(line)
        if match:
            port_queue.put(int(match.group(1)))
    process.stdout.close()


def read_health(port: int):
    request = Request(
        f"http://127.0.0.1:{port}/api/health",
        headers={"Connection": "close"},
    )
    try:
        with urlopen(request, timeout=2) as response:
            if response.status != 200:
                return None
            return json.loads(response.read().decode("utf-8"))
    except (OSError, HTTPError, URLError, ValueError, json.JSONDecodeError):
        return None


def stop_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        # stop the launcher and its Uvicorn child, not just the parent process
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main() -> int:
    if not (ROOT / "start_backend.py").exists():
        print("Run this script from a PitchProbe checkout.", file=sys.stderr)
        return 2

    print("PitchProbe local DeBERTa staging setup", flush=True)

    # Prefer a complete local copy restored from Drive. This avoids repeated
    # downloads of the 738 MB weight file through an unstable CDN connection.
    local_checkpoint = False
    if LOCAL_MODEL_DIR.exists():
        weights = LOCAL_MODEL_DIR / "model.safetensors"
        required_files = [
            LOCAL_MODEL_DIR / "config.json",
            weights,
            LOCAL_MODEL_DIR / "tokenizer.json",
            LOCAL_MODEL_DIR / "spm.model",
        ]
        missing = [str(p) for p in required_files if not p.is_file()]
        if missing:
            print(
                "[STAGING ERROR] Local checkpoint folder exists but is incomplete. "
                "Missing: " + ", ".join(missing),
                file=sys.stderr,
            )
            return 2
        if weights.stat().st_size < MIN_MODEL_WEIGHTS_BYTES:
            print(
                "[STAGING ERROR] Local model.safetensors is too small "
                f"({weights.stat().st_size:,} bytes). Expected at least "
                f"{MIN_MODEL_WEIGHTS_BYTES:,} bytes. Restore the complete file; "
                "the backend was not started.",
                file=sys.stderr,
            )
            return 2
        local_checkpoint = True
        classifier_path = str(LOCAL_MODEL_DIR.resolve())
        print(f"[STAGING] Using complete local checkpoint: {classifier_path}", flush=True)
    else:
        print("No local checkpoint found; using the pinned private Hub checkpoint.", flush=True)
        print("Use a Hugging Face READ token scoped to this private model.", flush=True)

        # Windows PowerShell may provide this through a masked popup.
        token = os.environ.pop("PITCHPROBE_HF_TOKEN", "").strip()
        if not token:
            token = getpass("Hugging Face read token (hidden; not saved): ").strip()
        if not token:
            print("No token entered. Nothing started.", file=sys.stderr)
            return 2

        try:
            validate_private_checkpoint(token)
        except Exception as exc:
            print(
                f"[STAGING ERROR] Private checkpoint preflight failed: {exc}",
                file=sys.stderr,
            )
            print(
                "Check token access and pinned revision. Nothing was started.",
                file=sys.stderr,
            )
            return 2

        classifier_path = MODEL_ID

    env = os.environ.copy()
    env["PITCHPROBE_CLASSIFIER_PATH"] = classifier_path
    env["PITCHPROBE_CLASSIFIER_REVISION"] = MODEL_REVISION
    if local_checkpoint:
        # Local-only load; do not use network or require a token.
        env.pop("HF_TOKEN", None)
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    else:
        env["HF_TOKEN"] = token
        env["HF_HUB_DOWNLOAD_TIMEOUT"] = "180"
        env["HF_HUB_DISABLE_XET"] = "0"
        env["HF_XET_HIGH_PERFORMANCE"] = "1"
    # Keep question generation on the existing targeted rules.
    env["PITCHPROBE_USE_FLAN_T5"] = "0"

    # Never write a token to .env.local, the repository, or logs.
    if not local_checkpoint:
        del token

    creationflags = (
        subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    )
    try:
        process = subprocess.Popen(
            [sys.executable, "-u", str(ROOT / "start_backend.py")],
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )
    except OSError as exc:
        print(f"[STAGING ERROR] Could not start backend launcher: {exc}", file=sys.stderr)
        return 2

    port_queue = Queue()
    threading.Thread(
        target=read_child_output,
        args=(process, port_queue),
        daemon=True,
    ).start()

    selected_port = None
    deadline = time.monotonic() + LOAD_TIMEOUT_SECONDS
    print("[STAGING] Starting backend and waiting for the fine-tuned model to load...", flush=True)

    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                print(
                    f"[STAGING ERROR] Backend launcher exited with code {process.returncode}.",
                    file=sys.stderr,
                )
                return int(process.returncode or 1)

            if selected_port is None:
                try:
                    selected_port = port_queue.get(timeout=POLL_SECONDS)
                except Empty:
                    continue

            health = read_health(selected_port)
            if not health or health.get("status") != "ok":
                time.sleep(POLL_SECONDS)
                continue

            if health.get("classifier_loaded"):
                mode = health.get("classifier_mode")
                classifier = health.get("classifier")

                expected_classifier = classifier_path
                if mode != "fine_tuned" or classifier != expected_classifier:
                    print(
                        "[STAGING ERROR] Backend loaded a fallback or unexpected "
                        f"classifier: mode={mode!r}, classifier={classifier!r}. "
                        "Stopping instead of leaving NLI active by mistake.",
                        file=sys.stderr,
                    )
                    stop_process_tree(process)
                    return 3

                print("\\n[STAGING VERIFIED] Fine-tuned classifier is loaded.", flush=True)
                print(f"Classifier: {classifier}", flush=True)
                if local_checkpoint:
                    print(f"Source model revision: {MODEL_ID}@{MODEL_REVISION}", flush=True)
                    print("Weights loaded from the restored local Drive copy.", flush=True)
                else:
                    print(f"Revision: {MODEL_REVISION}", flush=True)
                print(f"Health: http://127.0.0.1:{selected_port}/api/health", flush=True)
                print(
                    "Question generation remains targeted-rule based "
                    "(PITCHPROBE_USE_FLAN_T5=0).",
                    flush=True,
                )
                print(
                    "Keep this terminal open. Open the health URL to confirm "
                    "classifier_mode=fine_tuned; use Ctrl+C to stop.",
                    flush=True,
                )
                process.wait()
                return int(process.returncode or 0)

            time.sleep(POLL_SECONDS)

        print(
            f"[STAGING ERROR] The requested model was not verified within "
            f"{LOAD_TIMEOUT_SECONDS} seconds. Stopping to avoid silently using NLI.",
            file=sys.stderr,
        )
        stop_process_tree(process)
        return 4

    except KeyboardInterrupt:
        print("\n[STAGING] Stopping local backend...", flush=True)
        if process.poll() is None and os.name == "nt":
            try:
                process.send_signal(signal.CTRL_BREAK_EVENT)
                process.wait(timeout=10)
            except Exception:
                stop_process_tree(process)
        else:
            stop_process_tree(process)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
