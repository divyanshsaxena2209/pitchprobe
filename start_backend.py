"""
PitchProbe backend launcher.

Uses the Enterprise Workplace Assistant startup pattern:
- Launch Uvicorn in a child process for each candidate port.
- Verify the child stays alive AND the health endpoint responds.
- If startup fails, advance to the next port and retry.
- Update .env.local only after a healthy backend is confirmed.
- Continue scanning valid TCP ports until successful or stopped with Ctrl+C.

On Windows, also applies the socket.socketpair fallback used in Busted.
"""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
PORT_START = 8000
PORT_MAX = 65535
STARTUP_TIMEOUT_SECONDS = 10
HEALTH_CHECK_INTERVAL_SECONDS = 0.4
SOCKETPAIR_PORT_START = 58000
SOCKETPAIR_PORT_END = 58999


class SocketPairWorkaroundError(RuntimeError):
    pass


def install_windows_socketpair_workaround() -> None:
    """Replace Windows socketpair with a local TCP pair for asyncio."""
    if os.name != "nt":
        return

    original_socketpair = socket.socketpair

    def tcp_socketpair(family=None, type=socket.SOCK_STREAM, proto=0):
        if family not in (None, socket.AF_INET):
            return original_socketpair(family, type, proto)

        last_error = None
        for port in range(SOCKETPAIR_PORT_START, SOCKETPAIR_PORT_END + 1):
            listener = client = accepted = None
            try:
                listener = socket.socket(socket.AF_INET, type, proto)
                if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                    listener.setsockopt(
                        socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1
                    )
                listener.bind(("127.0.0.1", port))
                listener.listen(1)

                client = socket.socket(socket.AF_INET, type, proto)
                client.connect(listener.getsockname())
                accepted, _ = listener.accept()
                accepted.setblocking(False)
                client.setblocking(False)
                accepted.set_inheritable(False)
                client.set_inheritable(False)
                return accepted, client
            except OSError as exc:
                last_error = exc
                if accepted is not None:
                    accepted.close()
                if client is not None:
                    client.close()
            finally:
                if listener is not None:
                    listener.close()

        raise SocketPairWorkaroundError(
            "Windows blocked every TCP socketpair candidate on ports "
            f"{SOCKETPAIR_PORT_START}-{SOCKETPAIR_PORT_END}. Last error: "
            f"{last_error}"
        )

    socket.socketpair = tcp_socketpair


def update_backend_url(port: int) -> None:
    """Preserve other .env.local values and update only BACKEND_URL."""
    env_path = ROOT / ".env.local"
    existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    lines = [
        line for line in existing.splitlines()
        if not line.startswith("BACKEND_URL=")
    ]
    lines.append(f"BACKEND_URL=http://127.0.0.1:{port}")
    env_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def backend_is_ready(port: int) -> bool:
    """Return True only after FastAPI's health endpoint answers successfully."""
    url = f"http://127.0.0.1:{port}/api/health"
    try:
        request = Request(url, headers={"Connection": "close"})
        with urlopen(request, timeout=0.5) as response:
            if response.status != 200:
                return False
            payload = json.loads(response.read().decode("utf-8"))
            return payload.get("status") == "ok"
    except (OSError, URLError, ValueError, json.JSONDecodeError):
        return False


def stop_child(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_server_child(port: int) -> int:
    # Must patch before Uvicorn initializes asyncio's Windows event loop.
    install_windows_socketpair_workaround()

    try:
        import uvicorn
    except ImportError:
        print(
            "Uvicorn is not installed. Install it with: "
            "python -m pip install uvicorn",
            flush=True,
        )
        return 2

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=port,
        log_level="info",
    )
    return 0


def main() -> int:
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    env = os.environ.copy()
    port = PORT_START
    attempts = 0

    while True:
        attempts += 1
        print(f"\n[INFO] Attempt {attempts}: starting backend on port {port}...", flush=True)

        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--serve",
            str(port),
        ]

        try:
            process = subprocess.Popen(command, cwd=ROOT, env=env)
        except OSError as exc:
            print(f"[WARN] Could not launch backend process: {exc}", flush=True)
            port += 1
            if port > PORT_MAX:
                port = PORT_START
                time.sleep(2)
            continue

        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        ready = False

        while time.monotonic() < deadline:
            if process.poll() is not None:
                break
            if backend_is_ready(port):
                ready = True
                break
            time.sleep(HEALTH_CHECK_INTERVAL_SECONDS)

        if ready:
            update_backend_url(port)
            print(f"\n[SUCCESS] PitchProbe backend is healthy at http://127.0.0.1:{port}", flush=True)
            print(f"[INFO] Updated BACKEND_URL in {ROOT / '.env.local'}", flush=True)
            print("[INFO] Keep this terminal open. Start/restart Next.js in another terminal.", flush=True)

            try:
                exit_code = process.wait()
            except KeyboardInterrupt:
                stop_child(process)
                print("\n[INFO] PitchProbe backend stopped by user.", flush=True)
                return 0

            if exit_code == 0:
                print("[INFO] Backend exited normally. Launcher stopped.", flush=True)
                return 0

            print(
                f"[WARN] Backend exited with code {exit_code}. "
                "Trying the next port.",
                flush=True,
            )
        else:
            exit_code = process.poll()
            if exit_code is None:
                stop_child(process)
                reason = f"health endpoint did not respond within {STARTUP_TIMEOUT_SECONDS}s"
            else:
                reason = f"backend process exited during startup (code {exit_code})"

            print(f"[WARN] Port {port} failed: {reason}", flush=True)
            print("[INFO] Incrementing port and retrying.", flush=True)

        port += 1
        if port > PORT_MAX:
            print(
                f"[WARN] Reached port {PORT_MAX}; starting another scan at "
                f"{PORT_START} after a short pause. Press Ctrl+C to stop.",
                flush=True,
            )
            port = PORT_START
            time.sleep(2)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--serve":
        raise SystemExit(run_server_child(int(sys.argv[2])))

    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[INFO] Launcher stopped by user.", flush=True)
        raise SystemExit(0)
