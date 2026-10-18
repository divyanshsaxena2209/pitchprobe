"""
PitchProbe backend launcher for Windows.

- Scans ports 8000 through 9000 and skips ports that cannot bind/listen.
- On Windows, provides a local-TCP socketpair fallback for asyncio's self-pipe
  when the platform's default socketpair raises WinError 10013.
- Writes BACKEND_URL to the root .env.local so Next.js proxies to the selected port.
"""

import errno
import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PORT_START = 8000
PORT_END = 9000  # Inclusive
SOCKETPAIR_PORT_START = 58000
SOCKETPAIR_PORT_END = 58999

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")


class SocketPairWorkaroundError(RuntimeError):
    """Raised when a replacement socketpair cannot be created."""


def install_windows_socketpair_workaround() -> None:
    if os.name != "nt":
        return

    original_socketpair = socket.socketpair

    def tcp_socketpair(family=None, type=socket.SOCK_STREAM, proto=0):
        # asyncio on Windows calls socketpair() without specifying a family.
        if family not in (None, socket.AF_INET):
            return original_socketpair(family, type, proto)

        last_error = None
        for port in range(SOCKETPAIR_PORT_START, SOCKETPAIR_PORT_END + 1):
            listener = None
            client = None
            accepted = None
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
            "Windows denied the local TCP sockets needed for asyncio's "
            f"socketpair workaround (ports {SOCKETPAIR_PORT_START}-"
            f"{SOCKETPAIR_PORT_END}). Last error: {last_error}"
        )

    socket.socketpair = tcp_socketpair


def port_is_bindable(port: int):
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        probe.bind(("127.0.0.1", port))
        probe.listen(1)
        return True, None
    except OSError as exc:
        return False, exc
    finally:
        probe.close()


def update_backend_url(port: int) -> None:
    env_path = ROOT / ".env.local"
    existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    lines = [
        line for line in existing.splitlines()
        if not line.startswith("BACKEND_URL=")
    ]
    lines.append(f"BACKEND_URL=http://127.0.0.1:{port}")
    env_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def restore_env_file(original_content, existed_before: bool) -> None:
    env_path = ROOT / ".env.local"
    if existed_before:
        env_path.write_text(original_content, encoding="utf-8")
    elif env_path.exists():
        env_path.unlink()


def main() -> int:
    env_path = ROOT / ".env.local"
    env_existed = env_path.exists()
    original_env = env_path.read_text(encoding="utf-8") if env_existed else ""

    install_windows_socketpair_workaround()

    try:
        import uvicorn
    except ImportError:
        print("Uvicorn is not installed. Install it with: python -m pip install uvicorn")
        return 1

    failed_probes = 0
    printed_probe_errors = 0

    for port in range(PORT_START, PORT_END + 1):
        usable, reason = port_is_bindable(port)
        if not usable:
            failed_probes += 1
            if printed_probe_errors < 3:
                print(f"[INFO] Skipping port {port}: {reason}")
                printed_probe_errors += 1
            continue

        update_backend_url(port)
        print(f"[INFO] Starting PitchProbe backend on http://127.0.0.1:{port}")
        print(f"[INFO] Next.js BACKEND_URL saved to {ROOT / '.env.local'}")
        print("[INFO] Keep this terminal open. Start the frontend in a second terminal.")

        try:
            uvicorn.run(
                "backend.main:app",
                host="127.0.0.1",
                port=port,
                log_level="info",
            )
            return 0
        except SocketPairWorkaroundError as exc:
            restore_env_file(original_env, env_existed)
            print(f"[ERROR] {exc}")
            print(
                "[ERROR] Port scanning cannot fix a Windows policy that blocks "
                "all local listening sockets."
            )
            return 1
        except OSError as exc:
            winerror = getattr(exc, "winerror", None)
            if winerror in (10013, 10048) or exc.errno in (
                errno.EACCES,
                errno.EADDRINUSE,
            ):
                print(
                    f"[WARN] Startup failed on port {port} ({exc}); "
                    "trying the next port."
                )
                continue

            restore_env_file(original_env, env_existed)
            raise

    restore_env_file(original_env, env_existed)
    print(
        f"[ERROR] No usable listening port found between "
        f"{PORT_START} and {PORT_END}. Failed checks: {failed_probes}."
    )
    print(
        "[ERROR] Windows is refusing local listening sockets across this range; "
        "trying more port numbers is unlikely to help."
    )
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[INFO] PitchProbe backend stopped.")
