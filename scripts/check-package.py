#!/usr/bin/env python3
"""Build from an sdist and smoke-test the installed wheel outside the checkout.

Run with the Python interpreter being certified. Requires `build` in that
interpreter; the fresh smoke venv installs its own package and test dependencies.
No provider credentials are inherited and all HTTP traffic is loopback only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.error
import urllib.request
import venv
import xml.etree.ElementTree as ET
from pathlib import Path


def clean_env() -> dict[str, str]:
    # Preserve transport/trust configuration needed by pip, never API credentials.
    names = (
        "PATH", "HOME", "TMPDIR", "TEMP", "TMP", "SYSTEMROOT", "LANG", "LC_ALL",
        "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL", "PIP_TRUSTED_HOST", "PIP_CERT", "PIP_CACHE_DIR",
    )
    env = {name: os.environ[name] for name in names if name in os.environ}
    env.update({
        "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True", "DO_NOT_TRACK": "1",
        "OTEL_SDK_DISABLED": "true",
    })
    return env


def run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def check_cli(executable: Path, work: Path, env: dict[str, str]) -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    cli_env = {
        **env, "ENGRAM_HOST": "127.0.0.1", "ENGRAM_PORT": str(port),
        "ENGRAM_STORAGE_BACKEND": "sqlite",
        "ENGRAM_SQLITE_PATH": str(work / "cli-smoke.sqlite3"),
        "ENGRAM_AUTH_ENABLED": "false",
    }
    with (work / "cli.log").open("w+") as log:
        process = subprocess.Popen([str(executable)], cwd=work, env=cli_env,
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("Installed engram CLI exited before serving /health")
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/health", timeout=1
                    ) as response:
                        assert response.status == 200
                        assert json.load(response)["status"] == "ok"
                    print("Installed engram CLI /health passed", flush=True)
                    break
                except (OSError, urllib.error.URLError):
                    time.sleep(0.2)
            else:
                raise RuntimeError("Installed engram CLI did not become ready in 45 seconds")
        except BaseException:
            log.seek(0)
            print(log.read(), file=sys.stderr)
            raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        help="Copy sdist, wheel, and smoke JUnit report here on success")
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parents[1]
    project = tomllib.loads((checkout / "pyproject.toml").read_text())["project"]
    env = clean_env()
    with tempfile.TemporaryDirectory(prefix="engram-wheel-check-") as temporary:
        work = Path(temporary).resolve()
        if work.is_relative_to(checkout):
            raise RuntimeError("Use a TMPDIR outside the source checkout")
        dist = work / "dist"
        # Explicitly build a source distribution, then a wheel from that archive.
        run([sys.executable, "-m", "build", "--sdist", "--outdir", str(dist),
             str(checkout)], cwd=work, env=env)
        sdists = list(dist.glob("*.tar.gz"))
        assert len(sdists) == 1, sdists
        run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir",
             str(dist), str(sdists[0])], cwd=work, env=env)
        wheels = list(dist.glob("*.whl"))
        assert len(wheels) == 1, wheels
        smoke_env = work / "venv"
        # Match `python -m venv` on POSIX. Managed macOS interpreters may depend
        # on loader paths next to the original executable and cannot be copied.
        venv.EnvBuilder(with_pip=True, symlinks=os.name != "nt").create(smoke_env)
        bindir = smoke_env / ("Scripts" if os.name == "nt" else "bin")
        python = bindir / ("python.exe" if os.name == "nt" else "python")
        # OSS publishes MCP as an extra; explicitly certify that extra.
        extra = "[mcp]" if "mcp" in project.get("optional-dependencies", {}) else ""
        run([str(python), "-m", "pip", "install", str(wheels[0]) + extra,
             "pytest>=8", "pytest-asyncio>=0.24"], cwd=work, env=env)
        run([str(python), "-c", (
            "import pathlib,sys,engram,mcp; "
            "from engram.server.app import create_app; "
            "from engram.storage.sqlite import SQLiteBackend; "
            "from engram.storage.postgres import PostgresBackend; "
            "from engram.integrations.mcp_server import create_mcp_server; "
            "assert pathlib.Path(engram.__file__).resolve().is_relative_to("
            "pathlib.Path(sys.prefix).resolve()), engram.__file__; "
            "print('Installed package imports:', engram.__file__)"
        )], cwd=work, env=env)
        check_cli(bindir / ("engram.exe" if os.name == "nt" else "engram"), work, env)
        # Copy only the test, so pytest cannot import the checkout's package.
        shutil.copy2(checkout / "tests/test_mcp_stdio.py", work / "test_mcp_stdio.py")
        report = work / "mcp-smoke.xml"
        run([str(python), "-m", "pytest", "-q", "-o", "asyncio_mode=auto",
             "--junitxml", str(report), "test_mcp_stdio.py"], cwd=work, env=env)
        cases = ET.parse(report).findall(".//testcase")
        if len(cases) != 6 or any(case.find("skipped") is not None for case in cases):
            raise RuntimeError("Installed MCP smoke must execute all six cases with zero skips")
        if args.output_dir:
            output = args.output_dir.resolve()
            output.mkdir(parents=True, exist_ok=True)
            for artifact in [*sdists, *wheels, report]:
                shutil.copy2(artifact, output / artifact.name)
        print(f"Package check passed: {project['name']} {project['version']}, "
              f"Python {sys.version.split()[0]}, CLI, imports, six MCP cases", flush=True)


if __name__ == "__main__":
    main()
