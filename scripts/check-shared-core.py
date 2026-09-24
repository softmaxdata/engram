#!/usr/bin/env python3
"""Check selected shared contracts, not whole-core equality across editions."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import tempfile
import tomllib
from pathlib import Path

MANIFEST = Path("shared-core-manifest.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(checkout: Path, manifest: dict, *, check_hashes: bool = True) -> list[str]:
    errors = []
    project = tomllib.loads((checkout / "pyproject.toml").read_text())["project"]["name"]
    if project not in manifest["editions"]:
        errors.append(f"{checkout}: unrecognized distribution {project!r}")
    for entry in manifest["identical_files"]:
        path = checkout / entry["path"]
        if not path.is_file():
            errors.append(f"{checkout}: missing shared file {entry['path']}")
        elif check_hashes and digest(path) != entry["sha256"]:
            errors.append(f"{checkout}: unported/unrecorded drift in {entry['path']}")
    for contract in manifest["contracts"]:
        path = checkout / contract["path"]
        if not path.is_file():
            errors.append(f"{checkout}: missing {contract['id']} at {contract['path']}")
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            errors.append(f"{checkout}: invalid Python in {contract['path']}")
            continue
        functions = {node.name for node in ast.walk(tree)
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for required in contract["tests"]:
            if required not in functions:
                errors.append(f"{checkout}: {contract['id']} missing test {required}")
    return errors


def compare(oss: Path, paid: Path, manifest: dict, *, check_hashes: bool = True) -> list[str]:
    errors = inspect(oss, manifest, check_hashes=check_hashes)
    errors += inspect(paid, manifest, check_hashes=check_hashes)
    for path, expected in [(oss, "engram-contextdb"), (paid, "engram")]:
        name = tomllib.loads((path / "pyproject.toml").read_text())["project"]["name"]
        if name != expected:
            errors.append(f"{path}: expected {expected}, got {name}; arguments are OSS then paid")
    peer_manifest = json.loads((paid / MANIFEST).read_text())
    if peer_manifest != manifest:
        errors.append("Edition manifests differ; port and review the manifest in both checkouts")
    for entry in manifest["identical_files"]:
        left, right = oss / entry["path"], paid / entry["path"]
        if left.is_file() and right.is_file() and left.read_bytes() != right.read_bytes():
            errors.append(f"Cross-edition mismatch: {entry['path']} ({entry['purpose']})")
    return errors


def self_test(oss: Path, paid: Path, manifest: dict) -> None:
    """Prove that a real file drift and a missing contract both fail, in copies."""
    with tempfile.TemporaryDirectory(prefix="engram-port-check-") as temporary:
        paths = {"pyproject.toml", str(MANIFEST)}
        paths.update(item["path"] for item in manifest["identical_files"])
        paths.update(item["path"] for item in manifest["contracts"])
        copies = []
        for source, label in [(oss, "oss"), (paid, "paid")]:
            target = Path(temporary) / label
            for relative in paths:
                destination = target / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / relative, destination)
            copies.append(target)
        changed = copies[1] / manifest["identical_files"][0]["path"]
        changed.write_bytes(changed.read_bytes() + b"\n# deliberate drift\n")
        assert any("mismatch" in error for error in compare(*copies, manifest))
        contract = manifest["contracts"][0]
        contract_path = copies[0] / contract["path"]
        contract_path.write_text(contract_path.read_text().replace(
            "def " + contract["tests"][0] + "(", "def removed_contract(", 1
        ))
        assert any("missing test" in error for error in compare(*copies, manifest))
    print("Self-test passed: file drift and missing contract rejected in disposable copies")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oss", type=Path, nargs="?", help="OSS checkout (first of two paths)")
    parser.add_argument("paid", type=Path, nargs="?", help="Paid checkout (second of two paths)")
    parser.add_argument("--checkout", type=Path, help="Check one checkout against pinned manifest")
    parser.add_argument("--refresh", action="store_true",
                        help="After review, refresh hashes in BOTH checkouts; requires equal files")
    parser.add_argument("--self-test", action="store_true",
                        help="Also prove deliberate drift fails using disposable copies")
    args = parser.parse_args()
    pair = args.oss is not None and args.paid is not None
    if bool(args.checkout) == pair or ((args.oss is None) != (args.paid is None)):
        parser.error("provide either --checkout PATH or OSS_PATH PAID_PATH")
    if (args.refresh or args.self_test) and not pair:
        parser.error("--refresh and --self-test require both checkout paths")
    root = (args.checkout or args.oss).resolve()
    manifest = json.loads((root / MANIFEST).read_text())
    if manifest.get("version") != 1:
        parser.error("unsupported shared-core manifest version")
    if pair:
        oss, paid = args.oss.resolve(), args.paid.resolve()
        if oss == paid:
            parser.error("OSS and paid checkout must be different paths")
        errors = compare(oss, paid, manifest, check_hashes=not args.refresh)
    else:
        errors = inspect(root, manifest)
    if errors:
        raise SystemExit("\n".join(errors) + "\nPort the change, run both suites, and record "
                         "provenance before refreshing hashes; see docs/shared-core-release.md.")
    if args.refresh:
        for entry in manifest["identical_files"]:
            entry["sha256"] = digest(oss / entry["path"])
        payload = json.dumps(manifest, indent=2) + "\n"
        for checkout in [oss, paid]:
            (checkout / MANIFEST).write_text(payload)
        print("Refreshed shared-file hashes in both manifests")
    if args.self_test:
        self_test(oss, paid, manifest)
    print(f"Shared-core check passed: {len(manifest['identical_files'])} pinned files, "
          f"{len(manifest['contracts'])} regression contracts")


if __name__ == "__main__":
    main()
