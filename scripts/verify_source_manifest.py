#!/usr/bin/env python3
"""Verify external source identities and local artifact checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "source-manifest.json"
REQUIRED_FIELDS = {
    "id",
    "kind",
    "name",
    "origin",
    "release_date",
    "url",
    "sha256",
    "licence",
    "transformation",
    "local_source",
    "local_output",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(
    path: Path = DEFAULT_MANIFEST, *, require_local: bool = True
) -> tuple[list[str], dict[str, object]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    artifacts = document.get("artifacts", [])
    errors: list[str] = []
    verified: list[dict[str, object]] = []
    identifiers: set[str] = set()

    for artifact in artifacts:
        missing = sorted(REQUIRED_FIELDS - set(artifact))
        identifier = str(artifact.get("id", "<missing-id>"))
        if missing:
            errors.append(f"{identifier}: missing fields {missing}")
            continue
        if identifier in identifiers:
            errors.append(f"{identifier}: duplicate artifact id")
        identifiers.add(identifier)

        local_source = ROOT / artifact["local_source"]
        actual_sha256 = None
        if local_source.exists():
            actual_sha256 = file_sha256(local_source)
            if actual_sha256 != artifact["sha256"]:
                errors.append(
                    f"{identifier}: expected {artifact['sha256']}, got {actual_sha256}"
                )
        elif require_local:
            errors.append(f"{identifier}: missing local source {local_source}")

        verified.append(
            {
                "id": identifier,
                "release_date": artifact["release_date"],
                "local_source": artifact["local_source"],
                "sha256": actual_sha256,
                "verified": actual_sha256 == artifact["sha256"],
            }
        )

    manifest_name = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    report = {
        "manifest": manifest_name,
        "artifact_count": len(artifacts),
        "verified_artifacts": verified,
        "errors": errors,
    }
    return errors, report


def verify_tool_versions() -> list[str]:
    errors: list[str] = []
    checks = (
        (
            [
                "docker",
                "run",
                "--rm",
                "-i",
                "--platform",
                "linux/amd64",
                "aero-mesh-avl:3.32",
            ],
            "Version  3.32",
            "AVL 3.32",
            "QUIT\n",
        ),
        (
            [str(ROOT / "scripts" / "run_gmsh_2_7_0.sh"), "-version"],
            "2.7.0",
            "Gmsh 2.7.0",
            None,
        ),
    )
    for command, expected, label, input_text in checks:
        process = subprocess.run(
            command,
            input=input_text,
            capture_output=True,
            text=True,
            check=False,
        )
        output = process.stdout + process.stderr
        if process.returncode != 0:
            errors.append(f"{label}: version command exited {process.returncode}")
        if expected not in output:
            errors.append(f"{label}: version marker {expected!r} was not found")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--require-tools", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "results" / "source_verification.json",
    )
    args = parser.parse_args()

    errors, report = verify_manifest(
        args.manifest, require_local=not args.allow_missing
    )
    if args.require_tools:
        tool_errors = verify_tool_versions()
        errors.extend(tool_errors)
        report["tool_errors"] = tool_errors
    report["errors"] = errors
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
