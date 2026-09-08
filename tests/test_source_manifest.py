import json
from pathlib import Path

from scripts.verify_source_manifest import verify_manifest


def _artifact(release_date: str) -> dict[str, str]:
    return {
        "id": "example",
        "kind": "geometry",
        "name": "Example geometry",
        "origin": "Example archive",
        "release_date": release_date,
        "url": "https://example.com/geometry.dat",
        "sha256": "0" * 64,
        "licence": "Example",
        "transformation": "None",
        "local_source": "vendor/example.dat",
        "local_output": "vendor/example.npz",
    }


def test_manifest_accepts_complete_unique_sources(tmp_path: Path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {"artifacts": [_artifact("2013-05-31")]}
        ),
        encoding="utf-8",
    )
    errors, report = verify_manifest(path, require_local=False)
    assert errors == []
    assert report["artifact_count"] == 1


def test_manifest_rejects_duplicate_identifiers(tmp_path: Path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {"artifacts": [_artifact("2013-05-31"), _artifact("2013-05-31")]}
        ),
        encoding="utf-8",
    )
    errors, _ = verify_manifest(path, require_local=False)
    assert any("duplicate artifact id" in error for error in errors)
