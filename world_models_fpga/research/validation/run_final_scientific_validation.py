"""Assemble a final report only from independently executed experiment records.

This intentionally replaces the historical script that mixed literal scores with
computed values. It does not run missing experiments or infer a conclusion from a
checkpoint; it validates provenance and reports missing evidence as such.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REQUIRED = {"config", "git_commit", "dataset_sha256", "checkpoint_sha256", "seed", "environment", "results"}
REQUIRED_EVALUATIONS = {
    "behavior", "prediction", "necessity", "sufficiency", "generalization",
    "transfer", "compositionality", "architecture_reconstruction", "hardware_equivalence",
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(records_dir: str | Path, output: str | Path) -> dict:
    records = []
    for path in sorted(Path(records_dir).glob("*.json")):
        item = json.loads(path.read_text())
        missing = REQUIRED - set(item)
        if missing:
            raise ValueError(f"{path}: missing provenance fields {sorted(missing)}")
        item["record_path"] = str(path)
        item["record_sha256"] = _digest(path)
        records.append(item)
    observed = {record.get("evaluation") for record in records}
    manifest = {
        "schema_version": "1.0",
        "records": records,
        "executed_evaluations": sorted(value for value in observed if value),
        "missing_evaluations": sorted(REQUIRED_EVALUATIONS - observed),
        "conclusion": "NO SCIENTIFIC CONCLUSION: required independently executed evidence is incomplete" if REQUIRED_EVALUATIONS - observed else "Evidence requires human statistical review; no automatic principle promotion is performed.",
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-dir", required=True, help="Machine-readable results produced by executed experiments")
    parser.add_argument("--output", default=Path(__file__).with_name("experiment_manifest.json"))
    print(json.dumps(build_manifest(parser.parse_args().records_dir, parser.parse_args().output), indent=2))
