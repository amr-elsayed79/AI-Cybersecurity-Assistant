"""Knowledge Base Ingestion and Schema Validation Pipeline."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings
from app.models.schemas import VulnerabilityRecord
from app.rag.normalization import normalize_records


def compute_file_sha256(path: Path) -> str:
    """Compute the SHA-256 checksum of a file in binary mode."""
    if not path.is_file():
        raise FileNotFoundError(f"Knowledge Base file not found at: {path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_kb_file_integrity(path: Path, expected_sha256: Optional[str] = None) -> str:
    """Verify that the Master KB file matches the expected baseline checksum."""
    expected = expected_sha256 or settings.BASELINE_KB_SHA256
    actual = compute_file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"Knowledge Base integrity check failed! Expected SHA256: {expected}, Actual: {actual}"
        )
    return actual


def load_master_kb(path: Path) -> Dict[str, Any]:
    """Read the Master Knowledge Base JSON in strictly read-only mode."""
    if not path.is_file():
        raise FileNotFoundError(f"Master KB not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def validate_raw_structure(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Validate top-level metadata, unique IDs, and record count."""
    required_top_keys = ["dataset_name", "version", "scope", "schema_version", "records"]
    for key in required_top_keys:
        if key not in data:
            raise ValueError(f"Missing required top-level key: {key}")

    records = data.get("records", [])
    if not isinstance(records, list):
        raise TypeError("Key 'records' must be a list.")

    if len(records) != settings.EXPECTED_CANONICAL_RECORDS:
        raise ValueError(
            f"Expected {settings.EXPECTED_CANONICAL_RECORDS} records, found {len(records)}."
        )

    # Validate uniqueness of record IDs
    seen_ids = set()
    for idx, r in enumerate(records):
        rid = r.get("id")
        if not rid:
            raise ValueError(f"Record at index {idx} is missing an 'id'.")
        if rid in seen_ids:
            raise ValueError(f"Duplicate record ID found: {rid}")
        seen_ids.add(rid)

    return records


def ingest_master_kb(
    path: Optional[Path] = None,
) -> Tuple[List[VulnerabilityRecord], Dict[str, Any]]:
    """Complete Phase 2 ingestion workflow:

    1. Verify file integrity (SHA-256).
    2. Load JSON in read-only mode.
    3. Validate raw schema and record count.
    4. Apply conservative in-memory normalization.
    5. Validate normalized records through Pydantic models.
    6. Re-verify file integrity to ensure zero modifications on disk.

    Returns:
        (validated_records, audit_stats)
    """
    kb_path = path or settings.MASTER_KB_PATH

    # Step 1: Pre-ingestion checksum verification
    pre_hash = verify_kb_file_integrity(kb_path)

    # Step 2: Read-only load
    raw_data = load_master_kb(kb_path)

    # Step 3: Validate raw structure
    raw_records = validate_raw_structure(raw_data)

    # Step 4: In-memory normalization (records 1-5 corrected, 6-74 preserved)
    normalized_dicts, stats = normalize_records(raw_records)

    # Step 5: Pydantic schema validation
    validated_records: List[VulnerabilityRecord] = []
    for r_dict in normalized_dicts:
        validated_records.append(VulnerabilityRecord(**r_dict))

    # Step 6: Post-ingestion checksum verification
    post_hash = compute_file_sha256(kb_path)
    if pre_hash != post_hash:
        raise RuntimeError("CRITICAL ERROR: Master KB file was modified on disk during ingestion!")

    stats["sha256_verified"] = post_hash
    stats["validated_record_count"] = len(validated_records)

    return validated_records, stats
