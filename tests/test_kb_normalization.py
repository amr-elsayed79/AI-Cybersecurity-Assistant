"""Phase 2 Acceptance Tests for Knowledge Base Validation, Normalization, and Chunking."""

import json
from pathlib import Path
import pytest

from app.config import settings
from app.rag.chunking import chunk_vulnerabilities, chunk_vulnerability
from app.rag.ingestion import (
    compute_file_sha256,
    ingest_master_kb,
    load_master_kb,
    validate_raw_structure,
)
from app.rag.normalization import normalize_record, normalize_records


@pytest.fixture(scope="session")
def raw_kb_data():
    """Load the raw JSON once for test comparisons."""
    return load_master_kb(settings.MASTER_KB_PATH)


@pytest.fixture(scope="session")
def raw_records(raw_kb_data):
    """Return raw record dictionaries."""
    return raw_kb_data["records"]


@pytest.fixture(scope="session")
def ingested_data():
    """Run ingestion once for the test session."""
    return ingest_master_kb(settings.MASTER_KB_PATH)


# ==============================================================================
# 1. Byte-for-Byte Checksum and Integrity Tests
# ==============================================================================

def test_kb_file_integrity_baseline():
    """Assert that the Master KB file on disk matches the baseline SHA-256 checksum."""
    current_hash = compute_file_sha256(settings.MASTER_KB_PATH)
    assert current_hash == settings.BASELINE_KB_SHA256, (
        f"Master KB checksum mismatch! Expected {settings.BASELINE_KB_SHA256}, got {current_hash}"
    )


# ==============================================================================
# 2. Structural and Metadata Validation Tests
# ==============================================================================

def test_kb_top_level_structure(raw_kb_data):
    """Validate top-level metadata fields."""
    assert raw_kb_data["dataset_name"] == "AI Security Assistant - Vulnerability Knowledge Base"
    assert raw_kb_data["version"] == "0.2"
    assert raw_kb_data["schema_version"] == "1.0"
    assert raw_kb_data["scope"]["scanner_scenarios"] == 119
    assert raw_kb_data["scope"]["canonical_records"] == 74
    assert raw_kb_data["scope"]["deferred_scenarios_included"] is False


def test_canonical_record_count(raw_records):
    """Assert that exactly 74 canonical records are present."""
    assert len(raw_records) == 74


def test_unique_record_ids(raw_records):
    """Assert that all 74 records have non-empty, unique IDs."""
    ids = [r["id"] for r in raw_records]
    assert len(ids) == 74
    assert len(set(ids)) == 74
    for rid in ids:
        assert isinstance(rid, str) and len(rid.strip()) > 0


def test_required_fields_and_data_types(raw_records):
    """Validate that every record contains the required 28 keys with proper data types."""
    required_keys = {
        "id": str,
        "name": str,
        "aliases": list,
        "category": str,
        "description": str,
        "how_it_works": str,
        "root_cause": list,
        "variants": list,
        "severity": dict,
        "impact": list,
        "risk_factors": list,
        "affected_technologies": list,
        "affected_languages": list,
        "affected_frameworks": list,
        "common_locations": list,
        "detection": dict,
        "prevention": list,
        "remediation": list,
        "secure_coding": dict,
        "common_mistakes": list,
        "cwe": list,
        "owasp": list,
        "related_vulnerabilities": list,
        "comparisons": list,
        "examples": list,
        "references": list,
        "user_questions": list,
        "knowledge_quality": dict,
    }

    for idx, r in enumerate(raw_records):
        for key, expected_type in required_keys.items():
            assert key in r, f"Record {idx} ({r.get('id')}) is missing key: '{key}'"
            assert isinstance(r[key], expected_type), (
                f"Record {idx} ({r.get('id')}) key '{key}' has type {type(r[key])}, expected {expected_type}"
            )


# ==============================================================================
# 3. Normalization Tests for the 5 Affected Records
# ==============================================================================

def test_normalization_sql_injection(raw_records):
    """Verify in-memory normalization of sql-injection."""
    raw = next(r for r in raw_records if r["id"] == "sql-injection")
    norm, was_modified = normalize_record(raw)

    assert was_modified is True
    # CWE restored from raw common_mistakes
    assert norm["cwe"] == ["CWE-89"]
    # OWASP preserved
    assert norm["owasp"] == ["A05:2025 - Injection"]
    # Prevention restored from raw testing_approaches
    assert "Parameterized queries" in norm["prevention"]
    assert "Prepared statements" in norm["prevention"]
    # Remediation restored from raw prevention
    assert "Replace unsafe query construction" in norm["remediation"]
    # Secure coding principles restored from raw remediation
    assert "Keep query structure separate from data" in norm["secure_coding"]["principles"]
    # Common mistakes restored from raw secure_coding.principles
    assert "Relying only on escaping" in norm["common_mistakes"]
    # Locations restored from raw frameworks
    assert "Login" in norm["common_locations"]
    assert "Search" in norm["common_locations"]
    # Frameworks restored from raw languages
    assert "Django" in norm["affected_frameworks"]
    assert "Flask" in norm["affected_frameworks"]
    # Languages restored from raw technologies
    assert "JavaScript" in norm["affected_languages"]
    assert "Python" in norm["affected_languages"]
    # Technologies restored from raw risk_factors
    assert "Web Applications" in norm["affected_technologies"]
    # Risk factors restored from raw impact
    assert "High database privileges" in norm["risk_factors"]
    # Impact is left empty (reported for manual review, zero hallucinations)
    assert norm["impact"] == []


def test_normalization_nosql_injection(raw_records):
    """Verify in-memory normalization of nosql-injection."""
    raw = next(r for r in raw_records if r["id"] == "nosql-injection")
    norm, was_modified = normalize_record(raw)

    assert was_modified is True
    assert norm["cwe"] == ["CWE-943"]
    assert norm["owasp"] == ["A05:2025 - Injection"]
    assert "Allow-list fields" in norm["prevention"]
    assert "Validate types" in norm["prevention"]
    assert "Separate query structure from input" in norm["remediation"]
    assert "Assuming NoSQL cannot have injection" in norm["common_mistakes"]
    assert "Node.js" in norm["affected_frameworks"]
    assert "JavaScript" in norm["affected_languages"]


def test_normalization_os_command_injection(raw_records):
    """Verify in-memory normalization of os-command-injection."""
    raw = next(r for r in raw_records if r["id"] == "os-command-injection")
    norm, was_modified = normalize_record(raw)

    assert was_modified is True
    assert norm["cwe"] == ["CWE-78"]
    assert norm["owasp"] == ["A05:2025 - Injection"]
    assert "Avoid shell invocation" in norm["prevention"]
    assert "Use argument arrays/APIs" in norm["prevention"]
    assert "Replace shell construction" in norm["remediation"]
    assert "Escaping instead of eliminating shell interpretation" in norm["common_mistakes"]


def test_normalization_ssti(raw_records):
    """Verify in-memory normalization of ssti."""
    raw = next(r for r in raw_records if r["id"] == "ssti")
    norm, was_modified = normalize_record(raw)

    assert was_modified is True
    assert norm["cwe"] == ["CWE-1336"]
    assert norm["owasp"] == ["A05:2025 - Injection"]
    assert "Treat user content as data" in norm["prevention"]
    assert "Use fixed templates" in norm["prevention"]
    assert "Replace dynamic template source" in norm["remediation"]
    assert "Assuming output encoding prevents SSTI" in norm["common_mistakes"]
    assert "Jinja2" in norm["affected_frameworks"]


def test_normalization_xxe(raw_records):
    """Verify in-memory normalization of xxe."""
    raw = next(r for r in raw_records if r["id"] == "xxe")
    norm, was_modified = normalize_record(raw)

    assert was_modified is True
    assert norm["cwe"] == ["CWE-611"]
    assert norm["owasp"] == ["A05:2025 - Injection"]
    assert "Disable external entities" in norm["prevention"]
    assert "Disable DTD where possible" in norm["prevention"]
    assert "Harden parser configuration" in norm["remediation"]
    assert "Assuming XML parsing is automatically safe" in norm["common_mistakes"]
    assert "Java" in norm["affected_frameworks"]
    assert "SOAP" in norm["common_locations"]


# ==============================================================================
# 4. Strict Preservation of Unaffected Records (6 to 74)
# ==============================================================================

def test_unaffected_records_strictly_preserved(raw_records):
    """Verify that Records 6 to 74 are 100% identical before and after normalization."""
    for record in raw_records[5:]:
        rid = record["id"]
        norm, was_modified = normalize_record(record)

        assert was_modified is False, f"Record '{rid}' was unexpectedly flagged as modified!"

        # Deep equality check on all fields
        for key in record.keys():
            assert norm[key] == record[key], (
                f"Field '{key}' in unaffected record '{rid}' was modified! Raw: {record[key]}, Norm: {norm[key]}"
            )


def test_normalize_records_bulk_stats(raw_records):
    """Verify bulk normalization statistics."""
    normalized_list, stats = normalize_records(raw_records)

    assert stats["total_records"] == 74
    assert stats["normalized_count"] == 5
    assert stats["unmodified_count"] == 69
    assert set(stats["normalized_ids"]) == {
        "sql-injection",
        "nosql-injection",
        "os-command-injection",
        "ssti",
        "xxe",
    }


# ==============================================================================
# 5. Semantic Chunking and Skipping Tests
# ==============================================================================

def test_semantic_chunking_counts(ingested_data):
    """Verify that chunking produces 588 total chunks and correctly skips empty sections."""
    records, _ = ingested_data
    chunks, stats = chunk_vulnerabilities(records)

    assert stats["total_chunks"] == 588
    assert stats["total_records"] == 74
    assert stats["average_chunks_per_record"] == 7.95

    counts = stats["per_section_counts"]
    assert counts["overview_and_mechanism"] == 74
    assert counts["severity_and_impact"] == 74
    assert counts["locations_and_technologies"] == 74
    assert counts["detection_and_indicators"] == 74
    assert counts["prevention_and_remediation"] == 74
    assert counts["secure_coding_and_mistakes"] == 74
    assert counts["standards_and_references"] == 74

    # 4 records have empty comparisons/examples and were skipped (70 generated)
    assert counts["comparisons_and_examples"] == 70


def test_chunk_skips_empty_comparisons_and_examples(ingested_data):
    """Verify that the 4 records with empty comparisons/examples generate exactly 7 chunks each."""
    records, _ = ingested_data
    records_by_id = {r.id: r for r in records}

    # nosql-injection, os-command-injection, ssti, xxe have no comparisons or examples
    empty_example_ids = ["nosql-injection", "os-command-injection", "ssti", "xxe"]
    for rid in empty_example_ids:
        r = records_by_id[rid]
        r_chunks = chunk_vulnerability(r)
        sections = [c.section for c in r_chunks]
        assert len(r_chunks) == 7
        assert "comparisons_and_examples" not in sections


def test_chunk_metadata_validity(ingested_data):
    """Assert that every generated chunk contains valid, non-empty metadata and content."""
    records, _ = ingested_data
    chunks, _ = chunk_vulnerabilities(records)

    for chunk in chunks:
        assert chunk.chunk_id.startswith(chunk.vulnerability_id)
        assert len(chunk.content.strip()) > 20
        assert chunk.section in {
            "overview_and_mechanism",
            "severity_and_impact",
            "locations_and_technologies",
            "detection_and_indicators",
            "prevention_and_remediation",
            "secure_coding_and_mistakes",
            "standards_and_references",
            "comparisons_and_examples",
        }
        assert isinstance(chunk.cwe, list)
        assert isinstance(chunk.owasp, list)


# ==============================================================================
# 6. Hardened Value Preservation and Source-to-Destination Tracking
# ==============================================================================

def test_exact_source_to_destination_value_preservation_for_all_affected_records(raw_records):
    """Verify that every single source list in the 5 affected records is mapped to its exact destination with zero loss."""
    affected_ids = {"sql-injection", "nosql-injection", "os-command-injection", "ssti", "xxe"}

    for raw in raw_records:
        if raw["id"] not in affected_ids:
            continue

        norm, was_modified = normalize_record(raw)
        assert was_modified is True

        # 1. raw impact -> norm risk_factors
        assert norm["risk_factors"] == raw["impact"], (
            f"Record {raw['id']}: raw impact values were not preserved in risk_factors!"
        )

        # 2. raw risk_factors -> norm affected_technologies
        assert norm["affected_technologies"] == raw["risk_factors"]

        # 3. raw affected_technologies -> norm affected_languages
        assert norm["affected_languages"] == raw["affected_technologies"]

        # 4. raw affected_languages -> norm affected_frameworks
        assert norm["affected_frameworks"] == raw["affected_languages"]

        # 5. raw affected_frameworks -> norm common_locations
        assert norm["common_locations"] == raw["affected_frameworks"]

        # 6. raw common_locations -> norm detection.indicators
        assert norm["detection"]["indicators"] == raw["common_locations"]

        # 7. raw detection.indicators -> norm detection.code_patterns
        assert norm["detection"]["code_patterns"] == raw["detection"]["indicators"]

        # 8. raw detection.code_patterns -> norm detection.testing_approaches
        assert norm["detection"]["testing_approaches"] == raw["detection"]["code_patterns"]

        # 9. raw detection.testing_approaches -> norm prevention
        assert norm["prevention"] == raw["detection"]["testing_approaches"]

        # 10. raw prevention -> norm remediation
        assert norm["remediation"] == raw["prevention"]

        # 11. raw remediation -> norm secure_coding.principles
        assert norm["secure_coding"]["principles"] == raw["remediation"]

        # 12. raw secure_coding.principles -> norm common_mistakes
        assert norm["common_mistakes"] == raw["secure_coding"]["principles"]

        # 13. raw common_mistakes -> norm cwe
        assert norm["cwe"] == raw["common_mistakes"]

        # 14. norm impact is explicitly empty (reported for manual review without artificial synthesis)
        assert norm["impact"] == []

        # 15. Audit notes properly added to knowledge_quality
        notes = norm["knowledge_quality"].get("normalization_notes", [])
        assert any("displaced at source" in n for n in notes), (
            f"Record {raw['id']} is missing normalization audit notes!"
        )


def test_multiset_string_preservation_for_affected_records(raw_records):
    """Mathematical proof: assert that the multiset of all strings across all shifted fields is 100% preserved."""
    affected_ids = {"sql-injection", "nosql-injection", "os-command-injection", "ssti", "xxe"}

    for raw in raw_records:
        if raw["id"] not in affected_ids:
            continue

        norm, _ = normalize_record(raw)

        # Collect all raw strings across the 13 shifted array fields
        raw_strings = (
            raw["impact"]
            + raw["risk_factors"]
            + raw["affected_technologies"]
            + raw["affected_languages"]
            + raw["affected_frameworks"]
            + raw["common_locations"]
            + raw["detection"]["indicators"]
            + raw["detection"]["code_patterns"]
            + raw["detection"]["testing_approaches"]
            + raw["prevention"]
            + raw["remediation"]
            + raw["secure_coding"]["principles"]
            + raw["common_mistakes"]
        )

        # Collect all normalized strings across the corresponding 13 destination fields
        norm_strings = (
            norm["risk_factors"]
            + norm["affected_technologies"]
            + norm["affected_languages"]
            + norm["affected_frameworks"]
            + norm["common_locations"]
            + norm["detection"]["indicators"]
            + norm["detection"]["code_patterns"]
            + norm["detection"]["testing_approaches"]
            + norm["prevention"]
            + norm["remediation"]
            + norm["secure_coding"]["principles"]
            + norm["common_mistakes"]
            + norm["cwe"]
        )

        assert sorted(raw_strings) == sorted(norm_strings), (
            f"String mismatch in {raw['id']}! Elements were lost, added, or corrupted during normalization."
        )


# ==============================================================================
# 7. Detector Guard and Adversarial Isolation Tests
# ==============================================================================

def test_detector_guards_against_synthetic_unrelated_records(raw_records):
    """Verify that unaffected records cannot be falsely identified as displaced, even under simulated perturbations."""
    from app.rag.normalization import is_record_displaced

    # 1. Unaffected real records must never trigger
    for r in raw_records[5:]:
        assert is_record_displaced(r) is False

    # 2. Adversarial test: an unrelated record with a synthetic CWE injected into common_mistakes
    # MUST NOT trigger because of the ID guard
    spoofed_unaffected = dict(raw_records[5])
    spoofed_unaffected["common_mistakes"] = ["CWE-999"]
    spoofed_unaffected["cwe"] = ["A05:2025 - Injection"]
    spoofed_unaffected["references"] = ["MITRE CWE: A05:2025 - Injection"]
    assert is_record_displaced(spoofed_unaffected) is False

    # 3. Adversarial test: an affected ID missing one of the content guard signals MUST NOT trigger
    spoofed_affected = dict(raw_records[0])
    spoofed_affected["common_mistakes"] = ["Genuine mistake, not a CWE"]
    assert is_record_displaced(spoofed_affected) is False


def test_normalization_idempotency(raw_records):
    """Verify that normalizing an already-normalized record is a strict no-op."""
    for raw in raw_records[:5]:
        norm1, was_mod1 = normalize_record(raw)
        assert was_mod1 is True

        # Second pass on the normalized dictionary
        norm2, was_mod2 = normalize_record(norm1)
        assert was_mod2 is False
        assert norm1 == norm2


# ==============================================================================
# 8. Post-Test File Immutability Assertion
# ==============================================================================

def test_source_file_unmodified_after_all_tests():
    """Verify that the Master KB file on disk was NEVER modified during test execution."""
    current_hash = compute_file_sha256(settings.MASTER_KB_PATH)
    assert current_hash == settings.BASELINE_KB_SHA256, (
        "CRITICAL FAILURE: Master KB file was modified on disk during test execution!"
    )

