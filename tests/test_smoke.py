"""
tests/test_smoke.py
====================
Smoke tests that can run with ONLY numpy + scikit-learn (no torch/gymnasium).
Verifies that all preprocessing logic is correct before you install the
full dependency set.

Run with: python -m pytest tests/test_smoke.py -v
  OR:     python tests/test_smoke.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# =============================================================================
# Test 1: Lexical Scanner
# =============================================================================

def test_lexical_scanner_clean():
    from src.preprocessing.lexical_scanner import scan
    result = scan("/search?q=laptop+computer", body="", user_agent="Mozilla/5.0")
    assert result.risk_score == 0.0, f"Expected 0 risk, got {result.risk_score}"
    assert result.threat_tags == [], f"Expected no tags, got {result.threat_tags}"
    print("  [PASS] Lexical scanner: clean request")


def test_lexical_scanner_sqli():
    from src.preprocessing.lexical_scanner import scan
    result = scan("/search?q=' OR 1=1--")
    assert "sqli" in result.threat_tags, "Expected sqli tag"
    assert result.risk_score > 0.0, "Expected positive risk score"
    assert result.has_sqli, "Expected has_sqli=True"
    print("  [PASS] Lexical scanner: SQLi detection")


def test_lexical_scanner_xss():
    from src.preprocessing.lexical_scanner import scan
    result = scan("/page", body="<script>alert(document.cookie)</script>")
    assert "xss" in result.threat_tags
    assert result.has_xss
    print("  [PASS] Lexical scanner: XSS detection")


def test_lexical_scanner_path_traversal():
    from src.preprocessing.lexical_scanner import scan
    result = scan("/download?file=../../../etc/passwd")
    assert "path_traversal" in result.threat_tags
    assert result.has_path
    print("  [PASS] Lexical scanner: Path traversal detection")


def test_lexical_scanner_cmd_injection():
    from src.preprocessing.lexical_scanner import scan
    result = scan("/ping?host=127.0.0.1; cat /etc/passwd")
    assert "cmd_injection" in result.threat_tags
    assert result.has_cmd
    print("  [PASS] Lexical scanner: Command injection detection")


def test_lexical_scanner_idempotent_tags():
    """Tagging same attack type twice should not add duplicate tags."""
    from src.preprocessing.lexical_scanner import scan
    result = scan("/?a=' OR 1=1-- &b=UNION SELECT 1,2--")
    assert result.threat_tags.count("sqli") == 1, "Duplicate tags detected"
    print("  [PASS] Lexical scanner: Idempotent tagging")


def test_risk_score_clamped():
    from src.preprocessing.lexical_scanner import ScanResult
    r = ScanResult()
    r.add_risk(0.7)
    r.add_risk(0.7)
    assert r.risk_score == 1.0, f"Risk not clamped: {r.risk_score}"
    print("  [PASS] Risk score clamped to 1.0")


# =============================================================================
# Test 2: HTTP Parser
# =============================================================================

def test_csic_parser_benign():
    from src.preprocessing.http_parser import _parse_raw_http_block
    block = """GET /tienda1/publico/pagar.jsp?B1=Comprar HTTP/1.1
User-Agent: Mozilla/5.0
Host: localhost
Connection: close"""
    req = _parse_raw_http_block(block, "benign", "test")
    assert req is not None
    assert req.method == "GET"
    assert "pagar.jsp" in req.uri
    assert req.label == "benign"
    assert req.user_agent == "Mozilla/5.0"
    print("  [PASS] CSIC parser: benign GET request")


def test_csic_parser_post_with_body():
    from src.preprocessing.http_parser import _parse_raw_http_block
    block = """POST /api/login HTTP/1.1
Host: localhost
Content-Type: application/x-www-form-urlencoded

username=admin&password=secret"""
    req = _parse_raw_http_block(block, "attack", "test")
    assert req.method == "POST"
    assert req.body == "username=admin&password=secret"
    assert req.content_type == "application/x-www-form-urlencoded"
    print("  [PASS] CSIC parser: POST with body")


def test_payload_file_parser(tmp_path):
    import tempfile, os
    from src.preprocessing.http_parser import parse_payload_file

    # Create temp payload file
    payload_file = os.path.join(str(tmp_path), "test_payloads.txt")
    with open(payload_file, "w") as f:
        f.write("' OR 1=1--\n")
        f.write("UNION SELECT 1,2,3--\n")
        f.write("# This is a comment\n")
        f.write("admin'--\n")

    requests = parse_payload_file(payload_file, "sqli", samples_per_payload=2)
    # 3 real payloads × 2 templates = 6
    assert len(requests) == 6, f"Expected 6, got {len(requests)}"
    assert all(r.label == "attack" for r in requests)
    assert all(r.attack_type == "sqli" for r in requests)
    print("  [PASS] Payload parser: 3 payloads × 2 templates = 6 requests")


# =============================================================================
# Test 3: Feature Extractor
# =============================================================================

def test_feature_extractor_state_dim():
    from src.preprocessing.http_parser import ParsedRequest
    from src.preprocessing.feature_extractor import FeatureExtractor, STATE_DIM

    req = ParsedRequest(
        method="GET", uri="/search?q=hello&page=2",
        headers={"user-agent": "Mozilla/5.0"},
        label="benign", attack_type="none", source_dataset="test",
    )
    extractor = FeatureExtractor()
    state = extractor.extract(req)

    assert state.shape == (STATE_DIM,), f"Expected ({STATE_DIM},), got {state.shape}"
    assert state.dtype == np.float32
    print(f"  [PASS] Feature extractor: state shape = {state.shape}")


def test_feature_extractor_values_in_range():
    from src.preprocessing.http_parser import ParsedRequest
    from src.preprocessing.feature_extractor import FeatureExtractor

    req = ParsedRequest(
        method="POST",
        uri="/login?redirect=/home",
        body="username=admin&password=' OR 1=1--",
        headers={"content-type": "application/x-www-form-urlencoded",
                 "user-agent": "sqlmap/1.7"},
        label="attack", attack_type="sqli", source_dataset="test",
    )
    extractor = FeatureExtractor()
    state = extractor.extract(req)

    assert np.all(state >= 0.0), f"Values below 0: {state[state < 0]}"
    assert np.all(state <= 1.0), f"Values above 1: {state[state > 1]}"
    assert state[0] > 0.0,  "Expected positive risk score for SQLi"
    assert state[2] == 1.0, "Expected has_sqli=1"
    print(f"  [PASS] Feature extractor: all values in [0,1], sqli detected")


def test_feature_extractor_rolling_window():
    from src.preprocessing.http_parser import ParsedRequest
    from src.preprocessing.feature_extractor import FeatureExtractor

    extractor = FeatureExtractor(window_size=10)
    req = ParsedRequest(uri="/", label="benign", attack_type="none", source_dataset="t")

    # Simulate 5 attacks out of 10
    for is_attack in [True]*5 + [False]*5:
        extractor.update_window(was_attack=is_attack)

    state = extractor.extract(req)
    assert abs(state[10] - 0.5) < 0.01, f"Expected attack rate ≈ 0.5, got {state[10]}"
    print(f"  [PASS] Feature extractor: rolling attack rate = {state[10]:.2f}")


# =============================================================================
# Test 4: Synthetic data generation
# =============================================================================

def test_synthetic_data_generation():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Import the generate function directly
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "prepare_data",
        os.path.join(os.path.dirname(__file__), "..", "scripts", "prepare_data.py")
    )
    module = importlib.util.load_from_spec = None  # Skip full module load

    from scripts.prepare_data import generate_synthetic_dataset
    requests = generate_synthetic_dataset(n_samples=1000, seed=0)
    labels = [r.label for r in requests]
    attacks = [r for r in requests if r.label == "attack"]
    benign  = [r for r in requests if r.label == "benign"]

    assert len(requests) > 900, "Should have ~1000 samples"
    assert len(attacks) > 0,  "Should have attack samples"
    assert len(benign)  > 0,  "Should have benign samples"
    print(f"  [PASS] Synthetic data: {len(requests)} total, "
          f"{len(attacks)} attack, {len(benign)} benign")


# =============================================================================
# Runner
# =============================================================================

def run_all():
    import tempfile

    tests = [
        test_lexical_scanner_clean,
        test_lexical_scanner_sqli,
        test_lexical_scanner_xss,
        test_lexical_scanner_path_traversal,
        test_lexical_scanner_cmd_injection,
        test_lexical_scanner_idempotent_tags,
        test_risk_score_clamped,
        test_csic_parser_benign,
        test_csic_parser_post_with_body,
        test_feature_extractor_state_dim,
        test_feature_extractor_values_in_range,
        test_feature_extractor_rolling_window,
        test_synthetic_data_generation,
    ]

    print("\n" + "="*50)
    print("  NexusWAF RL — Smoke Tests")
    print("="*50)

    passed = failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        for test_fn in tests:
            try:
                if "tmp_path" in test_fn.__code__.co_varnames:
                    test_fn(tmp)
                else:
                    test_fn()
                passed += 1
            except Exception as e:
                print(f"  [FAIL] {test_fn.__name__}: {e}")
                failed += 1

    print("="*50)
    print(f"  {passed} passed, {failed} failed")
    if failed > 0:
        print("  Fix failures before proceeding to training.")
        sys.exit(1)
    else:
        print("  All smoke tests passed. Run prepare_data.py next.")
    print("="*50 + "\n")


if __name__ == "__main__":
    run_all()
