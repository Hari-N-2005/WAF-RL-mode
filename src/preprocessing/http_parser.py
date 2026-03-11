"""
nexus-ml/src/preprocessing/http_parser.py
==========================================
Parsers for every HTTP log/dataset format used in NexusWAF training.

Supports:
  - CSIC 2010 raw HTTP format
  - Apache/Nginx Combined Log Format
  - OWASP Juice Shop access logs (Apache Combined variant)
  - Raw payload files (one payload per line)
  - Synthetic request generation from raw payloads

Each parser returns a list of ParsedRequest dataclasses, which are then
fed into the feature extractor to produce state vectors.
"""

import re
import os
import io
import random
from dataclasses import dataclass, field
from typing import List, Optional, Iterator, Dict
from pathlib import Path


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class ParsedRequest:
    """
    A normalised HTTP request ready for feature extraction.
    
    Fields map directly to what RequestContext holds in the Rust codebase.
    """
    method: str = "GET"
    uri: str = "/"
    http_version: str = "HTTP/1.1"
    headers: Dict[str, str] = field(default_factory=dict)
    body: str = ""
    
    # Ground truth labels
    label: str = "benign"           # "benign" | "attack"
    attack_type: str = "none"       # "none" | "sqli" | "xss" | "path_traversal"
                                    #          "cmd_injection" | "csrf" | "other"
    source_dataset: str = "unknown"

    # Derived convenience fields
    @property
    def user_agent(self) -> str:
        return self.headers.get("user-agent", "")

    @property
    def referer(self) -> str:
        return self.headers.get("referer", "")

    @property
    def content_type(self) -> str:
        return self.headers.get("content-type", "")

    @property
    def query_string(self) -> str:
        if "?" in self.uri:
            return self.uri.split("?", 1)[1]
        return ""

    @property
    def path(self) -> str:
        return self.uri.split("?")[0]

    @property
    def query_param_count(self) -> int:
        qs = self.query_string
        if not qs:
            return 0
        return len([p for p in qs.split("&") if p])

    @property
    def body_length(self) -> int:
        return len(self.body.encode("utf-8", errors="replace"))


# =============================================================================
# CSIC 2010 Parser
# =============================================================================
# The CSIC 2010 dataset stores requests in raw HTTP format,
# separated by blank lines. Each block looks like:
#
#   GET /tienda1/publico/anadir.jsp?... HTTP/1.1
#   User-Agent: Mozilla/5.0
#   Pragma: no-cache
#   Cache-control: no-cache
#   Accept: text/xml...
#   Accept-Language: es
#   Accept-Charset: ISO-8859-1
#   Host: localhost
#   Cookie: ...
#   [blank line]
#   [optional body]
#
# Files: csic_normalTrafico.txt, csic_anomalousTrafico.txt

def parse_csic_file(filepath: str, label: str) -> List[ParsedRequest]:
    """
    Parse a CSIC 2010 HTTP traffic file.

    Args:
        filepath: Path to normalTrafico.txt or anomalousTrafico.txt
        label:    "benign" or "attack"

    Returns:
        List of ParsedRequest objects
    """
    requests = []
    
    try:
        with open(filepath, "r", encoding="latin-1", errors="replace") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"[WARN] CSIC file not found: {filepath}. Skipping.")
        return requests

    # Split into individual request blocks (separated by double newlines)
    # CSIC uses \r\n or \n depending on the version
    blocks = re.split(r"\n\s*\n", content.strip())

    for block in blocks:
        if not block.strip():
            continue
        req = _parse_raw_http_block(block.strip(), label, "csic2010")
        if req is not None:
            # Heuristic: determine attack type from CSIC anomalous requests
            if label == "attack":
                req.attack_type = _infer_attack_type_csic(req)
            requests.append(req)

    return requests


def _parse_raw_http_block(block: str, label: str, source: str) -> Optional[ParsedRequest]:
    """Parse a raw HTTP request block into a ParsedRequest."""
    lines = block.split("\n")
    if not lines:
        return None

    req = ParsedRequest(label=label, source_dataset=source)

    # --- Request line --------------------------------------------------------
    request_line = lines[0].strip()
    parts = request_line.split(" ")
    if len(parts) >= 3:
        req.method = parts[0].upper()
        req.uri = parts[1]
        req.http_version = parts[2]
    elif len(parts) == 2:
        req.method = parts[0].upper()
        req.uri = parts[1]
    else:
        return None  # Malformed, skip

    # --- Headers -------------------------------------------------------------
    body_start = 1
    for i, line in enumerate(lines[1:], start=1):
        line = line.strip()
        if not line:
            body_start = i + 1
            break
        if ":" in line:
            key, _, value = line.partition(":")
            req.headers[key.strip().lower()] = value.strip()

    # --- Body ----------------------------------------------------------------
    if body_start < len(lines):
        req.body = "\n".join(lines[body_start:]).strip()

    return req


# =============================================================================
# CSIC 2010 Kaggle CSV Parser
# =============================================================================
# Columns (tab or comma separated):
#   Method, User-Agent, Pragma, Cache-Control, Accept, Accept-encoding,
#   Accept-charset, language, host, cookie, content-type, connection,
#   lenght, content, classification, URL
#
# classification column values: "Normal" → benign, anything else → attack
# The "content" column is the POST body.
# The "URL" column is the full request URI.

def parse_csic_csv(filepath: str) -> List[ParsedRequest]:
    """
    Parse the CSIC 2010 Kaggle CSV file (csic_database.csv).

    Handles both tab-separated and comma-separated variants.
    The 'classification' column drives the label:
        'Normal' (case-insensitive) → benign
        anything else               → attack

    Args:
        filepath: Path to csic_database.csv

    Returns:
        List of ParsedRequest objects
    """
    requests = []
    
    import csv  # stdlib module

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            if not lines:
                return requests
            
            # Fix CSIC CSV header issue: first column is empty but should be "label"
            header_line = lines[0].strip()
            if header_line.startswith(","):
                header_line = "label," + header_line[1:]  # Add comma after "label"!
            
            # Split header into columns
            header_fields = [h.strip().lower() for h in header_line.split(",")]
            
            # Parse data rows manually to handle complex quoting
            for line_num, line in enumerate(lines[1:], start=2):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Use csv.reader to handle quoted fields properly
                    row_values = next(csv.reader([line]))
                    
                    # Create row dict by zipping headers with values
                    # Handle case where row has different number of fields
                    row = {}
                    for i, field_name in enumerate(header_fields):
                        if i < len(row_values):
                            row[field_name] = row_values[i].strip() if row_values[i] else ""
                        else:
                            row[field_name] = ""
                    
                    # --- Label ---------------------------------------------------
                    classification = row.get("label", row.get("classification", "")).lower()
                    label = "benign" if classification == "normal" else "attack"

                    # --- URI -----------------------------------------------------
                    uri = row.get("url", "/")
                    if not uri:
                        uri = "/"
                    # Some rows store just the path; others include full URL
                    if uri.startswith("http://") or uri.startswith("https://"):
                        from urllib.parse import urlparse
                        parsed_url = urlparse(uri)
                        uri = parsed_url.path
                        if parsed_url.query:
                            uri += "?" + parsed_url.query

                    # --- Headers -------------------------------------------------
                    headers = {}
                    if row.get("user-agent"):
                        headers["user-agent"] = row["user-agent"]
                    if row.get("content-type"):
                        headers["content-type"] = row["content-type"]
                    if row.get("cookie"):
                        headers["cookie"] = row["cookie"]
                    if row.get("host"):
                        headers["host"] = row["host"]
                    if row.get("accept"):
                        headers["accept"] = row["accept"]

                    # --- Body (stored in 'content' column) -----------------------
                    body = row.get("content", "")

                    # --- Method --------------------------------------------------
                    method = row.get("method", "GET").upper()
                    if method not in {"GET", "POST", "PUT", "DELETE", "PATCH",
                                       "HEAD", "OPTIONS"}:
                        method = "GET"

                    req = ParsedRequest(
                        method=method,
                        uri=uri,
                        headers=headers,
                        body=body,
                        label=label,
                        attack_type="none",  # will be inferred below
                        source_dataset="csic2010_csv",
                    )

                    # Infer attack sub-type for attack rows
                    if label == "attack":
                        req.attack_type = _infer_attack_type_csic(req)

                    requests.append(req)
                    
                except Exception as row_error:
                    # Skip malformed rows
                    if line_num <= 10:  # Only warn about first few errors
                        print(f"[WARN] Skipping malformed row {line_num}: {row_error}")
                    continue

    except FileNotFoundError:
        print(f"[WARN] CSIC CSV not found: {filepath}. Skipping.")
        return requests
    except Exception as e:
        import traceback
        print(f"[WRN] Error parsing CSIC CSV: {e}")
        print(f"[WARN] Traceback: {traceback.format_exc()}")
        return requests

    n_attack = sum(1 for r in requests if r.label == "attack")
    n_benign = len(requests) - n_attack
    print(f"  CSIC CSV parsed: {len(requests)} total "
          f"({n_benign} benign, {n_attack} attack)")
    return requests


def _infer_attack_type_csic(req: ParsedRequest) -> str:
    """
    Infer the attack type from a CSIC anomalous request using simple heuristics.
    CSIC doesn't label attack sub-types, so we derive them from content.
    """
    text = req.uri + " " + req.body + " " + req.user_agent
    text_lower = text.lower()

    if any(kw in text_lower for kw in ["union", "select", "insert", "drop", "exec", "sleep(", "--", "' or", '" or']):
        return "sqli"
    if any(kw in text_lower for kw in ["<script", "onerror", "javascript:", "eval(", "alert("]):
        return "xss"
    if re.search(r"\.\./|%2e%2e|\.\.\\", text_lower):
        return "path_traversal"
    if re.search(r";\s*\w|`|\$\(|\bcat\b|\bls\b|\bwhoami\b", text_lower):
        return "cmd_injection"
    return "other"


# =============================================================================
# Apache / Nginx Combined Log Format Parser
# =============================================================================
# Log line format:
#   127.0.0.1 - frank [10/Oct/2000:13:55:36 -0700] "GET /apache_pb.gif HTTP/1.0" 200 2326 "http://ref.com/" "Mozilla/4.08"
#
# Used for: benign traffic class (Apache/Nginx production logs)

_APACHE_LOG_RE = re.compile(
    r'(?P<ip>\S+)\s+'           # client IP
    r'\S+\s+'                   # ident (usually -)
    r'\S+\s+'                   # user (usually -)
    r'\[(?P<time>[^\]]+)\]\s+'  # timestamp
    r'"(?P<request>[^"]+)"\s+'  # request line
    r'(?P<status>\d{3})\s+'     # status code
    r'(?P<bytes>\S+)'           # bytes sent
    r'(?:\s+"(?P<referer>[^"]*)"\s+"(?P<ua>[^"]*)")?'  # optional referer + UA
)


def parse_apache_log_file(filepath: str, label: str = "benign") -> List[ParsedRequest]:
    """
    Parse an Apache/Nginx combined log file.

    Args:
        filepath: Path to access.log file
        label:    Usually "benign" for production logs

    Returns:
        List of ParsedRequest objects
    """
    requests = []

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"[WARN] Apache log file not found: {filepath}. Skipping.")
        return requests

    for line in lines:
        line = line.strip()
        if not line:
            continue

        m = _APACHE_LOG_RE.match(line)
        if not m:
            continue

        request_str = m.group("request")
        parts = request_str.split(" ")
        if len(parts) < 2:
            continue

        req = ParsedRequest(label=label, source_dataset="apache_log")
        req.method = parts[0].upper()
        req.uri = parts[1] if len(parts) > 1 else "/"
        req.http_version = parts[2] if len(parts) > 2 else "HTTP/1.1"

        if m.group("ua"):
            req.headers["user-agent"] = m.group("ua")
        if m.group("referer") and m.group("referer") != "-":
            req.headers["referer"] = m.group("referer")

        # Filter out server errors (5xx) and empty requests — not useful as benign
        status = int(m.group("status"))
        if status >= 500:
            continue

        # Quarantine: if content looks malicious, skip (don't mislabel as benign)
        if _looks_malicious_quick(req.uri):
            continue

        requests.append(req)

    return requests


def _looks_malicious_quick(text: str) -> bool:
    """Quick check to avoid mislabelling obvious attacks as benign."""
    indicators = [
        "union+select", "union%20select", "' or ", "\" or ",
        "<script", "onerror=", "javascript:", "../../../",
        "%2e%2e%2f", "/etc/passwd", "cmd.exe",
    ]
    text_lower = text.lower()
    return any(ind in text_lower for ind in indicators)


# =============================================================================
# Raw Payload File Parser
# =============================================================================
# Payload files (GitHub SecLists, OWASP) contain one raw payload per line.
# We wrap each payload in synthetic HTTP requests to create training samples.

# Synthetic request templates — covers all analysable_text() surfaces
_SYNTHETIC_TEMPLATES = [
    # Payload in GET query parameter (most common)
    lambda p: {
        "method": "GET",
        "uri": f"/search?q={p}&page=1",
        "headers": {
            "host": "target.example.com",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "accept": "text/html,application/xhtml+xml",
        },
        "body": "",
    },
    # Payload in POST body (form-encoded)
    lambda p: {
        "method": "POST",
        "uri": "/login",
        "headers": {
            "host": "target.example.com",
            "content-type": "application/x-www-form-urlencoded",
            "user-agent": "Mozilla/5.0 (compatible; MSIE 10.0)",
        },
        "body": f"username=admin&password={p}&submit=Login",
    },
    # Payload in POST body (JSON)
    lambda p: {
        "method": "POST",
        "uri": "/api/v1/query",
        "headers": {
            "host": "target.example.com",
            "content-type": "application/json",
            "user-agent": "curl/7.68.0",
        },
        "body": f'{{"query": "{p}", "limit": 10}}',
    },
    # Payload in User-Agent header
    lambda p: {
        "method": "GET",
        "uri": "/products?category=electronics",
        "headers": {
            "host": "target.example.com",
            "user-agent": p,
        },
        "body": "",
    },
    # Payload as path segment (for path traversal)
    lambda p: {
        "method": "GET",
        "uri": f"/files/{p}",
        "headers": {
            "host": "target.example.com",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64)",
        },
        "body": "",
    },
]


def parse_payload_file(
    filepath: str,
    attack_type: str,
    samples_per_payload: int = 2,
) -> List[ParsedRequest]:
    """
    Read raw payload file and wrap each payload in synthetic HTTP requests.

    Args:
        filepath:           Path to payload file (one payload per line)
        attack_type:        "sqli" | "xss" | "path_traversal" | "cmd_injection"
        samples_per_payload: Number of different synthetic templates per payload

    Returns:
        List of ParsedRequest objects (all labelled as "attack")
    """
    requests = []

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            payloads = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        print(f"[WARN] Payload file not found: {filepath}. Skipping.")
        return requests

    # For path traversal, prefer path template; for others, use query/body templates
    if attack_type == "path_traversal":
        preferred_templates = [_SYNTHETIC_TEMPLATES[4], _SYNTHETIC_TEMPLATES[0]]
    elif attack_type == "cmd_injection":
        preferred_templates = [_SYNTHETIC_TEMPLATES[1], _SYNTHETIC_TEMPLATES[2]]
    else:
        preferred_templates = _SYNTHETIC_TEMPLATES[:3]

    templates_to_use = preferred_templates[:samples_per_payload]

    for payload in payloads:
        if not payload or len(payload) > 2048:
            continue
        for template_fn in templates_to_use:
            tmpl = template_fn(payload)
            req = ParsedRequest(
                method=tmpl["method"],
                uri=tmpl["uri"],
                headers=tmpl["headers"],
                body=tmpl["body"],
                label="attack",
                attack_type=attack_type,
                source_dataset="synthetic_payload",
            )
            requests.append(req)

    return requests


# =============================================================================
# Dataset loader — loads all sources and returns combined list
# =============================================================================

def load_all_datasets(data_dir: str) -> List[ParsedRequest]:
    """
    Load all available dataset files from data_dir.

    Expected files in data_dir/raw/:
        csic_normalTrafico.txt
        csic_anomalousTrafico.txt
        juice_shop_access.log
        nginx_benign.log
        sqli_payloads.txt
        xss_payloads.txt
        path_traversal.txt
        cmd_injection.txt

    Missing files are skipped with a warning (allows partial datasets).

    Returns:
        Combined list of ParsedRequest from all available sources
    """
    raw_dir = os.path.join(data_dir, "raw")
    all_requests: List[ParsedRequest] = []

    print("=" * 60)
    print("Loading datasets...")
    print("=" * 60)

    # CSIC 2010 -- try Kaggle CSV first, fall back to raw txt files
    csv_path = os.path.join(raw_dir, "csic_database.csv")
    txt_norm  = os.path.join(raw_dir, "csic_normalTrafico.txt")
    txt_anom  = os.path.join(raw_dir, "csic_anomalousTrafico.txt")

    if os.path.exists(csv_path):
        samples = parse_csic_csv(csv_path)
        all_requests.extend(samples)
    elif os.path.exists(txt_norm) or os.path.exists(txt_anom):
        samples = parse_csic_file(txt_norm, "benign")
        print(f"  CSIC Normal (txt):     {len(samples):>6} samples")
        all_requests.extend(samples)
        samples = parse_csic_file(txt_anom, "attack")
        print(f"  CSIC Anomalous (txt):  {len(samples):>6} samples")
        all_requests.extend(samples)
    else:
        print(f"  CSIC 2010:                [NOT FOUND -- skipping]")

    # OWASP Juice Shop logs
    path = os.path.join(raw_dir, "juice_shop_access.log")
    samples = parse_apache_log_file(path, "benign")  # Will be re-labelled below
    # Juice Shop: re-label requests with attack signatures as attacks
    relabelled = 0
    for req in samples:
        if _looks_malicious_quick(req.uri) or _looks_malicious_quick(req.body):
            req.label = "attack"
            req.attack_type = _infer_attack_type_csic(req)
            req.source_dataset = "juice_shop"
            relabelled += 1
        else:
            req.source_dataset = "juice_shop"
    print(f"  Juice Shop logs:       {len(samples):>6} samples ({relabelled} attack, {len(samples)-relabelled} benign)")
    all_requests.extend(samples)

    # Apache/Nginx benign logs
    path = os.path.join(raw_dir, "nginx_benign.log")
    samples = parse_apache_log_file(path, "benign")
    print(f"  Nginx benign logs:     {len(samples):>6} samples")
    all_requests.extend(samples)

    # Payload files — SQLi
    path = os.path.join(raw_dir, "sqli_payloads.txt")
    samples = parse_payload_file(path, "sqli", samples_per_payload=2)
    print(f"  SQLi payloads:         {len(samples):>6} samples (synthetic)")
    all_requests.extend(samples)

    # Payload files — XSS
    path = os.path.join(raw_dir, "xss_payloads.txt")
    samples = parse_payload_file(path, "xss", samples_per_payload=2)
    print(f"  XSS payloads:          {len(samples):>6} samples (synthetic)")
    all_requests.extend(samples)

    # Payload files — Path Traversal
    path = os.path.join(raw_dir, "path_traversal.txt")
    samples = parse_payload_file(path, "path_traversal", samples_per_payload=2)
    print(f"  Path traversal:        {len(samples):>6} samples (synthetic)")
    all_requests.extend(samples)

    # Payload files — Command Injection
    path = os.path.join(raw_dir, "cmd_injection.txt")
    samples = parse_payload_file(path, "cmd_injection", samples_per_payload=2)
    print(f"  Cmd injection:         {len(samples):>6} samples (synthetic)")
    all_requests.extend(samples)

    print("-" * 60)
    benign_count  = sum(1 for r in all_requests if r.label == "benign")
    attack_count  = sum(1 for r in all_requests if r.label == "attack")
    print(f"  TOTAL:                 {len(all_requests):>6} samples")
    print(f"  Benign:                {benign_count:>6} ({100*benign_count/max(len(all_requests),1):.1f}%)")
    print(f"  Attack:                {attack_count:>6} ({100*attack_count/max(len(all_requests),1):.1f}%)")
    print("=" * 60)

    return all_requests


# =============================================================================
# Demo / smoke test
# =============================================================================

if __name__ == "__main__":
    # Demonstrate parsing with a hard-coded CSIC-style example
    example_block = """GET /tienda1/publico/anadir.jsp?id=2&nombre=Jam%F3n+Ib%E9rico&precio=85&cantidad=6&B1=Comprar HTTP/1.1
User-Agent: Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.8
Pragma: no-cache
Cache-control: no-cache
Accept: text/xml,application/xml,application/xhtml+xml
Accept-Language: es
Accept-Charset: iso-8859-1,utf-8
Host: localhost
Connection: close
Cookie: JSESSIONID=48574B02E89CEAFA00DA9DA2"""

    req = _parse_raw_http_block(example_block, "benign", "csic2010")
    print("Parsed CSIC request:")
    print(f"  Method:  {req.method}")
    print(f"  URI:     {req.uri}")
    print(f"  UA:      {req.user_agent}")
    print(f"  Label:   {req.label}")
    print(f"  Params:  {req.query_param_count}")