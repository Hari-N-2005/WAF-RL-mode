"""
nexus-ml/src/preprocessing/feature_extractor.py
================================================
Converts ParsedRequest → 15-dimensional state vector.

The feature vector is the exact representation the RL agent receives
as its observation. Every feature here MUST correspond to a field that
the production Rust pipeline can compute from RequestContext at runtime.

State vector index mapping (must stay in sync with config.yaml features):

    Index  Feature                   Source in Rust codebase
    -----  -------                   -----------------------
    0      risk_score                ctx.risk_score (after lexical layer)
    1      threat_tag_count          ctx.threat_tags.len()
    2      has_sqli                  "sqli" in ctx.threat_tags
    3      has_xss                   "xss" in ctx.threat_tags
    4      has_path_traversal        "path_traversal" in ctx.threat_tags
    5      has_cmd_injection         "cmd_injection" in ctx.threat_tags
    6      body_length_norm          ctx.body.len() / max_body_bytes
    7      uri_length_norm           ctx.uri.len() / max_uri_length
    8      query_param_count_norm    ctx.query_params.len() / 20.0
    9      rate_limited              ctx.rate_limited as f32
    10     recent_attack_rate        sliding window: blocks / total (last 100)
    11     p95_latency_norm          prometheus P95 / latency_budget_ms
    12     has_referer               headers.get("referer").is_some() as f32
    13     content_type_encoded      0=none,1=json,2=form,3=multi,4=xml,5=text
    14     grammar_risk              nexus-grammar additional risk delta
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import deque

from .http_parser import ParsedRequest
from .lexical_scanner import scan as lexical_scan, grammar_scan


# =============================================================================
# Constants — must match config.yaml
# =============================================================================

STATE_DIM            = 15
MAX_BODY_BYTES       = 8192
MAX_URI_LENGTH       = 2048
MAX_QUERY_PARAMS     = 20.0
MAX_THREAT_TAGS      = 10.0
RECENT_WINDOW_SIZE   = 100    # rolling window for attack rate computation

# Content type encoding map
_CONTENT_TYPE_MAP = {
    "application/json":                 1,
    "application/x-www-form-urlencoded": 2,
    "multipart/form-data":              3,
    "application/xml":                  4,
    "text/xml":                         4,
    "text/plain":                       5,
}


# =============================================================================
# Stateful feature extractor
# =============================================================================

class FeatureExtractor:
    """
    Converts a ParsedRequest into a normalised numpy state vector.

    Maintains a sliding window of recent decisions to compute the
    `recent_attack_rate` feature, mirroring what the production
    Prometheus metrics window provides.

    Usage:
        extractor = FeatureExtractor()
        state = extractor.extract(request)
        extractor.update_window(was_attack=True)
    """

    def __init__(
        self,
        max_body_bytes: int = MAX_BODY_BYTES,
        max_uri_length: int = MAX_URI_LENGTH,
        window_size: int = RECENT_WINDOW_SIZE,
        simulated_latency_norm: float = 0.1,  # Simulated P95 latency (0=fast, 1=overloaded)
    ):
        self.max_body_bytes          = max_body_bytes
        self.max_uri_length          = max_uri_length
        self.simulated_latency_norm  = simulated_latency_norm
        self._recent_window: deque   = deque(maxlen=window_size)
        self._rate_limited           = False

    def set_rate_limited(self, value: bool):
        """Call this if the rate limiter layer has triggered for the current request."""
        self._rate_limited = value

    def set_latency_norm(self, value: float):
        """Update the simulated P95 latency (0.0 = under budget, 1.0 = at limit)."""
        self.simulated_latency_norm = max(0.0, min(1.0, value))

    def update_window(self, was_attack: bool):
        """
        Call after each request is processed to update the rolling attack rate.
        was_attack=True if the request was an actual attack (ground truth).
        """
        self._recent_window.append(1 if was_attack else 0)

    @property
    def recent_attack_rate(self) -> float:
        if not self._recent_window:
            return 0.0
        return sum(self._recent_window) / len(self._recent_window)

    def extract(self, req: ParsedRequest) -> np.ndarray:
        """
        Extract the 15-dimensional state vector from a ParsedRequest.

        Returns:
            np.ndarray of shape (STATE_DIM,), dtype float32, all values in [0, 1]
        """
        # Run lexical scanner (mirrors nexus-lexical layer)
        scan_result = lexical_scan(
            uri=req.uri,
            body=req.body,
            user_agent=req.user_agent,
            referer=req.referer,
        )

        # Run grammar heuristics (mirrors nexus-grammar layer)
        grammar_risk = grammar_scan(req.body, req.uri)

        # Total risk from both layers (clamped)
        total_risk = min(scan_result.risk_score + grammar_risk, 1.0)

        # Content type encoding
        ct_encoded = self._encode_content_type(req.content_type)

        # Normalised body length
        body_norm = min(req.body_length / self.max_body_bytes, 1.0)

        # Normalised URI length
        uri_norm = min(len(req.uri) / self.max_uri_length, 1.0)

        # Normalised query param count
        param_norm = min(req.query_param_count / MAX_QUERY_PARAMS, 1.0)

        # Normalised threat tag count
        tag_count_norm = min(len(scan_result.threat_tags) / MAX_THREAT_TAGS, 1.0)

        # Build vector
        state = np.array([
            total_risk,                                        # 0
            tag_count_norm,                                    # 1
            float(scan_result.has_sqli),                       # 2
            float(scan_result.has_xss),                        # 3
            float(scan_result.has_path),                       # 4
            float(scan_result.has_cmd),                        # 5
            body_norm,                                         # 6
            uri_norm,                                          # 7
            param_norm,                                        # 8
            float(self._rate_limited),                         # 9
            self.recent_attack_rate,                           # 10
            self.simulated_latency_norm,                       # 11
            float(bool(req.referer)),                          # 12
            ct_encoded / 5.0,                                  # 13  (normalised 0–1)
            min(grammar_risk, 1.0),                            # 14
        ], dtype=np.float32)

        assert state.shape == (STATE_DIM,), f"State dim mismatch: {state.shape}"
        assert np.all((state >= 0.0) & (state <= 1.0)), \
            f"State out of [0,1]: {state}"

        return state

    def _encode_content_type(self, content_type: str) -> float:
        """Map content-type string to integer code 0–5."""
        if not content_type:
            return 0
        ct = content_type.lower().split(";")[0].strip()
        for key, code in _CONTENT_TYPE_MAP.items():
            if ct.startswith(key):
                return code
        return 0


# =============================================================================
# Batch extraction — converts entire dataset to (states, labels) arrays
# =============================================================================

@dataclass
class ExtractedDataset:
    states:       np.ndarray    # shape (N, STATE_DIM)  float32
    labels:       np.ndarray    # shape (N,)             int32  0=benign 1=attack
    attack_types: np.ndarray    # shape (N,)             str    for per-class metrics
    sources:      np.ndarray    # shape (N,)             str    for debugging


_ATTACK_TYPE_TO_INT = {
    "none":             0,
    "other":            0,
    "sqli":             1,
    "xss":              2,
    "path_traversal":   3,
    "cmd_injection":    4,
    "csrf":             5,
}


def extract_dataset(
    requests: List[ParsedRequest],
    latency_schedule: Optional[List[float]] = None,
    verbose: bool = True,
) -> ExtractedDataset:
    """
    Convert a list of ParsedRequest into numpy arrays ready for training.

    The extractor is stateful — it updates the rolling attack window
    per request, simulating the streaming context the agent sees in
    production.

    Args:
        requests:         List of ParsedRequest from the parsers
        latency_schedule: Optional per-request simulated latency norms.
                          If None, randomly sample from [0.0, 0.4] (realistic).
        verbose:          Print progress

    Returns:
        ExtractedDataset with states, labels, attack_types, sources
    """
    extractor = FeatureExtractor()
    n = len(requests)

    states       = np.zeros((n, STATE_DIM), dtype=np.float32)
    labels       = np.zeros(n, dtype=np.int32)
    attack_types = np.empty(n, dtype=object)
    sources      = np.empty(n, dtype=object)

    if latency_schedule is None:
        # Simulate realistic latency: mostly low, occasional spikes
        rng = np.random.default_rng(42)
        latency_schedule = (rng.beta(1.5, 8, size=n) * 0.8).tolist()

    if verbose:
        print(f"Extracting features for {n} requests...")

    for i, req in enumerate(requests):
        extractor.set_latency_norm(latency_schedule[i])
        extractor.set_rate_limited(False)  # Rate limiting not simulated here

        states[i]       = extractor.extract(req)
        labels[i]       = 1 if req.label == "attack" else 0
        attack_types[i] = req.attack_type
        sources[i]      = req.source_dataset

        # Update rolling window with ground truth
        extractor.update_window(was_attack=(req.label == "attack"))

        if verbose and (i + 1) % 5000 == 0:
            print(f"  Processed {i+1}/{n} ({100*(i+1)/n:.1f}%)")

    if verbose:
        n_attack = labels.sum()
        n_benign = n - n_attack
        print(f"Extraction complete: {n_attack} attack, {n_benign} benign")
        _print_attack_type_counts(attack_types, labels)

    return ExtractedDataset(
        states=states,
        labels=labels,
        attack_types=attack_types,
        sources=sources,
    )


def _print_attack_type_counts(attack_types: np.ndarray, labels: np.ndarray):
    from collections import Counter
    attack_mask = labels == 1
    counts = Counter(attack_types[attack_mask])
    print("  Attack type breakdown:")
    for atype, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {atype:<20} {count:>5}")


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    import sys
    from .http_parser import ParsedRequest

    test_requests = [
        ParsedRequest(method="GET", uri="/search?q=hello+world", body="",
                      headers={"user-agent": "Mozilla/5.0"},
                      label="benign", attack_type="none", source_dataset="test"),
        ParsedRequest(method="POST", uri="/login", body="user=admin&pass=' OR 1=1--",
                      headers={"content-type": "application/x-www-form-urlencoded"},
                      label="attack", attack_type="sqli", source_dataset="test"),
        ParsedRequest(method="GET", uri="/page?id=1", body="<script>alert(1)</script>",
                      headers={},
                      label="attack", attack_type="xss", source_dataset="test"),
    ]

    ds = extract_dataset(test_requests, verbose=True)
    print("\nExtracted states:")
    for i, req in enumerate(test_requests):
        print(f"  [{req.label:>6} / {req.attack_type:<15}] risk={ds.states[i][0]:.2f}  "
              f"sqli={ds.states[i][2]:.0f}  xss={ds.states[i][3]:.0f}")

    print("Feature extractor smoke test passed.")
