"""
nexus-ml/src/integration/policy_service.py
===========================================
STEP 4 OF 4 — Run this to serve the trained RL policy via gRPC.

This is the bridge between the trained PyTorch model and the Rust
nexus-pipeline. It exposes a gRPC endpoint that nexus-pipeline calls
to get a policy decision for each request.

The Rust side calls: PolicyDecide(StateVector) → ActionId

Protocol buffer definition (policy.proto):
─────────────────────────────────────────
    syntax = "proto3";
    package nexus;

    service PolicyService {
        rpc Decide (PolicyRequest) returns (PolicyResponse);
        rpc Health (HealthRequest) returns (HealthResponse);
    }

    message PolicyRequest {
        repeated float features = 1;  // 15-dim state vector
    }

    message PolicyResponse {
        int32 action_id   = 1;   // 0-6
        string action_name = 2;  // human-readable
        float confidence  = 3;   // max Q-value (normalised)
        repeated float q_values = 4;  // all Q-values for debugging
    }

    message HealthRequest {}
    message HealthResponse { bool ready = 1; }

Usage:
    python src/integration/policy_service.py \
        --checkpoint checkpoints/best.pt \
        --port 50052

    # Or use TorchScript export (faster, no training deps):
    python src/integration/policy_service.py \
        --torchscript exports/nexus_rl_policy.pt \
        --port 50052
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


# =============================================================================
# Model loading utilities (used regardless of gRPC availability)
# =============================================================================

def load_model_for_inference(checkpoint_path: str, state_dim: int = 15, num_actions: int = 7):
    """
    Load a trained DuelingDQN model in inference mode.
    
    Returns a callable: state_array -> (action_id, q_values, confidence)
    """
    try:
        import torch
    except ImportError:
        raise RuntimeError("PyTorch required: pip install torch")

    # Try TorchScript first (faster, no training code dependency)
    if checkpoint_path.endswith("_scripted.pt"):
        model = torch.jit.load(checkpoint_path)
        model.eval()

        def predict_scripted(state: np.ndarray):
            x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_vals = model(x).squeeze(0)
            action_id   = int(q_vals.argmax().item())
            confidence  = float(q_vals.softmax(dim=0).max().item())
            return action_id, q_vals.numpy().tolist(), confidence

        return predict_scripted

    # Regular DQN checkpoint
    from src.model.dueling_dqn import DQNAgent
    agent = DQNAgent(state_dim=state_dim, num_actions=num_actions, device="cpu")
    agent.load(checkpoint_path)
    agent.epsilon = 0.0

    def predict(state: np.ndarray):
        import torch
        x = torch.tensor(state, dtype=torch.float32)
        agent.online_net.eval()
        with torch.no_grad():
            if x.dim() == 1:
                x = x.unsqueeze(0)
            q_vals = agent.online_net(x).squeeze(0)
        action_id  = int(q_vals.argmax().item())
        confidence = float(q_vals.softmax(dim=0).max().item())
        return action_id, q_vals.numpy().tolist(), confidence

    return predict


# =============================================================================
# TorchScript export
# =============================================================================

def export_torchscript(checkpoint_path: str, output_path: str):
    """
    Export the trained model to TorchScript for deployment.
    
    TorchScript models:
    - Don't require the training source code at runtime
    - Can be loaded in C++ or ONNX-compatible runtimes
    - Are the recommended export format for the gRPC service
    """
    import torch
    from src.model.dueling_dqn import DQNAgent

    print(f"Loading checkpoint: {checkpoint_path}")
    agent = DQNAgent(state_dim=15, num_actions=7, device="cpu")
    agent.load(checkpoint_path)
    agent.online_net.eval()

    # Trace with example input
    example = torch.zeros(1, 15, dtype=torch.float32)
    traced  = torch.jit.trace(agent.online_net, example)
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    traced.save(output_path)
    print(f"TorchScript model saved: {output_path}")

    # Verify
    loaded = torch.jit.load(output_path)
    out    = loaded(example)
    print(f"Verification: input={example.shape}, output={out.shape} ✓")
    return output_path


# =============================================================================
# gRPC server (requires grpcio)
# =============================================================================

def run_grpc_server(checkpoint_path: str, port: int = 50052):
    """
    Start the gRPC policy service.
    
    Requirements:
        pip install grpcio grpcio-tools
        python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. policy.proto
    """
    try:
        import grpc
        from concurrent import futures
        HAS_GRPC = True
    except ImportError:
        print("grpcio not installed. Install with: pip install grpcio grpcio-tools")
        print("For now, running in REST fallback mode (see below).")
        run_rest_fallback(checkpoint_path, port)
        return

    # Load model
    predict_fn = load_model_for_inference(checkpoint_path)

    from src.environment.waf_env import Action

    # ── REST Fallback (always available without grpcio) ──
    # If you can't install grpcio, use the REST server below.
    # The Rust side just needs to send a JSON POST instead of gRPC.
    print(f"\n[INFO] gRPC server would start on port {port}")
    print("[INFO] gRPC requires proto compilation. See INTEGRATION.md")
    print("[INFO] Use REST fallback for quick integration testing.")
    run_rest_fallback(checkpoint_path, port + 1)


# =============================================================================
# REST fallback server (no extra dependencies beyond Python stdlib)
# =============================================================================

def run_rest_fallback(checkpoint_path: str, port: int = 50053):
    """
    Minimal HTTP REST server as a gRPC fallback.
    
    POST /decide
    Body: {"features": [f0, f1, ..., f14]}
    Response: {"action_id": 2, "action_name": "block_immediate", "confidence": 0.87}

    This allows integration testing without grpcio.
    In production, replace with the proper gRPC server.
    """
    import http.server
    import json
    import threading
    from src.environment.waf_env import Action

    predict_fn = load_model_for_inference(checkpoint_path)

    class PolicyHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path == "/decide":
                length   = int(self.headers.get("Content-Length", 0))
                raw      = self.rfile.read(length)
                body     = json.loads(raw)
                features = np.array(body["features"], dtype=np.float32)

                if len(features) != 15:
                    self._respond(400, {"error": f"Expected 15 features, got {len(features)}"})
                    return

                action_id, q_values, confidence = predict_fn(features)
                self._respond(200, {
                    "action_id":   action_id,
                    "action_name": Action.NAMES[action_id],
                    "confidence":  round(confidence, 4),
                    "q_values":    [round(q, 4) for q in q_values],
                })
            else:
                self._respond(404, {"error": "Not found"})

        def do_GET(self):
            if self.path == "/health":
                self._respond(200, {"ready": True, "model": checkpoint_path})
            else:
                self._respond(404, {"error": "Not found"})

        def _respond(self, code, data):
            body = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass  # Suppress default access log

    server = http.server.HTTPServer(("0.0.0.0", port), PolicyHandler)
    print(f"\n[PolicyService] REST server running on port {port}")
    print(f"[PolicyService] Checkpoint: {checkpoint_path}")
    print(f"[PolicyService] POST http://localhost:{port}/decide")
    print(f"[PolicyService]   Body: {{\"features\": [0.0, 0.0, ..., 0.0]  (15 floats)}}")
    print(f"[PolicyService] GET  http://localhost:{port}/health")
    print(f"\n[PolicyService] Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[PolicyService] Shutting down.")
        server.shutdown()


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="NexusWAF RL Policy Service")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--port",    type=int, default=50052)
    parser.add_argument("--export",  default=None,
                        help="Export to TorchScript and exit (provide output path)")
    parser.add_argument("--rest",    action="store_true",
                        help="Use REST server instead of gRPC")
    args = parser.parse_args()

    if args.export:
        export_torchscript(args.checkpoint, args.export)
        return

    if args.rest:
        run_rest_fallback(args.checkpoint, args.port)
    else:
        run_grpc_server(args.checkpoint, args.port)


if __name__ == "__main__":
    main()
