#!/usr/bin/env python3
"""
HTTP-based WASM MCP Tool Profiling Script

Measures tools that require wasi:http (summarize, fetch) using wasmtime serve.
These tools cannot be measured with stdio transport.

Usage:
  python measure_http_tools.py                    # Measure all HTTP tools
  python measure_http_tools.py --tool summarize_text
  python measure_http_tools.py --tool fetch
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import statistics


# =============================================================================
# Configuration
# =============================================================================

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Default test configurations
DEFAULT_RUNS = 3
DEFAULT_WARMUP_RUNS = 1

# WASM binary locations
WASM_PATH_CANDIDATES = [
    Path.home() / "EdgeAgent/wasm_mcp/target/wasm32-wasip2/release",
    Path.home() / "DDPS/undergraduated/CCGrid-2026/EdgeAgent/EdgeAgent/wasm_mcp/target/wasm32-wasip2/release",
]

WASM_PATH = None
for path in WASM_PATH_CANDIDATES:
    if path.exists():
        WASM_PATH = path
        break

# HTTP tool configurations
HTTP_TOOL_CONFIGS = {
    # ===== summarize (3 tools) =====
    "summarize_text": {
        "server": "summarize",
        "wasm": "mcp_server_summarize.wasm",
        "payload": {"text": "This is a test document. " * 50, "max_length": 100}
    },
    "summarize_documents": {
        "server": "summarize",
        "wasm": "mcp_server_summarize.wasm",
        "payload": {"documents": ["Doc content " * 20], "max_length_per_doc": 100}
    },
    "get_provider_info": {
        "server": "summarize",
        "wasm": "mcp_server_summarize.wasm",
        "payload": {}
    },

    # ===== fetch (1 tool) =====
    "fetch": {
        "server": "fetch",
        "wasm": "mcp_server_fetch.wasm",
        "payload": {"url": "https://httpbin.org/get"}
    },
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TimingResult:
    """Single timing measurement result"""
    run_id: int
    total_ms: float
    server_startup_ms: float = 0.0
    request_ms: float = 0.0


@dataclass
class ToolMeasurement:
    """Complete measurement for a tool"""
    tool_name: str
    node: str
    runs: int
    timestamp: str

    # Timing statistics (ms)
    total_ms: float
    total_std_ms: float
    server_startup_ms: float  # wasmtime serve startup time
    request_ms: float         # HTTP request round-trip time

    measurements: List[float]


# =============================================================================
# Helper Functions
# =============================================================================

def get_node_name() -> str:
    """Detect current node name from hostname"""
    hostname = socket.gethostname().lower()

    if "rpi" in hostname or "raspberry" in hostname:
        return "device-rpi"
    elif "nuc" in hostname:
        return "edge-nuc"
    elif "orin" in hostname or "jetson" in hostname:
        return "edge-orin"
    elif "cloud" in hostname or "aws" in hostname:
        return "cloud"
    else:
        return os.environ.get("NODE_NAME", hostname)


def load_env():
    """Load environment variables from ~/.env if exists"""
    env = os.environ.copy()
    env_file = Path.home() / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env[key] = value
    return env


def find_free_port(start_port: int = 8080) -> int:
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free port found")


def wait_for_server(port: int, timeout: float = 5.0) -> bool:
    """Wait for server to be ready to accept connections"""
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.1)
                s.connect(('127.0.0.1', port))
                return True
        except (socket.error, OSError):
            time.sleep(0.05)
    return False


# =============================================================================
# HTTP Cold Start Measurement
# =============================================================================

def measure_http_cold_start(
    wasm_path: Path,
    tool_name: str,
    payload: Dict[str, Any],
    port: int = 8080
) -> TimingResult:
    """
    Measure cold start using HTTP (wasmtime serve).
    Each measurement starts a new wasmtime serve process.

    Returns:
        TimingResult with:
        - total_ms: total time from process start to response
        - server_startup_ms: time until server accepts connections
        - request_ms: HTTP request round-trip time
    """
    env = load_env()

    # Create JSON-RPC request
    jsonrpc_request = json.dumps({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": 1,
        "params": {
            "name": tool_name,
            "arguments": payload
        }
    })

    total_start = time.perf_counter()

    try:
        # Start wasmtime serve
        proc = subprocess.Popen(
            ["wasmtime", "serve", "-S", "cli", "--addr", f"127.0.0.1:{port}", str(wasm_path)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait for server to start (measure startup time)
        if not wait_for_server(port, timeout=10.0):
            # Capture stderr for debugging
            proc.terminate()
            try:
                _, stderr = proc.communicate(timeout=2)
                print(f"    Server failed to start on port {port}", file=sys.stderr)
                if stderr:
                    print(f"    wasmtime stderr: {stderr.decode()[:500]}", file=sys.stderr)
            except:
                print(f"    Server failed to start on port {port}", file=sys.stderr)
            return TimingResult(run_id=0, total_ms=-1)

        server_startup_ms = (time.perf_counter() - total_start) * 1000

        # Send HTTP request
        request_start = time.perf_counter()

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}",
            data=jsonrpc_request.encode('utf-8'),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                result_body = response.read().decode('utf-8')

                # Check for errors in response
                try:
                    resp_json = json.loads(result_body)
                    if "error" in resp_json:
                        print(f"    Tool error: {resp_json['error']}", file=sys.stderr)
                except json.JSONDecodeError:
                    pass

        except urllib.error.URLError as e:
            print(f"    HTTP error: {e}", file=sys.stderr)
            proc.terminate()
            return TimingResult(run_id=0, total_ms=-1)

        request_ms = (time.perf_counter() - request_start) * 1000
        total_ms = (time.perf_counter() - total_start) * 1000

        # Kill the server
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()

        return TimingResult(
            run_id=0,
            total_ms=total_ms,
            server_startup_ms=server_startup_ms,
            request_ms=request_ms
        )

    except Exception as e:
        print(f"    Error: {e}", file=sys.stderr)
        return TimingResult(run_id=0, total_ms=-1)


def measure_http_cold_start_multiple(
    wasm_path: Path,
    tool_name: str,
    payload: Dict[str, Any],
    runs: int = DEFAULT_RUNS,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    base_port: int = 8080
) -> List[TimingResult]:
    """Run multiple HTTP cold start measurements"""
    results = []

    # Warmup runs
    print(f"    Warmup ({warmup_runs})...", end="", flush=True)
    for i in range(warmup_runs):
        port = find_free_port(base_port + i * 10)
        measure_http_cold_start(wasm_path, tool_name, payload, port=port)
        print(".", end="", flush=True)
    print(" done")

    # Actual measurements
    print(f"    Measuring ({runs})...", end="", flush=True)
    for i in range(runs):
        port = find_free_port(base_port + (warmup_runs + i) * 10)
        result = measure_http_cold_start(wasm_path, tool_name, payload, port=port)
        result.run_id = i + 1
        results.append(result)
        print(".", end="", flush=True)
    print(" done")

    return results


# =============================================================================
# Result Processing
# =============================================================================

def process_results(
    tool_name: str,
    results: List[TimingResult]
) -> Optional[ToolMeasurement]:
    """Process timing results into a ToolMeasurement"""

    # Filter out failed measurements
    valid_results = [r for r in results if r.total_ms > 0]

    if not valid_results:
        return None

    # Calculate statistics
    total_times = [r.total_ms for r in valid_results]
    startup_times = [r.server_startup_ms for r in valid_results]
    request_times = [r.request_ms for r in valid_results]

    return ToolMeasurement(
        tool_name=tool_name,
        node=get_node_name(),
        runs=len(valid_results),
        timestamp=datetime.now().isoformat(),
        total_ms=statistics.mean(total_times),
        total_std_ms=statistics.stdev(total_times) if len(total_times) > 1 else 0.0,
        server_startup_ms=statistics.mean(startup_times),
        request_ms=statistics.mean(request_times),
        measurements=total_times
    )


# =============================================================================
# Main Measurement Logic
# =============================================================================

def run_tool_measurement(
    tool_name: str,
    runs: int = DEFAULT_RUNS
) -> Optional[ToolMeasurement]:
    """Run measurement for a single HTTP tool"""

    if tool_name not in HTTP_TOOL_CONFIGS:
        print(f"Unknown tool: {tool_name}", file=sys.stderr)
        print(f"Available tools: {', '.join(HTTP_TOOL_CONFIGS.keys())}")
        return None

    config = HTTP_TOOL_CONFIGS[tool_name]

    # Get WASM path
    if WASM_PATH is None:
        print("WASM path not found", file=sys.stderr)
        return None

    wasm_file = WASM_PATH / config["wasm"]
    if not wasm_file.exists():
        print(f"WASM file not found: {wasm_file}", file=sys.stderr)
        return None

    payload = config["payload"]
    input_size = len(json.dumps(payload))

    print(f"\n[{tool_name}] Input: {input_size} bytes, Transport: HTTP (wasmtime serve)")

    # Run measurements
    results = measure_http_cold_start_multiple(
        wasm_file, tool_name, payload,
        runs=runs
    )

    # Process results
    measurement = process_results(tool_name, results)

    if measurement:
        print(f"    Total:    {measurement.total_ms:.2f}ms (std: {measurement.total_std_ms:.2f}ms)")
        print(f"    Startup:  {measurement.server_startup_ms:.2f}ms (wasmtime serve)")
        print(f"    Request:  {measurement.request_ms:.2f}ms (HTTP round-trip)")

    return measurement


def save_results(measurements: List[ToolMeasurement], output_file: Path):
    """Save measurements to JSON file"""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    data = [asdict(m) for m in measurements]

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_file}")


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="HTTP-based WASM MCP Tool Profiler (wasmtime serve)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Measure all HTTP tools
  python measure_http_tools.py

  # Measure specific tool
  python measure_http_tools.py --tool summarize_text
  python measure_http_tools.py --tool fetch

  # More runs for better statistics
  python measure_http_tools.py --runs 10
"""
    )

    parser.add_argument("--tool", "-t", type=str,
                        help="Specific tool to measure (default: all)")
    parser.add_argument("--runs", "-r", type=int, default=DEFAULT_RUNS,
                        help=f"Number of measurement runs (default: {DEFAULT_RUNS})")
    parser.add_argument("--output", "-o", type=Path,
                        help="Output JSON file path")
    parser.add_argument("--list-tools", action="store_true",
                        help="List available tools and exit")

    args = parser.parse_args()

    if args.list_tools:
        print("Available HTTP tools:")
        for name, config in HTTP_TOOL_CONFIGS.items():
            print(f"  {name}")
            print(f"    Server: {config['server']}")
            print(f"    WASM: {config['wasm']}")
        return

    print("=" * 60)
    print("HTTP-based WASM MCP Tool Profiler")
    print("=" * 60)
    print(f"Node: {get_node_name()}")
    print(f"WASM Path: {WASM_PATH}")
    print()

    # Determine tools
    if args.tool:
        tools = [args.tool]
    else:
        tools = list(HTTP_TOOL_CONFIGS.keys())

    # Run measurements
    all_measurements = []

    for tool_name in tools:
        measurement = run_tool_measurement(tool_name, runs=args.runs)
        if measurement:
            all_measurements.append(measurement)

    # Save results
    if all_measurements:
        node_name = get_node_name()
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

        if args.output:
            output_file = args.output
        else:
            output_file = RESULTS_DIR / node_name / f"http_measurements_{timestamp_str}.json"

        save_results(all_measurements, output_file)

        # Print summary
        print("\n" + "=" * 60)
        print("Summary (HTTP Cold Start)")
        print("=" * 60)
        print(f"{'Tool':<25} {'Total (ms)':<15} {'Startup (ms)':<15} {'Request (ms)':<15}")
        print("-" * 70)
        for m in all_measurements:
            print(f"{m.tool_name:<25} {m.total_ms:<15.2f} {m.server_startup_ms:<15.2f} {m.request_ms:<15.2f}")
    else:
        print("\nNo measurements collected", file=sys.stderr)


if __name__ == "__main__":
    main()
