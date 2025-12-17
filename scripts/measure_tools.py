#!/usr/bin/env python3
"""
WASM MCP Tool Profiling Script

Measures execution time of MCP tools running in WASM environment.
Decomposes timing into: T_cold, T_io, T_serialize, T_compute

Based on Lumos methodology for serverless function profiling.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import statistics


# =============================================================================
# Configuration
# =============================================================================

WASM_DIR = Path(__file__).parent.parent / "wasm_modules"
TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Default test configurations
DEFAULT_RUNS = 10
DEFAULT_WARMUP_RUNS = 3

# Tool definitions with their test configurations
TOOL_CONFIGS = {
    "read_file": {
        "description": "Read file contents",
        "test_sizes": ["1KB", "10KB", "100KB", "1MB", "10MB", "50MB"],
        "params_template": lambda path: {"path": path},
    },
    "read_text_file": {
        "description": "Read text file with encoding",
        "test_sizes": ["1KB", "10KB", "100KB", "1MB", "10MB", "50MB"],
        "params_template": lambda path: {"path": path},
    },
    "read_media_file": {
        "description": "Read media file as base64",
        "test_sizes": ["100KB", "1MB", "5MB", "10MB"],
        "params_template": lambda path: {"path": path},
    },
    "read_multiple_files": {
        "description": "Read multiple files at once",
        "test_sizes": ["3x1KB", "3x100KB", "3x1MB"],
        "params_template": lambda paths: {"paths": paths},
    },
    "write_file": {
        "description": "Write file contents",
        "test_sizes": ["1KB", "10KB", "100KB", "1MB", "10MB"],
        "params_template": lambda path, content: {"path": path, "content": content},
    },
    "list_directory": {
        "description": "List directory contents",
        "test_sizes": ["10files", "100files", "1000files"],
        "params_template": lambda path: {"path": path},
    },
    "list_directory_with_sizes": {
        "description": "List directory with file sizes",
        "test_sizes": ["10files", "100files", "1000files"],
        "params_template": lambda path: {"path": path},
    },
    "directory_tree": {
        "description": "Recursive directory tree",
        "test_sizes": ["shallow", "medium", "deep"],
        "params_template": lambda path: {"path": path, "excludePatterns": []},
    },
    "search_files": {
        "description": "Search files with glob pattern",
        "test_sizes": ["small_dir", "medium_dir", "large_dir"],
        "params_template": lambda path, pattern: {"path": path, "pattern": pattern, "excludePatterns": []},
    },
    "get_file_info": {
        "description": "Get file metadata",
        "test_sizes": ["file", "directory"],
        "params_template": lambda path: {"path": path},
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
    internal_timing: Optional[Dict[str, float]] = None  # From tool response


@dataclass
class ToolMeasurement:
    """Complete measurement for a tool"""
    tool_name: str
    node: str
    mode: str  # "cold" or "warm"
    input_size: int
    input_size_label: str
    output_size: int
    runs: int
    timestamp: str

    # Timing statistics (ms)
    timing: Dict[str, float]  # mean values
    timing_std: Dict[str, float]  # standard deviation
    measurements: List[float]  # raw total_ms values

    # Internal timing if available
    internal_timings: Optional[List[Dict[str, float]]] = None


# =============================================================================
# Helper Functions
# =============================================================================

def get_node_name() -> str:
    """Detect current node name from hostname or environment"""
    hostname = socket.gethostname().lower()

    if "rpi" in hostname or "raspberry" in hostname:
        return "device-rpi"
    elif "nuc" in hostname:
        return "edge-nuc"
    elif "orin" in hostname or "jetson" in hostname:
        return "edge-orin"
    elif "cloud" in hostname or "aws" in hostname or "gcp" in hostname:
        return "cloud"
    else:
        return os.environ.get("NODE_NAME", hostname)


def get_file_size(path: Path) -> int:
    """Get file size in bytes"""
    if path.exists():
        return path.stat().st_size
    return 0


def create_jsonrpc_request(method: str, params: Dict[str, Any], req_id: int = 1) -> str:
    """Create a JSON-RPC 2.0 request"""
    request = {
        "jsonrpc": "2.0",
        "method": method,
        "id": req_id,
        "params": params
    }
    return json.dumps(request)


def create_mcp_tool_call(tool_name: str, arguments: Dict[str, Any], req_id: int = 1) -> str:
    """Create MCP tools/call request"""
    request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": req_id,
        "params": {
            "name": tool_name,
            "arguments": arguments
        }
    }
    return json.dumps(request)


def create_initialize_request() -> str:
    """Create MCP initialize request"""
    return create_jsonrpc_request("initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "wasm-profiler", "version": "1.0.0"}
    }, req_id=0)


def parse_jsonrpc_response(response: str) -> Dict[str, Any]:
    """Parse JSON-RPC response"""
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError as e:
        return {"error": str(e), "raw": response}


# =============================================================================
# Cold Start Measurement (Stdio Mode)
# =============================================================================

def measure_cold_start(
    wasm_path: Path,
    tool_name: str,
    arguments: Dict[str, Any],
    allowed_dir: str = "/tmp",
    timeout: float = 60.0
) -> TimingResult:
    """
    Measure cold start execution time.

    Each measurement creates a new wasmtime process (full cold start).

    Protocol flow:
    1. Send initialize request
    2. Send initialized notification
    3. Send tools/call request
    4. Receive response
    """

    # Prepare requests
    init_request = create_initialize_request()
    init_notification = '{"jsonrpc":"2.0","method":"notifications/initialized"}'
    tool_request = create_mcp_tool_call(tool_name, arguments)

    # Combine all requests (newline separated for stdio)
    full_input = f"{init_request}\n{init_notification}\n{tool_request}\n"

    # Measure execution time
    start_time = time.perf_counter()

    try:
        result = subprocess.run(
            ["wasmtime", "run", f"--dir={allowed_dir}", str(wasm_path)],
            input=full_input,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        end_time = time.perf_counter()
        total_ms = (end_time - start_time) * 1000

        # Parse response for internal timing
        internal_timing = None
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                try:
                    response = json.loads(line)
                    # Check if response contains _timing field
                    if "result" in response and isinstance(response["result"], dict):
                        if "_timing" in response["result"]:
                            internal_timing = response["result"]["_timing"]
                except json.JSONDecodeError:
                    continue

        return TimingResult(
            run_id=0,
            total_ms=total_ms,
            internal_timing=internal_timing
        )

    except subprocess.TimeoutExpired:
        return TimingResult(run_id=0, total_ms=timeout * 1000)
    except Exception as e:
        print(f"Error during cold start measurement: {e}", file=sys.stderr)
        return TimingResult(run_id=0, total_ms=-1)


def measure_cold_start_multiple(
    wasm_path: Path,
    tool_name: str,
    arguments: Dict[str, Any],
    runs: int = DEFAULT_RUNS,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    allowed_dir: str = "/tmp"
) -> List[TimingResult]:
    """Run multiple cold start measurements with warmup"""

    results = []

    # Warmup runs (not counted)
    print(f"  Warmup runs ({warmup_runs})...", end="", flush=True)
    for _ in range(warmup_runs):
        measure_cold_start(wasm_path, tool_name, arguments, allowed_dir)
        print(".", end="", flush=True)
    print(" done")

    # Actual measurements
    print(f"  Measuring ({runs} runs)...", end="", flush=True)
    for i in range(runs):
        result = measure_cold_start(wasm_path, tool_name, arguments, allowed_dir)
        result.run_id = i + 1
        results.append(result)
        print(".", end="", flush=True)
    print(" done")

    return results


# =============================================================================
# Warm Start Measurement (HTTP Mode)
# =============================================================================

def measure_warm_start(
    server_url: str,
    tool_name: str,
    arguments: Dict[str, Any],
    timeout: float = 60.0
) -> TimingResult:
    """
    Measure warm start execution time using HTTP mode.

    The WASM process should already be running via:
    wasmtime serve --addr 127.0.0.1:8000 -S cli=y --dir=/tmp tool-http.wasm
    """
    import urllib.request

    # Create tool call request
    tool_request = create_mcp_tool_call(tool_name, arguments)

    start_time = time.perf_counter()

    try:
        req = urllib.request.Request(
            server_url,
            data=tool_request.encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )

        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = response.read().decode('utf-8')

        end_time = time.perf_counter()
        total_ms = (end_time - start_time) * 1000

        # Parse response for internal timing
        internal_timing = None
        try:
            response_json = json.loads(result)
            if "result" in response_json and isinstance(response_json["result"], dict):
                if "_timing" in response_json["result"]:
                    internal_timing = response_json["result"]["_timing"]
        except json.JSONDecodeError:
            pass

        return TimingResult(
            run_id=0,
            total_ms=total_ms,
            internal_timing=internal_timing
        )

    except Exception as e:
        print(f"Error during warm start measurement: {e}", file=sys.stderr)
        return TimingResult(run_id=0, total_ms=-1)


def measure_warm_start_multiple(
    server_url: str,
    tool_name: str,
    arguments: Dict[str, Any],
    runs: int = DEFAULT_RUNS,
    warmup_runs: int = DEFAULT_WARMUP_RUNS
) -> List[TimingResult]:
    """Run multiple warm start measurements with warmup"""

    results = []

    # Warmup runs (not counted)
    print(f"  Warmup runs ({warmup_runs})...", end="", flush=True)
    for _ in range(warmup_runs):
        measure_warm_start(server_url, tool_name, arguments)
        print(".", end="", flush=True)
    print(" done")

    # Actual measurements
    print(f"  Measuring ({runs} runs)...", end="", flush=True)
    for i in range(runs):
        result = measure_warm_start(server_url, tool_name, arguments)
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
    mode: str,
    input_size: int,
    input_size_label: str,
    output_size: int,
    results: List[TimingResult]
) -> ToolMeasurement:
    """Process timing results into a ToolMeasurement"""

    # Filter out failed measurements
    valid_results = [r for r in results if r.total_ms > 0]

    if not valid_results:
        raise ValueError("No valid measurements")

    # Calculate statistics
    total_times = [r.total_ms for r in valid_results]

    timing = {
        "total_ms": statistics.mean(total_times),
        "cold_start_ms": 0.0,  # Will be calculated from internal timing if available
        "io_ms": 0.0,
        "serialize_ms": 0.0,
        "compute_ms": 0.0,
    }

    timing_std = {
        "total_ms": statistics.stdev(total_times) if len(total_times) > 1 else 0.0,
        "cold_start_ms": 0.0,
        "io_ms": 0.0,
        "serialize_ms": 0.0,
        "compute_ms": 0.0,
    }

    # Process internal timings if available
    internal_timings = []
    for r in valid_results:
        if r.internal_timing:
            internal_timings.append(r.internal_timing)

    if internal_timings:
        # Average internal timings
        keys = internal_timings[0].keys()
        for key in keys:
            values = [t[key] for t in internal_timings if key in t]
            if values:
                # Map internal timing keys to our standard keys
                if key in ["cold_start", "coldStart", "cold_start_ms"]:
                    timing["cold_start_ms"] = statistics.mean(values)
                    timing_std["cold_start_ms"] = statistics.stdev(values) if len(values) > 1 else 0.0
                elif key in ["io", "io_ms", "data_retrieval"]:
                    timing["io_ms"] = statistics.mean(values)
                    timing_std["io_ms"] = statistics.stdev(values) if len(values) > 1 else 0.0
                elif key in ["serialize", "serialize_ms", "serialization"]:
                    timing["serialize_ms"] = statistics.mean(values)
                    timing_std["serialize_ms"] = statistics.stdev(values) if len(values) > 1 else 0.0
                elif key in ["compute", "compute_ms"]:
                    timing["compute_ms"] = statistics.mean(values)
                    timing_std["compute_ms"] = statistics.stdev(values) if len(values) > 1 else 0.0

    return ToolMeasurement(
        tool_name=tool_name,
        node=get_node_name(),
        mode=mode,
        input_size=input_size,
        input_size_label=input_size_label,
        output_size=output_size,
        runs=len(valid_results),
        timestamp=datetime.now().isoformat(),
        timing=timing,
        timing_std=timing_std,
        measurements=total_times,
        internal_timings=internal_timings if internal_timings else None
    )


# =============================================================================
# Test Data Management
# =============================================================================

def get_test_file_path(size_label: str) -> Path:
    """Get path to test file of specified size"""
    return TEST_DATA_DIR / "files" / f"test_{size_label}.txt"


def get_test_image_path(size_label: str) -> Path:
    """Get path to test image of specified size"""
    return TEST_DATA_DIR / "images" / f"test_{size_label}.bin"


def get_test_directory_path(size_label: str) -> Path:
    """Get path to test directory"""
    return TEST_DATA_DIR / "directories" / size_label


# =============================================================================
# Main Measurement Logic
# =============================================================================

def run_tool_measurement(
    tool_name: str,
    input_size_label: str,
    mode: str = "cold",
    runs: int = DEFAULT_RUNS,
    wasm_path: Optional[Path] = None,
    server_url: str = "http://127.0.0.1:8000",
    allowed_dir: str = "/tmp"
) -> Optional[ToolMeasurement]:
    """Run measurement for a single tool with specified input size"""

    if tool_name not in TOOL_CONFIGS:
        print(f"Unknown tool: {tool_name}", file=sys.stderr)
        return None

    config = TOOL_CONFIGS[tool_name]

    # Prepare test arguments based on tool type
    if tool_name in ["read_file", "read_text_file"]:
        test_file = get_test_file_path(input_size_label)
        if not test_file.exists():
            print(f"Test file not found: {test_file}", file=sys.stderr)
            return None
        arguments = config["params_template"](str(test_file))
        input_size = get_file_size(test_file)

    elif tool_name == "read_media_file":
        test_file = get_test_image_path(input_size_label)
        if not test_file.exists():
            print(f"Test file not found: {test_file}", file=sys.stderr)
            return None
        arguments = config["params_template"](str(test_file))
        input_size = get_file_size(test_file)

    elif tool_name == "read_multiple_files":
        # Parse size like "3x1KB" -> 3 files of 1KB each
        parts = input_size_label.split("x")
        count = int(parts[0])
        size = parts[1]
        paths = [str(get_test_file_path(f"{size}_{i}")) for i in range(count)]
        arguments = config["params_template"](paths)
        input_size = sum(get_file_size(Path(p)) for p in paths)

    elif tool_name == "write_file":
        # Generate content of specified size
        size_bytes = parse_size_label(input_size_label)
        content = "x" * size_bytes
        output_path = f"/tmp/wasm_profiler_output_{input_size_label}.txt"
        arguments = config["params_template"](output_path, content)
        input_size = size_bytes

    elif tool_name in ["list_directory", "list_directory_with_sizes", "directory_tree", "search_files"]:
        test_dir = get_test_directory_path(input_size_label)
        if not test_dir.exists():
            print(f"Test directory not found: {test_dir}", file=sys.stderr)
            return None
        if tool_name == "search_files":
            arguments = config["params_template"](str(test_dir), "*.txt")
        else:
            arguments = config["params_template"](str(test_dir))
        input_size = 0  # Directory size not easily measurable

    elif tool_name == "get_file_info":
        if input_size_label == "file":
            test_path = get_test_file_path("1KB")
        else:
            test_path = get_test_directory_path("10files")
        if not test_path.exists():
            print(f"Test path not found: {test_path}", file=sys.stderr)
            return None
        arguments = config["params_template"](str(test_path))
        input_size = 0

    else:
        print(f"Tool {tool_name} not yet implemented for profiling", file=sys.stderr)
        return None

    print(f"\n[{tool_name}] Input: {input_size_label}, Mode: {mode}")

    # Run measurements
    if mode == "cold":
        if wasm_path is None:
            wasm_path = WASM_DIR / "mcp_server_filesystem.wasm"
        if not wasm_path.exists():
            print(f"WASM file not found: {wasm_path}", file=sys.stderr)
            return None
        results = measure_cold_start_multiple(
            wasm_path, tool_name, arguments, runs=runs, allowed_dir=allowed_dir
        )
    else:  # warm
        results = measure_warm_start_multiple(
            server_url, tool_name, arguments, runs=runs
        )

    # Process results
    output_size = 0  # TODO: capture actual output size

    try:
        measurement = process_results(
            tool_name, mode, input_size, input_size_label, output_size, results
        )
        print(f"  Mean: {measurement.timing['total_ms']:.2f}ms (std: {measurement.timing_std['total_ms']:.2f}ms)")
        return measurement
    except ValueError as e:
        print(f"  Error: {e}", file=sys.stderr)
        return None


def parse_size_label(label: str) -> int:
    """Convert size label to bytes"""
    label = label.upper()
    if label.endswith("KB"):
        return int(label[:-2]) * 1024
    elif label.endswith("MB"):
        return int(label[:-2]) * 1024 * 1024
    elif label.endswith("GB"):
        return int(label[:-2]) * 1024 * 1024 * 1024
    elif label.endswith("B"):
        return int(label[:-1])
    else:
        return int(label)


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
        description="WASM MCP Tool Profiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Measure all tools (cold start)
  python measure_tools.py

  # Measure specific tool
  python measure_tools.py --tool read_file

  # Measure warm start (requires HTTP server running)
  python measure_tools.py --mode warm --server-url http://127.0.0.1:8000

  # Measure with specific input size
  python measure_tools.py --tool read_file --size 1MB

  # Custom number of runs
  python measure_tools.py --runs 20
"""
    )

    parser.add_argument("--tool", "-t", type=str,
                        help="Specific tool to measure (default: all)")
    parser.add_argument("--mode", "-m", choices=["cold", "warm", "both"],
                        default="cold", help="Measurement mode")
    parser.add_argument("--size", "-s", type=str,
                        help="Specific input size to test")
    parser.add_argument("--runs", "-r", type=int, default=DEFAULT_RUNS,
                        help=f"Number of measurement runs (default: {DEFAULT_RUNS})")
    parser.add_argument("--wasm", "-w", type=Path,
                        help="Path to WASM file")
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:8000",
                        help="HTTP server URL for warm start mode")
    parser.add_argument("--allowed-dir", type=str, default="/tmp",
                        help="Allowed directory for WASM sandbox")
    parser.add_argument("--output", "-o", type=Path,
                        help="Output JSON file path")
    parser.add_argument("--list-tools", action="store_true",
                        help="List available tools and exit")

    args = parser.parse_args()

    if args.list_tools:
        print("Available tools:")
        for name, config in TOOL_CONFIGS.items():
            print(f"  {name}: {config['description']}")
            print(f"    Test sizes: {', '.join(config['test_sizes'])}")
        return

    # Determine which tools and sizes to test
    if args.tool:
        tools = [args.tool]
    else:
        tools = list(TOOL_CONFIGS.keys())

    # Determine modes to test
    if args.mode == "both":
        modes = ["cold", "warm"]
    else:
        modes = [args.mode]

    # Run measurements
    all_measurements = []

    for tool_name in tools:
        if tool_name not in TOOL_CONFIGS:
            print(f"Unknown tool: {tool_name}", file=sys.stderr)
            continue

        config = TOOL_CONFIGS[tool_name]

        # Determine sizes to test
        if args.size:
            sizes = [args.size]
        else:
            sizes = config["test_sizes"]

        for mode in modes:
            for size in sizes:
                measurement = run_tool_measurement(
                    tool_name=tool_name,
                    input_size_label=size,
                    mode=mode,
                    runs=args.runs,
                    wasm_path=args.wasm,
                    server_url=args.server_url,
                    allowed_dir=args.allowed_dir
                )
                if measurement:
                    all_measurements.append(measurement)

    # Save results
    if all_measurements:
        if args.output:
            output_file = args.output
        else:
            node_name = get_node_name()
            output_file = RESULTS_DIR / node_name / f"measurements_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        save_results(all_measurements, output_file)
    else:
        print("\nNo measurements collected", file=sys.stderr)


if __name__ == "__main__":
    main()
