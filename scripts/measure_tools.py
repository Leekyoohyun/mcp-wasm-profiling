#!/usr/bin/env python3
"""
WASM MCP Tool Profiling Script

Measures execution time of MCP tools running in WASM environment.
Decomposes timing into: T_cold, T_io, T_serialize, T_compute

Based on Lumos methodology for serverless function profiling.
Uses langchain_mcp_adapters (same as 2b_measure_wasm_tools_mcp.py)
"""

import asyncio
import argparse
import json
import os
import shutil
import socket
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import statistics

# Add wasm_mcp tests to path for MCPServerConfig
WASM_MCP_PATH_CANDIDATES = [
    Path.home() / "EdgeAgent/wasm_mcp",
    Path.home() / "DDPS/undergraduated/CCGrid-2026/EdgeAgent/EdgeAgent/wasm_mcp",
]

WASM_MCP_PATH = None
for path in WASM_MCP_PATH_CANDIDATES:
    if path.exists():
        WASM_MCP_PATH = path
        break

if WASM_MCP_PATH:
    sys.path.insert(0, str(WASM_MCP_PATH / "tests"))
    from mcp_comparator import MCPServerConfig, TransportType

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


# =============================================================================
# Configuration
# =============================================================================

RESULTS_DIR = Path(__file__).parent.parent / "results"
TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"

# Default test configurations
DEFAULT_RUNS = 10
DEFAULT_WARMUP_RUNS = 3

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

# Server WASM mapping
SERVER_WASM_MAP = {
    'filesystem': 'mcp_server_filesystem.wasm',
    'log-parser': 'mcp_server_log_parser.wasm',
}

# Tool test configurations
TOOL_CONFIGS = {
    "read_file": {
        "description": "Read file contents",
        "server": "filesystem",
        "test_sizes": ["1KB", "10KB", "100KB", "1MB", "10MB", "50MB"],
    },
    "read_text_file": {
        "description": "Read text file with encoding",
        "server": "filesystem",
        "test_sizes": ["1KB", "10KB", "100KB", "1MB", "10MB", "50MB"],
    },
    "write_file": {
        "description": "Write file contents",
        "server": "filesystem",
        "test_sizes": ["1KB", "10KB", "100KB", "1MB"],
    },
    "list_directory": {
        "description": "List directory contents",
        "server": "filesystem",
        "test_sizes": ["10files", "100files", "1000files"],
    },
    "get_file_info": {
        "description": "Get file metadata",
        "server": "filesystem",
        "test_sizes": ["default"],
    },
    "parse_logs": {
        "description": "Parse log content into structured entries",
        "server": "log-parser",
        "test_sizes": ["100lines", "1000lines", "10000lines"],
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
    internal_timing: Optional[Dict[str, float]] = None


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
    timing: Dict[str, float]
    timing_std: Dict[str, float]
    measurements: List[float]

    # Internal timing if available
    internal_timings: Optional[List[Dict[str, float]]] = None


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


def get_file_size(path: Path) -> int:
    """Get file size in bytes"""
    if path.exists():
        return path.stat().st_size
    return 0


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
        return int(label) if label.isdigit() else 0


def get_test_file_path(size_label: str) -> Path:
    """Get path to test file of specified size"""
    return TEST_DATA_DIR / "files" / f"test_{size_label}.txt"


def get_test_directory_path(size_label: str) -> Path:
    """Get path to test directory"""
    return TEST_DATA_DIR / "directories" / size_label


def generate_test_log_content(num_lines: int) -> str:
    """Generate test log content in Apache combined format"""
    import random

    ips = ["192.168.1.100", "10.0.0.50", "172.16.0.25", "8.8.8.8", "1.2.3.4"]
    methods = ["GET", "POST", "PUT", "DELETE"]
    paths = ["/api/users", "/api/data", "/index.html", "/static/js/app.js", "/favicon.ico"]
    statuses = [200, 200, 200, 301, 400, 404, 500]
    agents = ["Mozilla/5.0", "curl/7.68.0", "Python-requests/2.25.1"]

    lines = []
    for i in range(num_lines):
        ip = random.choice(ips)
        method = random.choice(methods)
        path = random.choice(paths)
        status = random.choice(statuses)
        size = random.randint(100, 50000)
        agent = random.choice(agents)
        timestamp = f"17/Dec/2025:10:{i % 60:02d}:{i % 60:02d} +0000"

        line = f'{ip} - - [{timestamp}] "{method} {path} HTTP/1.1" {status} {size} "-" "{agent}"'
        lines.append(line)

    return "\n".join(lines)


# =============================================================================
# Cold Start Measurement (subprocess - new process each time)
# =============================================================================

import subprocess

def create_jsonrpc_messages(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Create JSON-RPC messages for MCP tool call"""
    init_request = json.dumps({
        "jsonrpc": "2.0",
        "method": "initialize",
        "id": 1,
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "wasm-profiler", "version": "1.0.0"}
        }
    })

    init_notification = json.dumps({
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    })

    tool_request = json.dumps({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": 2,
        "params": {
            "name": tool_name,
            "arguments": arguments
        }
    })

    return f"{init_request}\n{init_notification}\n{tool_request}\n"


def measure_cold_start(
    wasm_path: Path,
    tool_name: str,
    payload: Dict[str, Any],
    allowed_dirs: list = None
) -> TimingResult:
    """
    Measure cold start using subprocess (new wasmtime process each time).

    Cold start includes:
    - wasmtime process spawn
    - WASM module load
    - MCP initialization
    - Tool execution
    """
    # Create JSON-RPC input
    json_input = create_jsonrpc_messages(tool_name, payload)

    # profiling: 여러 디렉토리 허용
    if allowed_dirs is None:
        allowed_dirs = ["/tmp"]

    dir_args = []
    for d in allowed_dirs:
        dir_args.extend(["--dir", d])

    start_time = time.perf_counter()

    try:
        result = subprocess.run(
            ["wasmtime", "run"] + dir_args + [str(wasm_path)],
            input=json_input,
            capture_output=True,
            text=True,
            timeout=60.0
        )

        end_time = time.perf_counter()
        total_ms = (end_time - start_time) * 1000

        # Parse response for internal timing
        internal_timing = None

        # profiling: stderr에서 ---TIMING--- 형식 파싱
        if result.stderr:
            for line in result.stderr.strip().split('\n'):
                if line.startswith("---TIMING---"):
                    try:
                        timing_json = line[len("---TIMING---"):]
                        internal_timing = json.loads(timing_json)
                    except json.JSONDecodeError:
                        pass

        # Fallback: stdout에서 _timing 필드 찾기 (기존 방식)
        if internal_timing is None and result.stdout:
            for line in result.stdout.strip().split('\n'):
                try:
                    response = json.loads(line)
                    if "result" in response and isinstance(response["result"], dict):
                        if "_timing" in response["result"]:
                            internal_timing = response["result"]["_timing"]
                except json.JSONDecodeError:
                    continue

        # Check for errors
        if result.returncode != 0:
            print(f"    wasmtime error: {result.stderr[:100]}", file=sys.stderr)
            return TimingResult(run_id=0, total_ms=-1)

        return TimingResult(
            run_id=0,
            total_ms=total_ms,
            internal_timing=internal_timing
        )

    except subprocess.TimeoutExpired:
        print(f"    Timeout", file=sys.stderr)
        return TimingResult(run_id=0, total_ms=-1)
    except Exception as e:
        print(f"    Error: {e}", file=sys.stderr)
        return TimingResult(run_id=0, total_ms=-1)


def measure_cold_start_multiple(
    wasm_path: Path,
    tool_name: str,
    payload: Dict[str, Any],
    runs: int = DEFAULT_RUNS,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    allowed_dirs: list = None
) -> List[TimingResult]:
    """Run multiple cold start measurements (synchronous, subprocess-based)"""
    results = []

    # Warmup runs
    print(f"    Warmup ({warmup_runs})...", end="", flush=True)
    for _ in range(warmup_runs):
        measure_cold_start(wasm_path, tool_name, payload, allowed_dirs)
        print(".", end="", flush=True)
    print(" done")

    # Actual measurements
    print(f"    Measuring ({runs})...", end="", flush=True)
    for i in range(runs):
        result = measure_cold_start(wasm_path, tool_name, payload, allowed_dirs)
        result.run_id = i + 1
        results.append(result)
        print(".", end="", flush=True)
    print(" done")

    return results


# =============================================================================
# Warm Start Measurement
# =============================================================================

async def measure_warm_start_multiple(
    server_name: str,
    wasm_path: Path,
    tool_name: str,
    payload: Dict[str, Any],
    runs: int = DEFAULT_RUNS,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    allowed_dirs: list = None
) -> List[TimingResult]:
    """
    Measure warm start - reuses same session for all measurements.

    Warm start excludes cold start overhead:
    - Session already established
    - WASM module already loaded
    - Only measures tool execution time
    """
    # profiling: 여러 디렉토리 중 첫 번째 사용 (MCP client 제약)
    if allowed_dirs is None:
        allowed_dirs = ["/tmp"]
    allowed_dir = allowed_dirs[0]
    server_config = MCPServerConfig.wasmmcp_stdio(allowed_dir, str(wasm_path))
    client = MultiServerMCPClient({server_name: server_config.config})

    results = []

    try:
        async with client.session(server_name) as session:
            tools = await load_mcp_tools(session)
            tool_map = {t.name: t for t in tools}

            if tool_name not in tool_map:
                print(f"    Tool {tool_name} not found")
                return results

            tool_obj = tool_map[tool_name]

            # Warmup runs
            print(f"    Warmup ({warmup_runs})...", end="", flush=True)
            for _ in range(warmup_runs):
                await tool_obj.ainvoke(payload)
                print(".", end="", flush=True)
            print(" done")

            # Actual measurements
            print(f"    Measuring ({runs})...", end="", flush=True)
            for i in range(runs):
                start_time = time.perf_counter()
                result = await tool_obj.ainvoke(payload)
                end_time = time.perf_counter()

                total_ms = (end_time - start_time) * 1000

                # Extract internal timing if available
                internal_timing = None
                if isinstance(result, dict) and "_timing" in result:
                    internal_timing = result["_timing"]

                results.append(TimingResult(
                    run_id=i + 1,
                    total_ms=total_ms,
                    internal_timing=internal_timing
                ))
                print(".", end="", flush=True)
            print(" done")

    except Exception as e:
        print(f"    Error: {e}", file=sys.stderr)

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
) -> Optional[ToolMeasurement]:
    """Process timing results into a ToolMeasurement"""

    # Filter out failed measurements
    valid_results = [r for r in results if r.total_ms > 0]

    if not valid_results:
        return None

    # Calculate statistics
    total_times = [r.total_ms for r in valid_results]

    timing = {
        "total_ms": statistics.mean(total_times),
        "cold_start_ms": 0.0,
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
    raw_internal_timings = []
    for r in valid_results:
        if r.internal_timing:
            raw_internal_timings.append(r.internal_timing)

    # profiling: internal_timings 평균값으로 저장
    internal_timings_avg = None
    if raw_internal_timings:
        internal_timings_avg = {}
        for key in ["io_ms", "serialize_ms", "compute_ms"]:
            values = [t.get(key, 0) for t in raw_internal_timings]
            if any(v > 0 for v in values):
                timing[key] = statistics.mean(values)
                timing_std[key] = statistics.stdev(values) if len(values) > 1 else 0.0
                internal_timings_avg[key] = timing[key]
            else:
                internal_timings_avg[key] = 0.0

        # profiling: cold_start = total - io - compute - serialize (overhead 계산)
        timing["cold_start_ms"] = timing["total_ms"] - timing["io_ms"] - timing["compute_ms"] - timing["serialize_ms"]
        if timing["cold_start_ms"] < 0:
            timing["cold_start_ms"] = 0.0
        internal_timings_avg["cold_start_ms"] = timing["cold_start_ms"]

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
        internal_timings=internal_timings_avg
    )


# =============================================================================
# Main Measurement Logic
# =============================================================================

async def run_tool_measurement(
    tool_name: str,
    input_size_label: str,
    mode: str = "cold",
    runs: int = DEFAULT_RUNS,
    allowed_dirs: list = None
) -> Optional[ToolMeasurement]:
    """Run measurement for a single tool with specified input size"""

    if tool_name not in TOOL_CONFIGS:
        print(f"Unknown tool: {tool_name}", file=sys.stderr)
        return None

    config = TOOL_CONFIGS[tool_name]
    server_name = config["server"]

    # Get WASM path
    if WASM_PATH is None:
        print("WASM path not found", file=sys.stderr)
        return None

    wasm_file = WASM_PATH / SERVER_WASM_MAP.get(server_name, "")
    if not wasm_file.exists():
        print(f"WASM file not found: {wasm_file}", file=sys.stderr)
        return None

    # Prepare test payload
    input_size = 0
    output_size = 0

    if tool_name in ["read_file", "read_text_file"]:
        test_file = get_test_file_path(input_size_label)
        if not test_file.exists():
            print(f"Test file not found: {test_file}", file=sys.stderr)
            return None
        payload = {"path": str(test_file)}
        input_size = get_file_size(test_file)

    elif tool_name == "write_file":
        size_bytes = parse_size_label(input_size_label)
        content = "x" * size_bytes
        output_path = f"/tmp/wasm_profiler_write_{input_size_label}.txt"
        payload = {"path": output_path, "content": content}
        input_size = size_bytes

    elif tool_name in ["list_directory"]:
        test_dir = get_test_directory_path(input_size_label)
        if not test_dir.exists():
            print(f"Test directory not found: {test_dir}", file=sys.stderr)
            return None
        payload = {"path": str(test_dir)}

    elif tool_name == "get_file_info":
        test_file = get_test_file_path("1KB")
        if not test_file.exists():
            test_file = Path("/tmp/test_info.txt")
            test_file.write_text("test content")
        payload = {"path": str(test_file)}

    elif tool_name == "parse_logs":
        # Generate log content based on size
        num_lines = int(input_size_label.replace("lines", ""))
        log_content = generate_test_log_content(num_lines)
        payload = {"log_content": log_content, "format_type": "auto"}
        input_size = len(log_content)

    else:
        print(f"Tool {tool_name} not yet configured", file=sys.stderr)
        return None

    print(f"\n[{tool_name}] Input: {input_size_label}, Mode: {mode}")

    # profiling: 기본 디렉토리 설정 (test_data + /tmp)
    if allowed_dirs is None:
        allowed_dirs = [str(TEST_DATA_DIR), "/tmp"]

    # Run measurements
    if mode == "cold":
        # Cold start: subprocess (synchronous) - new process each time
        results = measure_cold_start_multiple(
            wasm_file, tool_name, payload,
            runs=runs, allowed_dirs=allowed_dirs
        )
    else:  # warm
        # Warm start: MCP client (async) - reuse session
        results = await measure_warm_start_multiple(
            server_name, wasm_file, tool_name, payload,
            runs=runs, allowed_dirs=allowed_dirs
        )

    # Process results
    measurement = process_results(
        tool_name, mode, input_size, input_size_label, output_size, results
    )

    if measurement:
        print(f"    Mean: {measurement.timing['total_ms']:.2f}ms "
              f"(std: {measurement.timing_std['total_ms']:.2f}ms)")

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

async def main():
    parser = argparse.ArgumentParser(
        description="WASM MCP Tool Profiler (Lumos-style time decomposition)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Measure all tools (cold start)
  python measure_tools.py

  # Measure specific tool
  python measure_tools.py --tool read_file

  # Measure both cold and warm start
  python measure_tools.py --mode both

  # Measure with specific input size
  python measure_tools.py --tool read_file --size 1MB
"""
    )

    parser.add_argument("--tool", "-t", type=str,
                        help="Specific tool to measure (default: all)")
    parser.add_argument("--mode", "-m", choices=["cold", "warm", "both"],
                        default="both", help="Measurement mode")
    parser.add_argument("--size", "-s", type=str,
                        help="Specific input size to test")
    parser.add_argument("--runs", "-r", type=int, default=DEFAULT_RUNS,
                        help=f"Number of measurement runs (default: {DEFAULT_RUNS})")
    parser.add_argument("--output", "-o", type=Path,
                        help="Output JSON file path")
    parser.add_argument("--list-tools", action="store_true",
                        help="List available tools and exit")

    args = parser.parse_args()

    if args.list_tools:
        print("Available tools:")
        for name, config in TOOL_CONFIGS.items():
            print(f"  {name}: {config['description']}")
            print(f"    Server: {config['server']}")
            print(f"    Test sizes: {', '.join(config['test_sizes'])}")
        return

    print("=" * 60)
    print("WASM MCP Tool Profiler")
    print("=" * 60)
    print(f"Node: {get_node_name()}")
    print(f"WASM Path: {WASM_PATH}")
    print()

    # Determine tools and sizes
    if args.tool:
        tools = [args.tool]
    else:
        tools = list(TOOL_CONFIGS.keys())

    # Determine modes
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

        # Determine sizes
        if args.size:
            sizes = [args.size]
        else:
            sizes = config["test_sizes"]

        for mode in modes:
            for size in sizes:
                measurement = await run_tool_measurement(
                    tool_name=tool_name,
                    input_size_label=size,
                    mode=mode,
                    runs=args.runs
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
    asyncio.run(main())
