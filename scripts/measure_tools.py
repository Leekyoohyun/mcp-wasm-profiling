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

# Server WASM mapping (CLI mode - stdio)
SERVER_WASM_MAP_CLI = {
    'filesystem': 'mcp_server_filesystem_cli.wasm',
    'git': 'mcp_server_git_cli.wasm',
    'image-resize': 'mcp_server_image_resize_cli.wasm',
    'data-aggregate': 'mcp_server_data_aggregate_cli.wasm',
    'log-parser': 'mcp_server_log_parser_cli.wasm',
    'time': 'mcp_server_time_cli.wasm',
    'sequential-thinking': 'mcp_server_sequential_thinking_cli.wasm',
    'summarize': 'mcp_server_summarize_cli.wasm',
    'fetch': 'mcp_server_fetch_cli.wasm',
}

# Server WASM mapping (HTTP mode - wasmtime serve)
SERVER_WASM_MAP_HTTP = {
    'filesystem': 'mcp_server_filesystem_http.wasm',
    'git': 'mcp_server_git_http.wasm',
    'image-resize': 'mcp_server_image_resize_http.wasm',
    'data-aggregate': 'mcp_server_data_aggregate_http.wasm',
    'log-parser': 'mcp_server_log_parser_http.wasm',
    'time': 'mcp_server_time_http.wasm',
    'sequential-thinking': 'mcp_server_sequential_thinking_http.wasm',
    'summarize': 'mcp_server_summarize_http.wasm',
    'fetch': 'mcp_server_fetch_http.wasm',
}

# Default to CLI for backward compatibility
SERVER_WASM_MAP = SERVER_WASM_MAP_CLI

# HTTP server settings
HTTP_SERVER_PORT = 8080
HTTP_SERVER_HOST = "127.0.0.1"

# Tool test configurations (single size per tool - 비율 측정용)
# io_type: "disk" (filesystem/git), "network" (summarize/fetch), "none" (pure compute)
TOOL_CONFIGS = {
    # ===== filesystem (14 tools) - disk I/O =====
    "read_file": {"server": "filesystem", "test_sizes": ["10MB"], "io_type": "disk"},
    "read_text_file": {"server": "filesystem", "test_sizes": ["10MB"], "io_type": "disk"},
    "read_media_file": {"server": "filesystem", "test_sizes": ["default"], "io_type": "disk"},
    "read_multiple_files": {"server": "filesystem", "test_sizes": ["default"], "io_type": "disk"},
    "write_file": {"server": "filesystem", "test_sizes": ["10MB"], "io_type": "disk"},
    "edit_file": {"server": "filesystem", "test_sizes": ["default"], "io_type": "disk"},
    "create_directory": {"server": "filesystem", "test_sizes": ["default"], "io_type": "disk"},
    "list_directory": {"server": "filesystem", "test_sizes": ["100files"], "io_type": "disk"},
    "list_directory_with_sizes": {"server": "filesystem", "test_sizes": ["100files"], "io_type": "disk"},
    "directory_tree": {"server": "filesystem", "test_sizes": ["default"], "io_type": "disk"},
    "move_file": {"server": "filesystem", "test_sizes": ["default"], "io_type": "disk"},
    "search_files": {"server": "filesystem", "test_sizes": ["default"], "io_type": "disk"},
    "get_file_info": {"server": "filesystem", "test_sizes": ["default"], "io_type": "disk"},
    "list_allowed_directories": {"server": "filesystem", "test_sizes": ["default"], "io_type": "disk"},

    # ===== git (12 tools) - disk I/O =====
    "git_status": {"server": "git", "test_sizes": ["default"], "io_type": "disk"},
    "git_log": {"server": "git", "test_sizes": ["default"], "io_type": "disk"},
    "git_show": {"server": "git", "test_sizes": ["default"], "io_type": "disk"},
    "git_branch": {"server": "git", "test_sizes": ["default"], "io_type": "disk"},
    "git_diff_unstaged": {"server": "git", "test_sizes": ["default"], "io_type": "disk"},
    "git_diff_staged": {"server": "git", "test_sizes": ["default"], "io_type": "disk"},
    "git_diff": {"server": "git", "test_sizes": ["default"], "io_type": "disk"},
    "git_commit": {"server": "git", "test_sizes": ["default"], "io_type": "disk"},
    "git_add": {"server": "git", "test_sizes": ["default"], "io_type": "disk"},
    "git_reset": {"server": "git", "test_sizes": ["default"], "io_type": "disk"},
    "git_create_branch": {"server": "git", "test_sizes": ["default"], "io_type": "disk"},
    "git_checkout": {"server": "git", "test_sizes": ["default"], "io_type": "disk"},

    # ===== image-resize (6 tools) - disk I/O =====
    "get_image_info": {"server": "image-resize", "test_sizes": ["default"], "io_type": "disk"},
    "resize_image": {"server": "image-resize", "test_sizes": ["default"], "io_type": "disk"},
    "scan_directory": {"server": "image-resize", "test_sizes": ["default"], "io_type": "disk"},
    "compute_image_hash": {"server": "image-resize", "test_sizes": ["default"], "io_type": "disk"},
    "compare_hashes": {"server": "image-resize", "test_sizes": ["default"], "io_type": "none"},
    "batch_resize": {"server": "image-resize", "test_sizes": ["default"], "io_type": "disk"},

    # ===== data-aggregate (5 tools) - pure compute =====
    "aggregate_list": {"server": "data-aggregate", "test_sizes": ["default"], "io_type": "none"},
    "merge_summaries": {"server": "data-aggregate", "test_sizes": ["default"], "io_type": "none"},
    "combine_research_results": {"server": "data-aggregate", "test_sizes": ["default"], "io_type": "none"},
    "deduplicate": {"server": "data-aggregate", "test_sizes": ["default"], "io_type": "none"},
    "compute_trends": {"server": "data-aggregate", "test_sizes": ["default"], "io_type": "none"},

    # ===== log-parser (5 tools) - pure compute =====
    "parse_logs": {"server": "log-parser", "test_sizes": ["200lines"], "io_type": "none"},
    "filter_entries": {"server": "log-parser", "test_sizes": ["1000lines"], "io_type": "none"},
    "compute_log_statistics": {"server": "log-parser", "test_sizes": ["1000lines"], "io_type": "none"},
    "search_entries": {"server": "log-parser", "test_sizes": ["1000lines"], "io_type": "none"},
    "extract_time_range": {"server": "log-parser", "test_sizes": ["1000lines"], "io_type": "none"},

    # ===== time (2 tools) - pure compute =====
    "get_current_time": {"server": "time", "test_sizes": ["default"], "io_type": "none"},
    "convert_time": {"server": "time", "test_sizes": ["default"], "io_type": "none"},

    # ===== sequential-thinking (1 tool) - pure compute =====
    "sequentialthinking": {"server": "sequential-thinking", "test_sizes": ["default"], "io_type": "none"},

    # ===== summarize (3 tools) - network I/O =====
    "summarize_text": {"server": "summarize", "test_sizes": ["default"], "needs_http": True, "io_type": "network"},
    "summarize_documents": {"server": "summarize", "test_sizes": ["default"], "needs_http": True, "io_type": "network"},
    "get_provider_info": {"server": "summarize", "test_sizes": ["default"], "needs_http": True, "io_type": "none"},

    # ===== fetch (1 tool) - network I/O =====
    "fetch": {"server": "fetch", "test_sizes": ["default"], "needs_http": True, "io_type": "network"},
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
    memory_mb: float = 0.0  # Peak memory usage in MB


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
    timing_pct: Dict[str, float]  # 각 컴포넌트의 비율 (%)
    measurements: List[float]

    # Memory usage (MB)
    memory_mb: float = 0.0
    memory_std: float = 0.0

    # Internal timing if available (averaged)
    internal_timings: Optional[Dict[str, float]] = None


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
import requests

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def get_cgroup_memory_bytes(pid: int) -> Optional[int]:
    """
    Get memory usage from cgroups for a given PID.
    Supports both cgroups v1 and v2.

    Returns memory usage in bytes, or None if not available.
    """
    try:
        # First, find the cgroup for this process
        cgroup_file = Path(f"/proc/{pid}/cgroup")
        if not cgroup_file.exists():
            return None

        cgroup_content = cgroup_file.read_text()

        # cgroups v2 (unified hierarchy)
        # Format: 0::/path/to/cgroup
        for line in cgroup_content.strip().split('\n'):
            parts = line.split(':')
            if len(parts) >= 3 and parts[0] == '0':
                # cgroups v2
                cgroup_path = parts[2]

                # Skip root cgroup - would read system-wide memory
                if cgroup_path == "/" or cgroup_path == "":
                    continue

                # Try peak memory FIRST (captures max usage during execution)
                memory_peak = Path(f"/sys/fs/cgroup{cgroup_path}/memory.peak")
                if memory_peak.exists():
                    return int(memory_peak.read_text().strip())

                # Fallback to current memory
                memory_file = Path(f"/sys/fs/cgroup{cgroup_path}/memory.current")
                if memory_file.exists():
                    return int(memory_file.read_text().strip())

        # cgroups v1 (legacy hierarchy)
        # Format: N:memory:/path/to/cgroup
        for line in cgroup_content.strip().split('\n'):
            parts = line.split(':')
            if len(parts) >= 3 and parts[1] == 'memory':
                cgroup_path = parts[2]

                # Skip root cgroup - would read system-wide memory
                if cgroup_path == "/" or cgroup_path == "":
                    continue

                # Try peak memory FIRST (max_usage_in_bytes)
                memory_peak = Path(f"/sys/fs/cgroup/memory{cgroup_path}/memory.max_usage_in_bytes")
                if memory_peak.exists():
                    return int(memory_peak.read_text().strip())

                # Fallback to current usage
                memory_file = Path(f"/sys/fs/cgroup/memory{cgroup_path}/memory.usage_in_bytes")
                if memory_file.exists():
                    return int(memory_file.read_text().strip())

        # No cgroup memory info available, return None to use psutil fallback
        return None
    except (IOError, ValueError, PermissionError):
        return None


def get_process_memory_mb(pid: int) -> float:
    """
    Get memory usage for a process using psutil.

    Returns memory in MB.
    """
    if HAS_PSUTIL:
        try:
            proc = psutil.Process(pid)
            return proc.memory_info().rss / (1024 * 1024)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    return 0.0

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
    allowed_dirs: list = None,
    needs_http: bool = False
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

    # Load environment variables (including from ~/.env if exists)
    env = os.environ.copy()
    env_file = Path.home() / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env[key] = value

    # Build wasmtime command
    cmd = ["wasmtime", "run"]
    if needs_http:
        cmd.extend(["-S", "http"])  # Enable outbound HTTP via wasi:http
    cmd.extend(dir_args)
    cmd.extend(["--env", "OPENAI_API_KEY", "--env", "UPSTAGE_API_KEY"])
    cmd.append(str(wasm_path))

    start_time = time.perf_counter()

    try:
        result = subprocess.run(
            cmd,
            input=json_input,
            capture_output=True,
            text=True,
            timeout=60.0,
            env=env
        )

        end_time = time.perf_counter()
        total_ms = (end_time - start_time) * 1000

        # profiling: stderr에서 타이밍 마커 파싱
        json_parse_ms = None
        tool_exec_ms = None
        io_ms = None
        wasm_total_ms = None

        # DEBUG: stderr 원본 출력
        if os.environ.get("DEBUG") and result.stderr:
            print(f"\n    [DEBUG] stderr:\n{result.stderr}", file=sys.stderr)

        if result.stderr:
            for line in result.stderr.strip().split('\n'):
                if line.startswith("---JSON_PARSE---"):
                    try:
                        json_parse_ms = float(line[len("---JSON_PARSE---"):])
                    except ValueError:
                        pass
                elif line.startswith("---TOOL_EXEC---"):
                    try:
                        tool_exec_ms = float(line[len("---TOOL_EXEC---"):])
                    except ValueError:
                        pass
                elif line.startswith("---IO---"):
                    try:
                        io_ms = float(line[len("---IO---"):])
                    except ValueError:
                        pass
                elif line.startswith("---WASM_TOTAL---"):
                    try:
                        wasm_total_ms = float(line[len("---WASM_TOTAL---"):])
                    except ValueError:
                        pass

        # internal_timing에 파싱된 값들 추가
        internal_timing = {}
        if json_parse_ms is not None:
            internal_timing["json_parse_ms"] = json_parse_ms
        if tool_exec_ms is not None:
            internal_timing["tool_exec_ms"] = tool_exec_ms
        if io_ms is not None:
            internal_timing["io_ms"] = io_ms
        if wasm_total_ms is not None:
            internal_timing["wasm_total_ms"] = wasm_total_ms

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
            print(f"    wasmtime error: {result.stderr[:500]}", file=sys.stderr)
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
    allowed_dirs: list = None,
    needs_http: bool = False
) -> List[TimingResult]:
    """Run multiple cold start measurements (synchronous, subprocess-based)"""
    results = []

    # Warmup runs
    print(f"    Warmup ({warmup_runs})...", end="", flush=True)
    for _ in range(warmup_runs):
        measure_cold_start(wasm_path, tool_name, payload, allowed_dirs, needs_http)
        print(".", end="", flush=True)
    print(" done")

    # Actual measurements
    print(f"    Measuring ({runs})...", end="", flush=True)
    for i in range(runs):
        result = measure_cold_start(wasm_path, tool_name, payload, allowed_dirs, needs_http)
        result.run_id = i + 1
        results.append(result)
        print(".", end="", flush=True)
    print(" done")

    return results


# =============================================================================
# HTTP Mode Cold Start Measurement (wasmtime serve)
# =============================================================================

def find_free_port() -> int:
    """Find a free port to use for the HTTP server"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def measure_cold_start_http(
    wasm_path: Path,
    tool_name: str,
    payload: Dict[str, Any],
    allowed_dirs: list = None,
    needs_http: bool = False
) -> TimingResult:
    """
    Measure cold start using HTTP mode (wasmtime serve).

    For each measurement:
    1. Start wasmtime serve
    2. Send HTTP request for tool call
    3. Parse timing from response headers
    4. Kill the server

    This measures true cold start for HTTP/serverless deployment.
    """
    port = find_free_port()

    # Build wasmtime serve command
    if allowed_dirs is None:
        allowed_dirs = ["/tmp"]

    dir_args = []
    for d in allowed_dirs:
        dir_args.extend(["--dir", d])

    # Load environment variables
    env = os.environ.copy()
    env_file = Path.home() / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env[key] = value

    cmd = ["wasmtime", "serve", "--addr", f"{HTTP_SERVER_HOST}:{port}"]
    # Enable CLI interface for environment variables
    cmd.extend(["-S", "cli"])
    if needs_http:
        cmd.extend(["-S", "http"])
    # Pass environment variables to WASM (required for API keys)
    cmd.extend(["--env", "OPENAI_API_KEY"])
    cmd.extend(["--env", "UPSTAGE_API_KEY"])
    cmd.extend(["--env", "SUMMARIZE_PROVIDER"])
    cmd.extend(dir_args)
    cmd.append(str(wasm_path))

    server_proc = None
    try:
        # Start the server
        start_time = time.perf_counter()
        server_proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait for server to be ready (poll the port)
        url = f"http://{HTTP_SERVER_HOST}:{port}/"
        max_wait = 10.0  # seconds
        wait_start = time.perf_counter()
        server_ready = False

        while time.perf_counter() - wait_start < max_wait:
            try:
                # Try to connect
                requests.get(url, timeout=0.5)
                server_ready = True
                break
            except requests.exceptions.ConnectionError:
                time.sleep(0.05)
            except Exception:
                time.sleep(0.05)

        if not server_ready:
            print(f"    Server failed to start within {max_wait}s", file=sys.stderr)
            return TimingResult(run_id=0, total_ms=-1)

        # Startup time = server ready time (WASM 로드 + 서버 초기화)
        server_ready_time = time.perf_counter()
        startup_ms = (server_ready_time - start_time) * 1000

        # Create JSON-RPC tool call request
        tool_request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "id": 1,
            "params": {
                "name": tool_name,
                "arguments": payload
            }
        }

        # Send HTTP request and measure
        request_start = time.perf_counter()
        response = requests.post(
            url,
            json=tool_request,
            headers={"Content-Type": "application/json"},
            timeout=60.0
        )
        request_end = time.perf_counter()

        # Total time from process start to response received
        total_ms = (request_end - start_time) * 1000

        # Request processing time (JSON 파싱 + tool 실행)
        request_ms = (request_end - request_start) * 1000

        # Parse timing from response headers
        internal_timing = {
            "startup_ms": startup_ms,  # Python에서 측정한 정확한 startup
        }

        wasm_total = response.headers.get("X-WASM-Total-Ms")
        if wasm_total:
            internal_timing["wasm_total_ms"] = float(wasm_total)

        json_parse = response.headers.get("X-JSON-Parse-Ms")
        if json_parse:
            internal_timing["json_parse_ms"] = float(json_parse)

        tool_exec = response.headers.get("X-Tool-Exec-Ms")
        if tool_exec:
            internal_timing["tool_exec_ms"] = float(tool_exec)

        io_ms = response.headers.get("X-IO-Ms")
        if io_ms:
            internal_timing["io_ms"] = float(io_ms)

        # Measure peak memory usage (absolute value, wasmtime overhead will be subtracted later)
        memory_mb = 0.0
        if server_proc:
            memory_mb = get_process_memory_mb(server_proc.pid)

        if os.environ.get("DEBUG"):
            print(f"\n    [DEBUG] HTTP Response Status: {response.status_code}")
            print(f"    [DEBUG] Startup: {startup_ms:.2f}ms, Request: {request_ms:.2f}ms")
            print(f"    [DEBUG] Memory: {memory_mb:.2f}MB")
            print(f"    [DEBUG] Timing Headers: {dict(response.headers)}")

        return TimingResult(
            run_id=0,
            total_ms=total_ms,
            internal_timing=internal_timing if internal_timing else None,
            memory_mb=memory_mb
        )

    except subprocess.TimeoutExpired:
        print(f"    Timeout", file=sys.stderr)
        return TimingResult(run_id=0, total_ms=-1)
    except Exception as e:
        print(f"    Error: {e}", file=sys.stderr)
        return TimingResult(run_id=0, total_ms=-1)
    finally:
        # Clean up server process
        if server_proc:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                server_proc.kill()


def measure_cold_start_http_multiple(
    wasm_path: Path,
    tool_name: str,
    payload: Dict[str, Any],
    runs: int = DEFAULT_RUNS,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    allowed_dirs: list = None,
    needs_http: bool = False
) -> List[TimingResult]:
    """Run multiple HTTP cold start measurements"""
    results = []

    # Warmup runs
    print(f"    Warmup ({warmup_runs})...", end="", flush=True)
    for _ in range(warmup_runs):
        measure_cold_start_http(wasm_path, tool_name, payload, allowed_dirs, needs_http)
        print(".", end="", flush=True)
    print(" done")

    # Actual measurements
    print(f"    Measuring ({runs})...", end="", flush=True)
    for i in range(runs):
        result = measure_cold_start_http(wasm_path, tool_name, payload, allowed_dirs, needs_http)
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
    results: List[TimingResult],
    io_type: str = "none"  # "disk", "network", or "none"
) -> Optional[ToolMeasurement]:
    """Process timing results into a ToolMeasurement"""

    # Filter out failed measurements
    valid_results = [r for r in results if r.total_ms > 0]

    if not valid_results:
        return None

    # Calculate statistics
    total_times = [r.total_ms for r in valid_results]

    # 시간 분해 모델:
    # total = startup + json_parse + tool_exec
    # - startup: Python 측정 (wasmtime serve 시작 → 서버 준비)
    # - tool_exec: WASM 측정 (X-Tool-Exec-Ms)
    # - json_parse: 남는 시간 (total - startup - tool_exec)
    #
    # tool_exec = io + compute
    # - io: WASM 측정 (X-IO-Ms) → disk_io 또는 network_io로 분류
    # - compute: tool_exec - io
    timing = {
        "total_ms": statistics.mean(total_times),
        "startup_ms": 0.0,         # Python 측정
        "json_parse_ms": 0.0,      # total - startup - tool_exec (남는 시간)
        "tool_exec_ms": 0.0,       # WASM 측정
        "disk_io_ms": 0.0,         # WASM 측정 (io_type=disk)
        "network_io_ms": 0.0,      # WASM 측정 (io_type=network)
        "compute_ms": 0.0,         # tool_exec - io
    }

    timing_std = {
        "total_ms": statistics.stdev(total_times) if len(total_times) > 1 else 0.0,
        "startup_ms": 0.0,
        "json_parse_ms": 0.0,
        "tool_exec_ms": 0.0,
        "disk_io_ms": 0.0,
        "network_io_ms": 0.0,
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

        # wasm_total_ms 파싱 (from ---WASM_TOTAL---)
        wasm_total_values = [t.get("wasm_total_ms", 0) for t in raw_internal_timings]

        # tool_exec_ms 파싱 (from ---TOOL_EXEC---)
        tool_exec_values = [t.get("tool_exec_ms", 0) for t in raw_internal_timings]
        if any(v > 0 for v in tool_exec_values):
            avg_val = statistics.mean(tool_exec_values)
            std_val = statistics.stdev(tool_exec_values) if len(tool_exec_values) > 1 else 0.0
            timing["tool_exec_ms"] = avg_val
            timing_std["tool_exec_ms"] = std_val
            internal_timings_avg["tool_exec_ms"] = avg_val

        # io_ms 파싱 (from ---IO---) → io_type에 따라 disk_io 또는 network_io로 분류
        io_values = [t.get("io_ms", 0) for t in raw_internal_timings]
        if any(v > 0 for v in io_values):
            avg_val = statistics.mean(io_values)
            std_val = statistics.stdev(io_values) if len(io_values) > 1 else 0.0

            if io_type == "disk":
                timing["disk_io_ms"] = avg_val
                timing_std["disk_io_ms"] = std_val
                internal_timings_avg["disk_io_ms"] = avg_val
            elif io_type == "network":
                timing["network_io_ms"] = avg_val
                timing_std["network_io_ms"] = std_val
                internal_timings_avg["network_io_ms"] = avg_val

        # compute_ms = tool_exec - (disk_io + network_io)
        total_io = timing["disk_io_ms"] + timing["network_io_ms"]
        timing["compute_ms"] = max(0.0, timing["tool_exec_ms"] - total_io)
        internal_timings_avg["compute_ms"] = timing["compute_ms"]

        # 시간 분해 모델:
        # total = startup + json_parse + tool_exec
        # - startup: Python 측정
        # - tool_exec: WASM 측정
        # - json_parse: total - startup - tool_exec (남는 시간)

        # 1. startup_ms: Python에서 측정한 값 (wasmtime serve 시작 → 서버 준비)
        startup_values = [t.get("startup_ms", 0) for t in raw_internal_timings if t.get("startup_ms", 0) > 0]
        if startup_values:
            timing["startup_ms"] = statistics.mean(startup_values)
            timing_std["startup_ms"] = statistics.stdev(startup_values) if len(startup_values) > 1 else 0.0
        internal_timings_avg["startup_ms"] = timing["startup_ms"]

        # 2. json_parse_ms: 남는 시간으로 계산 (total - startup - tool_exec)
        json_parse_per_run = []
        for r in valid_results:
            startup = r.internal_timing.get("startup_ms", 0) if r.internal_timing else 0
            tool_exec = r.internal_timing.get("tool_exec_ms", 0) if r.internal_timing else 0
            json_parse = max(0.0, r.total_ms - startup - tool_exec)
            json_parse_per_run.append(json_parse)
        if json_parse_per_run:
            timing["json_parse_ms"] = statistics.mean(json_parse_per_run)
            timing_std["json_parse_ms"] = statistics.stdev(json_parse_per_run) if len(json_parse_per_run) > 1 else 0.0
        internal_timings_avg["json_parse_ms"] = timing["json_parse_ms"]
    else:
        # internal timing이 없으면 전체 시간을 startup으로 간주
        timing["startup_ms"] = timing["total_ms"]

    # profiling: tool_exec 중 disk_io, network_io, compute 비율 계산
    tool_exec = timing["tool_exec_ms"]
    timing_pct = {}
    if tool_exec > 0:
        timing_pct["disk_io_pct"] = round(timing["disk_io_ms"] / tool_exec * 100, 2)
        timing_pct["network_io_pct"] = round(timing["network_io_ms"] / tool_exec * 100, 2)
        timing_pct["compute_pct"] = round(timing["compute_ms"] / tool_exec * 100, 2)
    else:
        timing_pct["disk_io_pct"] = 0
        timing_pct["network_io_pct"] = 0
        timing_pct["compute_pct"] = 0

    # Calculate memory statistics
    memory_values = [r.memory_mb for r in valid_results if r.memory_mb > 0]
    memory_mb = statistics.mean(memory_values) if memory_values else 0.0
    memory_std = statistics.stdev(memory_values) if len(memory_values) > 1 else 0.0

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
        timing_pct=timing_pct,
        measurements=total_times,
        memory_mb=memory_mb,
        memory_std=memory_std,
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
    allowed_dirs: list = None,
    transport: str = "cli"  # "cli" (stdio) or "http"
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

    # Select WASM file based on transport mode
    if transport == "http":
        wasm_map = SERVER_WASM_MAP_HTTP
    else:
        wasm_map = SERVER_WASM_MAP_CLI

    wasm_file = WASM_PATH / wasm_map.get(server_name, "")
    if not wasm_file.exists():
        print(f"WASM file not found: {wasm_file}", file=sys.stderr)
        return None

    # Prepare test payload
    input_size = 0
    output_size = 0
    payload = None

    # ===== filesystem tools =====
    if tool_name in ["read_file", "read_text_file"]:
        test_file = get_test_file_path(input_size_label)
        if not test_file.exists():
            print(f"Test file not found: {test_file}", file=sys.stderr)
            return None
        payload = {"path": str(test_file)}
        input_size = get_file_size(test_file)

    elif tool_name == "read_media_file":
        img_path = TEST_DATA_DIR / "images" / "test.png"
        if not img_path.exists():
            print(f"Test image not found: {img_path}", file=sys.stderr)
            return None
        payload = {"path": str(img_path)}

    elif tool_name == "read_multiple_files":
        test_file = get_test_file_path("1KB")
        if not test_file.exists():
            print(f"Test file not found: {test_file}", file=sys.stderr)
            return None
        payload = {"paths": [str(test_file)]}

    elif tool_name == "write_file":
        size_bytes = parse_size_label(input_size_label)
        content = "x" * size_bytes
        output_path = f"/tmp/wasm_profiler_write_{input_size_label}.txt"
        payload = {"path": output_path, "content": content}
        input_size = size_bytes

    elif tool_name == "edit_file":
        test_file = Path("/tmp/wasm_profiler_edit.txt")
        test_file.write_text("hello world\nline two\nline three")
        payload = {"path": str(test_file), "edits": [{"oldText": "hello", "newText": "hi"}], "dryRun": True}

    elif tool_name == "create_directory":
        payload = {"path": "/tmp/wasm_profiler_test_dir"}

    elif tool_name in ["list_directory", "list_directory_with_sizes"]:
        test_dir = get_test_directory_path(input_size_label)
        if not test_dir.exists():
            print(f"Test directory not found: {test_dir}", file=sys.stderr)
            return None
        payload = {"path": str(test_dir)}

    elif tool_name == "directory_tree":
        payload = {"path": str(TEST_DATA_DIR)}

    elif tool_name == "move_file":
        # move_file은 실제 파일 이동이므로 /tmp 사용
        src = Path("/tmp/wasm_profiler_move_src.txt")
        src.write_text("test")
        payload = {"source": str(src), "destination": "/tmp/wasm_profiler_move_dst.txt"}

    elif tool_name == "search_files":
        payload = {"path": str(TEST_DATA_DIR / "files"), "pattern": "*.txt"}

    elif tool_name == "get_file_info":
        test_file = get_test_file_path("1KB")
        if not test_file.exists():
            print(f"Test file not found: {test_file}", file=sys.stderr)
            return None
        payload = {"path": str(test_file)}

    elif tool_name == "list_allowed_directories":
        payload = {}

    # ===== git tools =====
    # test_data/git_repo 사용 (setup_test_data.sh로 생성)
    git_repo = str(TEST_DATA_DIR / "git_repo")
    git_repo_path = TEST_DATA_DIR / "git_repo"

    # git 테스트 전에 변경사항 생성 (diff가 실제로 발생하도록)
    if tool_name.startswith("git_"):
        test_file = git_repo_path / "test_changes.txt"
        import random
        # 매번 다른 내용으로 파일 수정 (unstaged 변경사항 생성)
        test_file.write_text(f"Test content modified at {time.time()}\n" + "x" * 10000)

        # staged 변경사항도 생성
        staged_file = git_repo_path / "staged_changes.txt"
        staged_file.write_text(f"Staged content {time.time()}\n" + "y" * 10000)
        import subprocess
        subprocess.run(["git", "-C", git_repo, "add", "staged_changes.txt"],
                      capture_output=True, timeout=5)

    if tool_name == "git_status":
        payload = {"repo_path": git_repo}
    elif tool_name == "git_log":
        payload = {"repo_path": git_repo, "max_count": 5}
    elif tool_name == "git_show":
        payload = {"repo_path": git_repo, "revision": "HEAD"}
    elif tool_name == "git_branch":
        payload = {"repo_path": git_repo, "branch_type": "local"}
    elif tool_name in ["git_diff_unstaged", "git_diff_staged", "git_reset"]:
        payload = {"repo_path": git_repo}
    elif tool_name == "git_diff":
        payload = {"repo_path": git_repo, "target": "HEAD~1"}
    elif tool_name == "git_commit":
        payload = {"repo_path": git_repo, "message": "test commit"}
    elif tool_name == "git_add":
        payload = {"repo_path": git_repo, "files": ["test_changes.txt"]}
    elif tool_name == "git_create_branch":
        payload = {"repo_path": git_repo, "branch_name": f"test-branch-{int(time.time())}"}
    elif tool_name == "git_checkout":
        payload = {"repo_path": git_repo, "branch_name": "master"}

    # ===== image-resize tools =====
    if tool_name in ["get_image_info", "compute_image_hash", "resize_image"]:
        img_path = TEST_DATA_DIR / "images" / "test.png"
        if not img_path.exists():
            print(f"Test image not found: {img_path}", file=sys.stderr)
            return None
        if tool_name == "resize_image":
            payload = {"image_path": str(img_path), "width": 100, "height": 100}
        else:
            payload = {"image_path": str(img_path)}
    elif tool_name == "scan_directory":
        img_dir = TEST_DATA_DIR / "images"
        if not img_dir.exists():
            print(f"Test images directory not found: {img_dir}", file=sys.stderr)
            return None
        payload = {"directory": str(img_dir)}
    elif tool_name == "compare_hashes":
        payload = {
            "hashes": [
                {"path": "img1.png", "hash": "0123456789abcdef"},
                {"path": "img2.png", "hash": "0123456789abcdef"}
            ],
            "threshold": 5
        }
    elif tool_name == "batch_resize":
        img_path = TEST_DATA_DIR / "images" / "test.png"
        if not img_path.exists():
            print(f"Test image not found: {img_path}", file=sys.stderr)
            return None
        payload = {"image_paths": [str(img_path)], "max_size": 100}

    # ===== data-aggregate tools (100,000개 아이템 → ~5MB) =====
    if tool_name == "aggregate_list":
        # 100,000 items → ~5MB input (이전 테스트: 54MB)
        items = [{"level": ["INFO", "WARN", "ERROR"][i % 3], "value": i * 10} for i in range(100000)]
        payload = {"items": items, "group_by": "level"}
    elif tool_name == "merge_summaries":
        # 100,000 items → ~5MB input (이전 테스트: 51MB)
        summaries = [{"count": i * 10, "total": i * 100} for i in range(100000)]
        payload = {"summaries": summaries}
    elif tool_name == "combine_research_results":
        # 100,000 items with longer content → ~10MB input
        results = [{"title": f"Title {i}", "summary": f"Summary content {i} " * 10} for i in range(100000)]
        payload = {"results": results}
    elif tool_name == "deduplicate":
        # 100,000 items → ~5MB input (이전 테스트: 54MB)
        items = [{"id": i % 50000, "name": f"item_{i}"} for i in range(100000)]  # 50% duplicates
        payload = {"items": items, "key_fields": ["id"]}
    elif tool_name == "compute_trends":
        # 100,000 time series points → ~5MB input
        time_series = [{"timestamp": f"2025-01-{(i%28)+1:02d}", "value": 100 + i * 5} for i in range(100000)]
        payload = {"time_series": time_series}

    # ===== log-parser tools =====
    if tool_name == "parse_logs":
        num_lines = int(input_size_label.replace("lines", ""))
        log_content = generate_test_log_content(num_lines)
        payload = {"log_content": log_content, "format_type": "auto"}
        input_size = len(log_content)
    elif tool_name in ["filter_entries", "compute_log_statistics", "search_entries", "extract_time_range"]:
        num_lines = int(input_size_label.replace("lines", ""))
        log_content = generate_test_log_content(num_lines)
        entries = [{"timestamp": f"2025-01-01T10:{i:02d}:00", "level": "INFO", "message": f"msg{i}"} for i in range(num_lines)]
        if tool_name == "filter_entries":
            payload = {"entries": entries, "min_level": "info"}
        elif tool_name == "compute_log_statistics":
            payload = {"entries": entries}
        elif tool_name == "search_entries":
            payload = {"entries": entries, "pattern": "msg"}
        elif tool_name == "extract_time_range":
            payload = {"entries": entries}

    # ===== time tools =====
    if tool_name == "get_current_time":
        payload = {"timezone": "UTC"}
    elif tool_name == "convert_time":
        payload = {"source_timezone": "UTC", "time": "12:00", "target_timezone": "Asia/Seoul"}

    # ===== sequential-thinking tools =====
    if tool_name == "sequentialthinking":
        payload = {
            "thought": "This is a test thought for profiling. " * 100,
            "nextThoughtNeeded": True,
            "thoughtNumber": 1,
            "totalThoughts": 5
        }

    # ===== summarize tools =====
    if tool_name == "summarize_text":
        payload = {"text": "This is a test document. " * 50, "max_length": 100}
    elif tool_name == "summarize_documents":
        payload = {"documents": ["Doc content " * 20], "max_length_per_doc": 100}
    elif tool_name == "get_provider_info":
        payload = {}

    # ===== fetch tools =====
    if tool_name == "fetch":
        payload = {"url": "https://httpbin.org/get"}

    # payload가 설정되지 않은 경우
    if payload is None:
        print(f"Tool {tool_name} not yet configured", file=sys.stderr)
        return None

    # input_size가 0이면 payload 크기로 계산
    if input_size == 0 and payload:
        input_size = len(json.dumps(payload))

    # Check if tool needs outbound HTTP
    needs_http = config.get("needs_http", False)

    print(f"\n[{tool_name}] Input: {input_size_label} ({input_size} bytes), Mode: {mode}, Transport: {transport}" +
          (" (needs_http)" if needs_http else ""))

    # profiling: 기본 디렉토리 설정 (test_data + /tmp)
    if allowed_dirs is None:
        allowed_dirs = [str(TEST_DATA_DIR), "/tmp"]

    # Run measurements
    if mode == "cold":
        if transport == "http":
            # HTTP mode: wasmtime serve
            results = measure_cold_start_http_multiple(
                wasm_file, tool_name, payload,
                runs=runs, allowed_dirs=allowed_dirs,
                needs_http=needs_http
            )
        else:
            # CLI mode: subprocess (synchronous) - new process each time
            results = measure_cold_start_multiple(
                wasm_file, tool_name, payload,
                runs=runs, allowed_dirs=allowed_dirs,
                needs_http=needs_http
            )
    else:  # warm
        # Warm start: MCP client (async) - reuse session (CLI mode only)
        if transport == "http":
            print(f"    Warning: Warm start not supported in HTTP mode, using cold start", file=sys.stderr)
            results = measure_cold_start_http_multiple(
                wasm_file, tool_name, payload,
                runs=runs, allowed_dirs=allowed_dirs,
                needs_http=needs_http
            )
        else:
            results = await measure_warm_start_multiple(
                server_name, wasm_file, tool_name, payload,
                runs=runs, allowed_dirs=allowed_dirs
            )

    # Get io_type for this tool
    io_type = config.get("io_type", "none")

    # Process results
    measurement = process_results(
        tool_name, mode, input_size, input_size_label, output_size, results, io_type
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


def save_summary(measurements: List[ToolMeasurement], output_file: Path):
    """Save summary of cold-start measurements only"""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # cold 모드만 필터링
    cold_measurements = [m for m in measurements if m.mode == "cold"]

    summary = {
        "node": cold_measurements[0].node if cold_measurements else "unknown",
        "timestamp": datetime.now().isoformat(),
        "total_measurements": len(cold_measurements),
        "tools": {}
    }

    for m in cold_measurements:
        key = m.tool_name
        # 시간 분해 모델:
        # total = startup + wasm_total
        # startup = total - wasm_total (wasmtime 오버헤드 + 서버 생성)
        # json_parse = wasm_total - tool_exec
        # tool_exec = 도구 실행 시간 (---TOOL_EXEC---)
        # disk_io = 디스크 I/O 시간
        # network_io = 네트워크 I/O 시간
        # compute = tool_exec - (disk_io + network_io)
        summary["tools"][key] = {
            "tool_name": m.tool_name,
            "input_size": m.input_size,
            "input_size_label": m.input_size_label,
            "runs": m.runs,
            "timing_ms": {
                "total": round(m.timing["total_ms"], 3),
                "startup": round(m.timing["startup_ms"], 3),
                "json_parse": round(m.timing["json_parse_ms"], 3),
                "tool_exec": round(m.timing["tool_exec_ms"], 3),
                "disk_io": round(m.timing["disk_io_ms"], 3),
                "network_io": round(m.timing["network_io_ms"], 3),
                "compute": round(m.timing["compute_ms"], 3),
            },
            "timing_pct": m.timing_pct,
            "memory_mb": round(m.memory_mb, 2),
        }

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: {output_file} (cold mode only)")


# =============================================================================
# CLI Interface
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="WASM MCP Tool Profiler (Lumos-style time decomposition)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Measure all tools (cold start, CLI mode)
  python measure_tools.py

  # Measure with HTTP transport (wasmtime serve)
  python measure_tools.py --transport http

  # Measure specific tool
  python measure_tools.py --tool read_file

  # Measure both cold and warm start
  python measure_tools.py --mode both

  # Measure with specific input size
  python measure_tools.py --tool read_file --size 1MB

Transport modes:
  cli   - Uses wasmtime run with stdio (JSON-RPC over stdin/stdout)
  http  - Uses wasmtime serve (JSON-RPC over HTTP)
"""
    )

    parser.add_argument("--tool", "-t", type=str,
                        help="Specific tool to measure (default: all)")
    parser.add_argument("--transport", "-T", choices=["cli", "http"],
                        default="http", help="Transport mode: cli (stdio) or http (default: http)")
    parser.add_argument("--mode", "-m", choices=["cold", "warm", "both"],
                        default="cold", help="Measurement mode (default: cold)")
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
    print(f"Transport: {args.transport}")
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
                    runs=args.runs,
                    transport=args.transport
                )
                if measurement:
                    all_measurements.append(measurement)

    # Save results
    if all_measurements:
        node_name = get_node_name()
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        transport_suffix = f"_{args.transport}"

        if args.output:
            output_file = args.output
        else:
            output_file = RESULTS_DIR / node_name / f"measurements{transport_suffix}_{timestamp_str}.json"

        save_results(all_measurements, output_file)

        # Save summary file
        summary_file = RESULTS_DIR / node_name / f"summary{transport_suffix}_{timestamp_str}.json"
        save_summary(all_measurements, summary_file)
    else:
        print("\nNo measurements collected", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
