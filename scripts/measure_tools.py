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

# Server WASM mapping
SERVER_WASM_MAP = {
    'filesystem': 'mcp_server_filesystem.wasm',
    'git': 'mcp_server_git.wasm',
    'image-resize': 'mcp_server_image_resize.wasm',
    'data-aggregate': 'mcp_server_data_aggregate.wasm',
    'log-parser': 'mcp_server_log_parser.wasm',
    'summarize': 'mcp_server_summarize.wasm',
    'time': 'mcp_server_time.wasm',
    'fetch': 'mcp_server_fetch.wasm',
}

# Tool test configurations (single size per tool - 비율 측정용)
TOOL_CONFIGS = {
    # ===== filesystem (14 tools) =====
    "read_file": {"server": "filesystem", "test_sizes": ["10MB"]},
    "read_text_file": {"server": "filesystem", "test_sizes": ["10MB"]},
    "read_media_file": {"server": "filesystem", "test_sizes": ["default"]},
    "read_multiple_files": {"server": "filesystem", "test_sizes": ["default"]},
    "write_file": {"server": "filesystem", "test_sizes": ["10MB"]},
    "edit_file": {"server": "filesystem", "test_sizes": ["default"]},
    "create_directory": {"server": "filesystem", "test_sizes": ["default"]},
    "list_directory": {"server": "filesystem", "test_sizes": ["100files"]},
    "list_directory_with_sizes": {"server": "filesystem", "test_sizes": ["100files"]},
    "directory_tree": {"server": "filesystem", "test_sizes": ["default"]},
    "move_file": {"server": "filesystem", "test_sizes": ["default"]},
    "search_files": {"server": "filesystem", "test_sizes": ["default"]},
    "get_file_info": {"server": "filesystem", "test_sizes": ["default"]},
    "list_allowed_directories": {"server": "filesystem", "test_sizes": ["default"]},

    # ===== git (12 tools) =====
    "git_status": {"server": "git", "test_sizes": ["default"]},
    "git_log": {"server": "git", "test_sizes": ["default"]},
    "git_show": {"server": "git", "test_sizes": ["default"]},
    "git_branch": {"server": "git", "test_sizes": ["default"]},
    "git_diff_unstaged": {"server": "git", "test_sizes": ["default"]},
    "git_diff_staged": {"server": "git", "test_sizes": ["default"]},
    "git_diff": {"server": "git", "test_sizes": ["default"]},
    "git_commit": {"server": "git", "test_sizes": ["default"]},
    "git_add": {"server": "git", "test_sizes": ["default"]},
    "git_reset": {"server": "git", "test_sizes": ["default"]},
    "git_create_branch": {"server": "git", "test_sizes": ["default"]},
    "git_checkout": {"server": "git", "test_sizes": ["default"]},

    # ===== image-resize (6 tools) =====
    "get_image_info": {"server": "image-resize", "test_sizes": ["default"]},
    "resize_image": {"server": "image-resize", "test_sizes": ["default"]},
    "scan_directory": {"server": "image-resize", "test_sizes": ["default"]},
    "compute_image_hash": {"server": "image-resize", "test_sizes": ["default"]},
    "compare_hashes": {"server": "image-resize", "test_sizes": ["default"]},
    "batch_resize": {"server": "image-resize", "test_sizes": ["default"]},

    # ===== data-aggregate (5 tools) =====
    "aggregate_list": {"server": "data-aggregate", "test_sizes": ["default"]},
    "merge_summaries": {"server": "data-aggregate", "test_sizes": ["default"]},
    "combine_research_results": {"server": "data-aggregate", "test_sizes": ["default"]},
    "deduplicate": {"server": "data-aggregate", "test_sizes": ["default"]},
    "compute_trends": {"server": "data-aggregate", "test_sizes": ["default"]},

    # ===== log-parser (5 tools) =====
    "parse_logs": {"server": "log-parser", "test_sizes": ["1000lines"]},
    "filter_entries": {"server": "log-parser", "test_sizes": ["1000lines"]},
    "compute_log_statistics": {"server": "log-parser", "test_sizes": ["1000lines"]},
    "search_entries": {"server": "log-parser", "test_sizes": ["1000lines"]},
    "extract_time_range": {"server": "log-parser", "test_sizes": ["1000lines"]},

    # ===== summarize (3 tools) - SKIP: requires LLM API key =====
    # "summarize_text": {"server": "summarize", "test_sizes": ["default"]},
    # "summarize_documents": {"server": "summarize", "test_sizes": ["default"]},
    # "get_provider_info": {"server": "summarize", "test_sizes": ["default"]},

    # ===== time (2 tools) =====
    "get_current_time": {"server": "time", "test_sizes": ["default"]},
    "convert_time": {"server": "time", "test_sizes": ["default"]},

    # ===== fetch (1 tool) - SKIP: network dependent =====
    # "fetch": {"server": "fetch", "test_sizes": ["default"]},
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
    timing_pct: Dict[str, float]  # 각 컴포넌트의 비율 (%)
    measurements: List[float]

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
        wasm_total_ms = None
        cold_start_ms = None
        deser_ms = None

        # profiling: stderr에서 ---COLD_START---, ---DESER---, ---TIMING---, ---WASM_TOTAL--- 파싱
        if result.stderr:
            for line in result.stderr.strip().split('\n'):
                if line.startswith("---COLD_START---"):
                    try:
                        cold_start_ms = float(line[len("---COLD_START---"):])
                    except ValueError:
                        pass
                elif line.startswith("---DESER---"):
                    try:
                        deser_ms = float(line[len("---DESER---"):])
                    except ValueError:
                        pass
                elif line.startswith("---WASM_TOTAL---"):
                    try:
                        wasm_total_ms = float(line[len("---WASM_TOTAL---"):])
                    except ValueError:
                        pass
                elif line.startswith("---TIMING---"):
                    try:
                        timing_json = line[len("---TIMING---"):]
                        internal_timing = json.loads(timing_json)
                    except json.JSONDecodeError:
                        pass

        # internal_timing에 cold_start_ms, deser_ms, wasm_total_ms 추가
        if internal_timing is None:
            internal_timing = {}
        if cold_start_ms is not None:
            internal_timing["cold_start_ms"] = cold_start_ms
        if deser_ms is not None:
            internal_timing["deser_ms"] = deser_ms
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

    # 새로운 분해 방식:
    # cold_start_ms = WASM 로딩 시간 (---COLD_START---)
    # deser_ms = JSON 역직렬화 시간 (---DESER---)
    # fn_total_ms = deser + tool_exec (---TIMING---)
    # tool_exec_ms = fn_total - deser (순수 도구 실행)
    # io_ms = I/O 시간 (measure_io로 측정)
    # compute_ms = tool_exec - io
    timing = {
        "total_ms": statistics.mean(total_times),
        "cold_start_ms": 0.0,      # WASM 로딩 시간
        "deser_ms": 0.0,           # JSON 역직렬화 시간
        "fn_total_ms": 0.0,        # deser + tool_exec
        "tool_exec_ms": 0.0,       # 순수 도구 실행 시간
        "io_ms": 0.0,
        "compute_ms": 0.0,
    }

    timing_std = {
        "total_ms": statistics.stdev(total_times) if len(total_times) > 1 else 0.0,
        "cold_start_ms": 0.0,
        "deser_ms": 0.0,
        "fn_total_ms": 0.0,
        "tool_exec_ms": 0.0,
        "io_ms": 0.0,
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

        # cold_start_ms 파싱 (from ---COLD_START--- - 정확한 WASM 로딩 시간)
        cold_start_values = [t.get("cold_start_ms", 0) for t in raw_internal_timings]
        if any(v > 0 for v in cold_start_values):
            avg_val = statistics.mean(cold_start_values)
            std_val = statistics.stdev(cold_start_values) if len(cold_start_values) > 1 else 0.0
            timing["cold_start_ms"] = avg_val
            timing_std["cold_start_ms"] = std_val
            internal_timings_avg["cold_start_ms"] = avg_val

        # deser_ms 파싱 (from ---DESER--- - JSON 역직렬화 시간)
        deser_values = [t.get("deser_ms", 0) for t in raw_internal_timings]
        if any(v > 0 for v in deser_values):
            avg_val = statistics.mean(deser_values)
            std_val = statistics.stdev(deser_values) if len(deser_values) > 1 else 0.0
            timing["deser_ms"] = avg_val
            timing_std["deser_ms"] = std_val
            internal_timings_avg["deser_ms"] = avg_val

        # fn_total_ms, io_ms, compute_ms 파싱 (from ---TIMING---)
        for key in ["fn_total_ms", "io_ms", "compute_ms"]:
            values = [t.get(key, 0) for t in raw_internal_timings]
            if any(v > 0 for v in values):
                avg_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0.0
                internal_timings_avg[key] = avg_val
                timing[key] = avg_val
                timing_std[key] = std_val
            else:
                internal_timings_avg[key] = 0.0

        # tool_exec_ms 계산 (fn_total - deser = 순수 도구 실행 시간)
        timing["tool_exec_ms"] = max(0.0, timing["fn_total_ms"] - timing["deser_ms"])
        internal_timings_avg["tool_exec_ms"] = timing["tool_exec_ms"]

        # compute_ms 재계산 (tool_exec - io, Rust에서 fn_total - io로 잘못 계산되므로)
        timing["compute_ms"] = max(0.0, timing["tool_exec_ms"] - timing["io_ms"])
        internal_timings_avg["compute_ms"] = timing["compute_ms"]

        # wasm_total_ms 파싱 (from ---WASM_TOTAL---)
        wasm_total_values = [t.get("wasm_total_ms", 0) for t in raw_internal_timings]
        if any(v > 0 for v in wasm_total_values):
            internal_timings_avg["wasm_total_ms"] = statistics.mean(wasm_total_values)

        # cold_start가 없으면 fallback: total - fn_total
        if timing["cold_start_ms"] == 0.0:
            fn_total_ms = timing.get("fn_total_ms", 0.0)
            if fn_total_ms > 0:
                timing["cold_start_ms"] = max(0.0, timing["total_ms"] - fn_total_ms)
            else:
                timing["cold_start_ms"] = timing["total_ms"]
            internal_timings_avg["cold_start_ms"] = timing["cold_start_ms"]
    else:
        # internal timing이 없으면 전체 시간을 cold_start로 간주
        timing["cold_start_ms"] = timing["total_ms"]

    # profiling: 각 컴포넌트의 비율 (%) 계산
    # cold_start는 별도 측정 (비율 계산에서 제외)
    # fn_total = deser + tool_exec
    # tool_exec = io + compute
    fn_total = timing["fn_total_ms"]
    tool_exec = timing["tool_exec_ms"]
    timing_pct = {}
    if fn_total > 0:
        # fn_total 기준: deser + tool_exec = fn_total
        timing_pct["deser_pct"] = round(timing["deser_ms"] / fn_total * 100, 2)
        timing_pct["tool_exec_pct"] = round(tool_exec / fn_total * 100, 2)
    else:
        timing_pct["deser_pct"] = 0
        timing_pct["tool_exec_pct"] = 0
    if tool_exec > 0:
        # tool_exec 기준: io + compute = tool_exec
        timing_pct["io_pct"] = round(timing["io_ms"] / tool_exec * 100, 2)
        timing_pct["compute_pct"] = round(timing["compute_ms"] / tool_exec * 100, 2)
    else:
        timing_pct["io_pct"] = 0
        timing_pct["compute_pct"] = 0

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
        payload = {"repo_path": git_repo, "files": ["untracked.txt"]}
    elif tool_name == "git_create_branch":
        payload = {"repo_path": git_repo, "branch_name": "test-branch-new"}
    elif tool_name == "git_checkout":
        payload = {"repo_path": git_repo, "branch_name": "master"}

    # ===== image-resize tools =====
    elif tool_name in ["get_image_info", "compute_image_hash", "resize_image"]:
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
    elif tool_name == "aggregate_list":
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
    elif tool_name == "parse_logs":
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

    # ===== summarize tools =====
    elif tool_name == "summarize_text":
        payload = {"text": "This is a test document. " * 50, "max_length": 100}
    elif tool_name == "summarize_documents":
        payload = {"documents": ["Doc content " * 20], "max_length_per_doc": 100}
    elif tool_name == "get_provider_info":
        payload = {}

    # ===== time tools =====
    elif tool_name == "get_current_time":
        payload = {"timezone": "UTC"}
    elif tool_name == "convert_time":
        payload = {"source_timezone": "UTC", "time": "12:00", "target_timezone": "Asia/Seoul"}

    # ===== fetch tools =====
    elif tool_name == "fetch":
        payload = {"url": "https://httpbin.org/get"}

    else:
        print(f"Tool {tool_name} not yet configured", file=sys.stderr)
        return None

    # input_size가 0이면 payload 크기로 계산
    if input_size == 0 and payload:
        input_size = len(json.dumps(payload))

    print(f"\n[{tool_name}] Input: {input_size_label} ({input_size} bytes), Mode: {mode}")

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
        # cold_start: WASM 로딩 시간 (별도 측정)
        # fn_total = deser + tool_exec
        # tool_exec = io + compute
        summary["tools"][key] = {
            "tool_name": m.tool_name,
            "input_size": m.input_size,
            "input_size_label": m.input_size_label,
            "runs": m.runs,
            "timing_ms": {
                "total": round(m.timing["total_ms"], 3),
                "cold_start": round(m.timing["cold_start_ms"], 3),
                "fn_total": round(m.timing["fn_total_ms"], 3),
                "deser": round(m.timing["deser_ms"], 3),
                "tool_exec": round(m.timing["tool_exec_ms"], 3),
                "io": round(m.timing["io_ms"], 3),
                "compute": round(m.timing["compute_ms"], 3),
            },
            "timing_pct": m.timing_pct,
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
        node_name = get_node_name()
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

        if args.output:
            output_file = args.output
        else:
            output_file = RESULTS_DIR / node_name / f"measurements_{timestamp_str}.json"

        save_results(all_measurements, output_file)

        # Save summary file
        summary_file = RESULTS_DIR / node_name / f"summary_{timestamp_str}.json"
        save_summary(all_measurements, summary_file)
    else:
        print("\nNo measurements collected", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
