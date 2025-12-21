#!/usr/bin/env python3
"""
measure_io_size.py - Measure input/output data sizes for MCP tools

Measures:
- tool_name: Tool name
- input_size: JSON-RPC request payload size (bytes)
- output_size: JSON-RPC response size (bytes)
- execution_time_ms: Total execution time

Uses HTTP transport (wasmtime serve) for WASM execution.
"""

import argparse
import json
import os
import socket
import subprocess
import time
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).parent

WASM_MCP_PATH_CANDIDATES = [
    Path.home() / "EdgeAgent/wasm_mcp/target/wasm32-wasip2/release",
    Path.home() / "DDPS/undergraduated/CCGrid-2026/EdgeAgent/EdgeAgent/wasm_mcp/target/wasm32-wasip2/release",
]

WASM_PATH = None
for path in WASM_MCP_PATH_CANDIDATES:
    if path.exists():
        WASM_PATH = path
        break

TEST_DATA_DIR = SCRIPT_DIR.parent / "test_data"
RESULTS_DIR = SCRIPT_DIR.parent / "results"

TOOL_CONFIGS = {
    # filesystem
    "read_file": {"server": "filesystem", "test_sizes": ["500KB"]},
    "read_text_file": {"server": "filesystem", "test_sizes": ["500KB"]},
    "read_media_file": {"server": "filesystem", "test_sizes": ["default"]},
    "read_multiple_files": {"server": "filesystem", "test_sizes": ["default"]},
    "write_file": {"server": "filesystem", "test_sizes": ["500KB"]},
    "edit_file": {"server": "filesystem", "test_sizes": ["default"]},
    "create_directory": {"server": "filesystem", "test_sizes": ["default"]},
    "list_directory": {"server": "filesystem", "test_sizes": ["100files"]},
    "list_directory_with_sizes": {"server": "filesystem", "test_sizes": ["100files"]},
    "directory_tree": {"server": "filesystem", "test_sizes": ["default"]},
    "move_file": {"server": "filesystem", "test_sizes": ["default"]},
    "search_files": {"server": "filesystem", "test_sizes": ["default"]},
    "get_file_info": {"server": "filesystem", "test_sizes": ["default"]},
    "list_allowed_directories": {"server": "filesystem", "test_sizes": ["default"]},
    # git
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
    # image-resize
    "get_image_info": {"server": "image-resize", "test_sizes": ["default"]},
    "resize_image": {"server": "image-resize", "test_sizes": ["default"]},
    "scan_directory": {"server": "image-resize", "test_sizes": ["default"]},
    "compute_image_hash": {"server": "image-resize", "test_sizes": ["default"]},
    "compare_hashes": {"server": "image-resize", "test_sizes": ["default"]},
    "batch_resize": {"server": "image-resize", "test_sizes": ["default"]},
    # data-aggregate
    "aggregate_list": {"server": "data-aggregate", "test_sizes": ["default"]},
    "merge_summaries": {"server": "data-aggregate", "test_sizes": ["default"]},
    "combine_research_results": {"server": "data-aggregate", "test_sizes": ["default"]},
    "deduplicate": {"server": "data-aggregate", "test_sizes": ["default"]},
    "compute_trends": {"server": "data-aggregate", "test_sizes": ["default"]},
    # log-parser
    "parse_logs": {"server": "log-parser", "test_sizes": ["1000lines"]},
    "filter_entries": {"server": "log-parser", "test_sizes": ["1000lines"]},
    "compute_log_statistics": {"server": "log-parser", "test_sizes": ["1000lines"]},
    "search_entries": {"server": "log-parser", "test_sizes": ["1000lines"]},
    "extract_time_range": {"server": "log-parser", "test_sizes": ["1000lines"]},
    # time
    "get_current_time": {"server": "time", "test_sizes": ["default"]},
    "convert_time": {"server": "time", "test_sizes": ["default"]},
    # sequential-thinking
    "sequentialthinking": {"server": "sequential-thinking", "test_sizes": ["default"]},
    # summarize
    "summarize_text": {"server": "summarize", "test_sizes": ["default"], "needs_http": True},
    "summarize_documents": {"server": "summarize", "test_sizes": ["default"], "needs_http": True},
    "get_provider_info": {"server": "summarize", "test_sizes": ["default"], "needs_http": True},
    # fetch
    "fetch": {"server": "fetch", "test_sizes": ["default"], "needs_http": True},
}

# HTTP mode WASM files (wasmtime serve)
SERVER_WASM = {
    "filesystem": "mcp_server_filesystem_http.wasm",
    "git": "mcp_server_git_http.wasm",
    "image-resize": "mcp_server_image_resize_http.wasm",
    "data-aggregate": "mcp_server_data_aggregate_http.wasm",
    "log-parser": "mcp_server_log_parser_http.wasm",
    "time": "mcp_server_time_http.wasm",
    "sequential-thinking": "mcp_server_sequential_thinking_http.wasm",
    "summarize": "mcp_server_summarize_http.wasm",
    "fetch": "mcp_server_fetch_http.wasm",
}


def get_node_name() -> str:
    hostname = socket.gethostname().lower()
    if "rpi" in hostname or "raspberry" in hostname:
        return "device-rpi"
    elif "nuc" in hostname:
        return "edge-nuc"
    elif "orin" in hostname or "jetson" in hostname:
        return "edge-orin"
    return os.environ.get("NODE_NAME", hostname)


def create_jsonrpc_request(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Create a single JSON-RPC tools/call request for HTTP transport."""
    return json.dumps({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": 1,
        "params": {"name": tool_name, "arguments": arguments}
    })


# Server port allocation
BASE_PORT = 18100
_current_port = BASE_PORT


def get_next_port() -> int:
    """Get next available port."""
    global _current_port
    port = _current_port
    _current_port += 1
    return port


def find_free_port() -> int:
    """Find a free port on the system."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


class WasmServer:
    """Manages a wasmtime serve process (matches measure_tools.py approach)."""

    def __init__(self, wasm_path: Path, port: int, allowed_dirs: List[str], needs_http: bool = False):
        self.wasm_path = wasm_path
        self.port = port
        self.allowed_dirs = allowed_dirs
        self.needs_http = needs_http
        self.process: Optional[subprocess.Popen] = None
        self.url = f"http://127.0.0.1:{port}/"
        self._start_error = ""

    def start(self, timeout: float = 10.0) -> bool:
        """Start the wasmtime serve process."""
        # Build dir args (simple format like measure_tools.py)
        dir_args = []
        for d in self.allowed_dirs:
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

        # Build command (matching measure_tools.py)
        cmd = ["wasmtime", "serve", "--addr", f"127.0.0.1:{self.port}"]
        cmd.extend(["-S", "cli"])  # Enable CLI interface for env vars
        if self.needs_http:
            cmd.extend(["-S", "http"])
        cmd.extend(["--env", "OPENAI_API_KEY"])
        cmd.extend(["--env", "UPSTAGE_API_KEY"])
        cmd.extend(["--env", "SUMMARIZE_PROVIDER"])
        cmd.extend(dir_args)
        cmd.append(str(self.wasm_path))

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
        except Exception as e:
            self._start_error = str(e)
            return False

        # Wait for server to be ready (poll the port)
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check if process died
            if self.process.poll() is not None:
                try:
                    stderr = self.process.stderr.read().decode()[:500]
                except:
                    stderr = ""
                self._start_error = stderr or f"Process exited with code {self.process.returncode}"
                return False

            # Try to connect using requests
            try:
                requests.get(self.url, timeout=0.5)
                return True
            except requests.exceptions.ConnectionError:
                time.sleep(0.05)
            except Exception:
                time.sleep(0.05)

        # Check one more time if process is still alive
        if self.process.poll() is None:
            return True  # Process running, assume ready

        try:
            stderr = self.process.stderr.read().decode()[:500]
        except:
            stderr = ""
        self._start_error = stderr or "Timeout waiting for server"
        return False

    def stop(self):
        """Stop the wasmtime serve process."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> tuple:
        """Call a tool via HTTP and return (input_size, output_size, exec_time_ms, success, error)."""
        request_body = create_jsonrpc_request(tool_name, arguments)
        input_size = len(request_body.encode('utf-8'))

        try:
            start = time.perf_counter()
            response = requests.post(
                self.url,
                data=request_body,
                headers={'Content-Type': 'application/json'},
                timeout=120.0
            )
            exec_time = (time.perf_counter() - start) * 1000

            output_size = len(response.content)

            # Check for JSON-RPC error
            try:
                result = response.json()
                if "error" in result:
                    return input_size, output_size, exec_time, False, result["error"].get("message", "Unknown error")
            except:
                pass

            return input_size, output_size, exec_time, True, None

        except requests.exceptions.HTTPError as e:
            return input_size, 0, 0, False, f"HTTP Error: {e}"
        except requests.exceptions.ConnectionError as e:
            return input_size, 0, 0, False, f"Connection Error: {e}"
        except Exception as e:
            return input_size, 0, 0, False, str(e)


def get_test_file_path(size_label: str) -> Path:
    """Get test file path for given size, create or recreate if wrong size"""
    file_path = TEST_DATA_DIR / "files" / f"test_{size_label}.txt"

    # Create directory if needed
    file_path.parent.mkdir(parents=True, exist_ok=True)

    size_map = {"1KB": 1024, "10KB": 10240, "100KB": 102400, "500KB": 512000, "1MB": 1048576, "10MB": 10485760}
    size_bytes = size_map.get(size_label, 1024)

    # Create file if missing or wrong size
    needs_create = not file_path.exists()
    if file_path.exists() and file_path.stat().st_size != size_bytes:
        needs_create = True
        print(f"  Recreating test file (wrong size): {file_path}")

    if needs_create:
        file_path.write_text("x" * size_bytes)
        print(f"  Created test file: {file_path} ({size_label}, {size_bytes} bytes)")

    return file_path


def get_test_directory_path(size_label: str) -> Path:
    return TEST_DATA_DIR / "directories" / size_label


def get_test_image_path() -> Path:
    """Get test image path, create small PNG if missing or too large (>10KB)"""
    images_dir = TEST_DATA_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    image_path = images_dir / "test.png"
    # Recreate if missing or too large (we want small ~500 byte PNG)
    needs_create = not image_path.exists()
    if image_path.exists() and image_path.stat().st_size > 10240:  # > 10KB
        needs_create = True
        print(f"  Recreating test image (too large): {image_path}")

    if needs_create:
        # Create a small 100x100 PNG (~500 bytes)
        import struct
        import zlib

        width, height = 100, 100

        def png_chunk(chunk_type, data):
            chunk_len = struct.pack('>I', len(data))
            chunk_crc = struct.pack('>I', zlib.crc32(chunk_type + data) & 0xffffffff)
            return chunk_len + chunk_type + data + chunk_crc

        png_data = b'\x89PNG\r\n\x1a\n'
        ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
        png_data += png_chunk(b'IHDR', ihdr_data)

        raw_data = b''
        for _ in range(height):
            raw_data += b'\x00' + b'\x80\x80\x80' * width
        compressed = zlib.compress(raw_data, 9)
        png_data += png_chunk(b'IDAT', compressed)
        png_data += png_chunk(b'IEND', b'')

        image_path.write_bytes(png_data)
        print(f"  Created test image: {image_path} ({len(png_data)} bytes)")

    return image_path


def generate_log_content(num_lines: int) -> str:
    import random
    lines = []
    for i in range(num_lines):
        ip = f"192.168.1.{random.randint(1,255)}"
        ts = f"20/Dec/2025:10:{i%60:02d}:{i%60:02d} +0000"
        lines.append(f'{ip} - - [{ts}] "GET /api HTTP/1.1" 200 {random.randint(100,5000)} "-" "Mozilla"')
    return "\n".join(lines)


# Server cache: reuse servers for same WASM file
_server_cache: Dict[str, WasmServer] = {}


def prepare_payload(tool_name: str, size_label: str) -> tuple:
    config = TOOL_CONFIGS.get(tool_name)
    if not config:
        return None, None, False

    allowed_dirs = ["/tmp", str(TEST_DATA_DIR)]
    needs_http = config.get("needs_http", False)
    payload = None

    if tool_name in ["read_file", "read_text_file"]:
        f = get_test_file_path(size_label)
        if f.exists(): payload = {"path": str(f)}
    elif tool_name == "read_media_file":
        f = get_test_image_path()
        payload = {"path": str(f)}
    elif tool_name == "read_multiple_files":
        f = get_test_file_path("1KB")
        if f.exists(): payload = {"paths": [str(f)]}
    elif tool_name == "write_file":
        sz = {"1KB": 1024, "10KB": 10240, "100KB": 102400, "500KB": 512000, "1MB": 1048576, "10MB": 10485760}.get(size_label, 1024)
        payload = {"path": f"/tmp/io_test_{size_label}.txt", "content": "x" * sz}
    elif tool_name == "edit_file":
        f = get_test_file_path("1KB")
        if f.exists(): payload = {"path": str(f), "edits": "test", "dry_run": True}
    elif tool_name == "create_directory":
        payload = {"path": "/tmp/io_test_dir"}
    elif tool_name in ["list_directory", "list_directory_with_sizes"]:
        d = get_test_directory_path(size_label)
        if d.exists(): payload = {"path": str(d)}
    elif tool_name == "directory_tree":
        payload = {"path": str(TEST_DATA_DIR)}
    elif tool_name == "move_file":
        payload = {"source": "/tmp/x", "destination": "/tmp/y"}
    elif tool_name == "search_files":
        payload = {"path": str(TEST_DATA_DIR), "pattern": "*.txt"}
    elif tool_name == "get_file_info":
        f = get_test_file_path("1KB")
        if f.exists(): payload = {"path": str(f)}
    elif tool_name == "list_allowed_directories":
        payload = {}
    elif tool_name == "git_status":
        payload = {"repo_path": str(TEST_DATA_DIR / "git_repo")}
    elif tool_name == "git_log":
        payload = {"repo_path": str(TEST_DATA_DIR / "git_repo"), "max_count": 10}
    elif tool_name == "git_show":
        payload = {"repo_path": str(TEST_DATA_DIR / "git_repo"), "revision": "HEAD"}
    elif tool_name == "git_branch":
        payload = {"repo_path": str(TEST_DATA_DIR / "git_repo"), "list_all": True}
    elif tool_name in ["git_diff_unstaged", "git_diff_staged", "git_reset"]:
        payload = {"repo_path": str(TEST_DATA_DIR / "git_repo")}
    elif tool_name == "git_diff":
        payload = {"repo_path": str(TEST_DATA_DIR / "git_repo"), "target": "HEAD"}
    elif tool_name == "git_commit":
        payload = {"repo_path": str(TEST_DATA_DIR / "git_repo"), "message": "test"}
    elif tool_name == "git_add":
        payload = {"repo_path": str(TEST_DATA_DIR / "git_repo"), "files": ["."]}
    elif tool_name == "git_create_branch":
        payload = {"repo_path": str(TEST_DATA_DIR / "git_repo"), "branch_name": "test"}
    elif tool_name == "git_checkout":
        payload = {"repo_path": str(TEST_DATA_DIR / "git_repo"), "branch_name": "main"}
    elif tool_name in ["get_image_info", "compute_image_hash"]:
        f = get_test_image_path()
        payload = {"path": str(f)}
    elif tool_name == "resize_image":
        f = get_test_image_path()
        payload = {"path": str(f), "width": 100, "height": 100}
    elif tool_name == "scan_directory":
        get_test_image_path()  # Ensure image exists
        payload = {"path": str(TEST_DATA_DIR / "images")}
    elif tool_name == "compare_hashes":
        payload = {"hash1": "abc123", "hash2": "abc123"}
    elif tool_name == "batch_resize":
        payload = {"directory": str(TEST_DATA_DIR / "images"), "width": 100, "height": 100}
    elif tool_name == "aggregate_list":
        payload = {"data": [{"v": i, "c": f"c{i%5}"} for i in range(10000)], "group_by": "c", "operation": "sum", "field": "v"}
    elif tool_name == "merge_summaries":
        payload = {"summaries": [{"topic": f"T{i}", "summary": f"S{i}"*100} for i in range(100)]}
    elif tool_name == "combine_research_results":
        payload = {"results": [{"source": f"S{i}", "findings": f"F{i}"*200, "score": i*0.1} for i in range(500)], "strategy": "weighted"}
    elif tool_name == "deduplicate":
        payload = {"items": [{"id": i%100, "content": f"C{i%100}"*50} for i in range(1000)], "key": "id"}
    elif tool_name == "compute_trends":
        payload = {"data": [{"timestamp": f"2025-01-{(i%28)+1:02d}", "value": i*1.5} for i in range(10000)], "time_field": "timestamp", "value_field": "value"}
    elif tool_name == "parse_logs":
        n = {"100lines": 100, "1000lines": 1000}.get(size_label, 1000)
        payload = {"log_content": generate_log_content(n), "format": "apache_combined"}
    elif tool_name == "filter_entries":
        n = {"100lines": 100, "1000lines": 1000}.get(size_label, 1000)
        payload = {"entries": [{"level": ["INFO","WARN","ERROR"][i%3], "message": f"M{i}"} for i in range(n)], "level": "ERROR"}
    elif tool_name == "compute_log_statistics":
        n = {"100lines": 100, "1000lines": 1000}.get(size_label, 1000)
        payload = {"entries": [{"level": "INFO", "response_time": i*10} for i in range(n)]}
    elif tool_name == "search_entries":
        n = {"100lines": 100, "1000lines": 1000}.get(size_label, 1000)
        payload = {"entries": [{"message": f"M{i} error" if i%10==0 else f"M{i}"} for i in range(n)], "query": "error"}
    elif tool_name == "extract_time_range":
        n = {"100lines": 100, "1000lines": 1000}.get(size_label, 1000)
        payload = {"entries": [{"timestamp": f"2025-01-01T{i%24:02d}:00:00Z", "message": f"L{i}"} for i in range(n)], "start": "2025-01-01T10:00:00Z", "end": "2025-01-01T12:00:00Z"}
    elif tool_name == "get_current_time":
        payload = {"timezone": "UTC"}
    elif tool_name == "convert_time":
        payload = {"time": "2025-01-01T12:00:00Z", "from_tz": "UTC", "to_tz": "Asia/Seoul"}
    elif tool_name == "sequentialthinking":
        payload = {"thought": "Test thought.", "thoughtNumber": 1, "totalThoughts": 3, "nextThoughtNeeded": True}
    elif tool_name == "summarize_text":
        payload = {"text": "Test text. " * 50}
    elif tool_name == "summarize_documents":
        payload = {"paths": [str(get_test_file_path("1KB"))]}
    elif tool_name == "get_provider_info":
        payload = {}
    elif tool_name == "fetch":
        payload = {"url": "https://httpbin.org/get"}

    return payload, allowed_dirs, needs_http


def get_or_create_server(wasm_file: Path, allowed_dirs: List[str], needs_http: bool) -> WasmServer:
    """Get existing server or create a new one."""
    cache_key = str(wasm_file)
    if cache_key in _server_cache:
        return _server_cache[cache_key]

    port = find_free_port()
    server = WasmServer(wasm_file, port, allowed_dirs, needs_http)
    if server.start():
        _server_cache[cache_key] = server
        return server
    return server


def cleanup_servers():
    """Stop all cached servers."""
    for server in _server_cache.values():
        server.stop()
    _server_cache.clear()


def measure_tool(tool_name: str, size_label: str = "default") -> dict:
    """Measure tool I/O sizes using HTTP transport."""
    config = TOOL_CONFIGS.get(tool_name)
    if not config:
        return {"tool_name": tool_name, "error": "Unknown tool", "success": False}

    wasm_file = WASM_PATH / SERVER_WASM.get(config["server"], "")
    if not wasm_file.exists():
        return {"tool_name": tool_name, "error": f"WASM not found: {wasm_file}", "success": False}

    payload, allowed_dirs, needs_http = prepare_payload(tool_name, size_label)
    if payload is None:
        img_path = TEST_DATA_DIR / "images" / "test.png"
        return {"tool_name": tool_name, "error": f"Payload failed (test_image exists: {img_path.exists()}, path: {img_path})", "success": False}

    # Get or create server
    server = get_or_create_server(wasm_file, allowed_dirs, needs_http)
    if server.process is None or server.process.poll() is not None:
        # Server failed to start or died
        error_msg = getattr(server, '_start_error', '') or "Unknown error"
        return {"tool_name": tool_name, "error": f"Server failed to start: {error_msg}", "success": False}

    # Call tool via HTTP
    input_size, output_size, exec_time, success, error = server.call_tool(tool_name, payload)

    return {
        "tool_name": tool_name,
        "input_size": input_size,
        "output_size": output_size,
        "execution_time_ms": round(exec_time, 2),
        "success": success,
        "error": error
    }


def save_results(results: List[dict], output_file: Path):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "node": get_node_name(),
        "timestamp": datetime.now().isoformat(),
        "total_tools": len(results),
        "tools": {r["tool_name"]: {
            "tool_name": r["tool_name"],
            "input_size": r.get("input_size", 0),
            "output_size": r.get("output_size", 0),
            "execution_time_ms": r.get("execution_time_ms", 0),
        } for r in results if r.get("success")}
    }
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Measure MCP tool I/O sizes")
    parser.add_argument("--tool", "-t", help="Specific tool")
    parser.add_argument("--output", "-o", type=Path, help="Output file")
    parser.add_argument("--list", "-l", action="store_true", help="List tools")
    args = parser.parse_args()

    if args.list:
        for name in TOOL_CONFIGS:
            print(f"  {name}")
        return

    print("=" * 70)
    print("MCP Tool I/O Size Profiler")
    print(f"Node: {get_node_name()}")
    print(f"WASM_PATH: {WASM_PATH}")
    print(f"WASM_PATH exists: {WASM_PATH.exists() if WASM_PATH else False}")
    for candidate in WASM_MCP_PATH_CANDIDATES:
        print(f"  Candidate: {candidate} -> exists: {candidate.exists()}")
    print(f"TEST_DATA_DIR: {TEST_DATA_DIR}")
    print(f"TEST_DATA_DIR exists: {TEST_DATA_DIR.exists()}")
    print(f"  files/test_500KB.txt: {(TEST_DATA_DIR / 'files' / 'test_500KB.txt').exists()}")
    print(f"  images/test.png: {(TEST_DATA_DIR / 'images' / 'test.png').exists()}")
    print("Transport: HTTP (wasmtime serve)")
    print("=" * 70)

    results = []
    tools = [args.tool] if args.tool else list(TOOL_CONFIGS.keys())

    for name in tools:
        cfg = TOOL_CONFIGS.get(name)
        if not cfg:
            continue
        size = cfg["test_sizes"][0]
        print(f"[{name}]...", end=" ", flush=True)
        r = measure_tool(name, size)
        results.append(r)
        if r.get("success"):
            print(f"IN:{r['input_size']:,}B OUT:{r['output_size']:,}B ({r['execution_time_ms']:.0f}ms)")
        else:
            print(f"FAILED: {r.get('error')}")

    if results:
        success_count = sum(1 for r in results if r.get("success"))
        fail_count = len(results) - success_count

        print("\n" + "=" * 70)
        print(f"Results: {success_count}/{len(results)} succeeded, {fail_count} failed")
        print("=" * 70)

        if success_count > 0:
            out = args.output or (RESULTS_DIR / f"io_sizes_{get_node_name()}.json")
            save_results(results, out)

            print(f"{'Tool':<30} {'Input':>12} {'Output':>12} {'Time':>10}")
            print("-" * 70)
            for r in sorted(results, key=lambda x: x.get("output_size", 0), reverse=True):
                if r.get("success"):
                    print(f"{r['tool_name']:<30} {r['input_size']:>12,} {r['output_size']:>12,} {r['execution_time_ms']:>10.0f}")

    # Cleanup servers
    cleanup_servers()


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_servers()
