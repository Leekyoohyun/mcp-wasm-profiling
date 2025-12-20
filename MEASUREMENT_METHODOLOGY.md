# WASM MCP Tool Timing Measurement Methodology

## Overview

This document describes the timing decomposition model used for profiling WASM MCP tools.

## Timing Model

```
Total = Startup + JSONParse + ToolExec
ToolExec = IO + Compute
IO = DiskIO | NetworkIO (mutually exclusive based on tool type)
```

## Component Definitions

| Component | Measurement Method | Description |
|-----------|-------------------|-------------|
| **Total** | Python `time.perf_counter()` | End-to-end time from process start to response received |
| **Startup** | Python measured | `wasmtime serve` process start → HTTP server ready |
| **JSONParse** | Calculated | `Total - Startup - ToolExec` (remaining time) |
| **ToolExec** | WASM header `X-Tool-Exec-Ms` | Pure tool execution time inside WASM |
| **DiskIO** | WASM header `X-IO-Ms` | File system I/O time (for disk-bound tools) |
| **NetworkIO** | WASM header `X-IO-Ms` | Network I/O time (for network-bound tools) |
| **Compute** | Calculated | `ToolExec - IO` |
| **Memory** | cgroups / psutil | Peak memory usage of wasmtime process |

## Measurement Flow (HTTP Mode)

```
┌─────────────────────────────────────────────────────────────────┐
│                         Total Time                               │
├──────────────────┬─────────────────┬────────────────────────────┤
│     Startup      │   JSONParse     │         ToolExec           │
│                  │                 ├──────────────┬─────────────┤
│                  │                 │     IO       │   Compute   │
└──────────────────┴─────────────────┴──────────────┴─────────────┘

Timeline:
  t0: subprocess.Popen(wasmtime serve)     ─┐
      ...                                   │ Startup
  t1: HTTP server ready (first response)   ─┘
  t2: POST request sent                    ─┐
      - JSON-RPC parsing                    │ JSONParse + ToolExec
      - Tool execution (IO + Compute)       │
  t3: Response received                    ─┘
```

## WASM-side Timing Instrumentation

### Rust Implementation

```rust
// In wasmmcp/src/timing.rs
thread_local! {
    static IO_DURATION: RefCell<Duration> = RefCell::new(Duration::ZERO);
}

pub fn measure_io<F, T>(f: F) -> T
where F: FnOnce() -> T {
    let start = Instant::now();
    let result = f();
    IO_DURATION.with(|d| *d.borrow_mut() += start.elapsed());
    result
}
```

### Usage in Tool Code

```rust
// Wrap I/O operations
let content = measure_io(|| fs::read_to_string(&path))?;
let data = measure_io(|| http_client.get(url))?;
```

### HTTP Response Headers

WASM tools emit timing via HTTP headers:

| Header | Description |
|--------|-------------|
| `X-WASM-Total-Ms` | Total WASM execution time |
| `X-JSON-Parse-Ms` | JSON parsing time |
| `X-Tool-Exec-Ms` | Tool execution time |
| `X-IO-Ms` | I/O time (disk or network) |

## Tool I/O Classification

| I/O Type | Tools |
|----------|-------|
| **Disk** | filesystem/*, git/*, image-resize/* |
| **Network** | summarize/*, fetch/* |
| **None** | data-aggregate/*, log-parser/*, time/*, sequential-thinking/* |

## Memory Measurement

Priority order:
1. **cgroups v2**: `/sys/fs/cgroup{path}/memory.current`
2. **cgroups v1**: `/sys/fs/cgroup/memory{path}/memory.usage_in_bytes`
3. **psutil fallback**: `psutil.Process(pid).memory_info().rss`

## Example Output

```json
{
  "tool_name": "read_file",
  "timing_ms": {
    "total": 245.123,
    "startup": 180.456,
    "json_parse": 12.345,
    "tool_exec": 52.322,
    "disk_io": 48.100,
    "network_io": 0.0,
    "compute": 4.222
  },
  "timing_pct": {
    "disk_io_pct": 91.93,
    "network_io_pct": 0.0,
    "compute_pct": 8.07
  },
  "memory_mb": 45.2
}
```

## Scripts

| Script | Description |
|--------|-------------|
| `measure_tools.py` | Main profiling script |
| `setup_test_data.sh` | Generate test data (files, git repo, images) |

## Usage

```bash
# Measure all tools (HTTP mode, cold start)
python scripts/measure_tools.py --transport http

# Measure specific tool
python scripts/measure_tools.py --tool read_file --runs 5

# Debug mode (show timing headers)
DEBUG=1 python scripts/measure_tools.py --tool git_diff
```
