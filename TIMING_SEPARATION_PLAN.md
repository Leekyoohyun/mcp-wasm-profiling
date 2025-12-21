# Disk I/O vs Network I/O Timing Separation Plan

## Current State

### timing.rs
```rust
thread_local! {
    static IO_ACCUMULATOR: RefCell<Duration> = RefCell::new(Duration::ZERO);
}

pub fn measure_io<F, T>(f: F) -> T { ... }  // Single function for all I/O
pub fn get_io_duration() -> Duration { ... }
```

### Problem
- Single `measure_io()` function used for both disk I/O and network I/O
- Cannot distinguish between `disk_io` and `network_io` in profiling results
- Python profiling script uses `io_type` column to classify, but this is metadata-based, not measured

## Goal
Separate accumulation of disk I/O time and network I/O time in WASM Rust code.

---

## Implementation Plan

### Step 1: Update `timing.rs` - Add Separate Accumulators

```rust
thread_local! {
    static DISK_IO_ACCUMULATOR: RefCell<Duration> = RefCell::new(Duration::ZERO);
    static NETWORK_IO_ACCUMULATOR: RefCell<Duration> = RefCell::new(Duration::ZERO);
}

/// Reset both I/O accumulators (call before each tool execution)
pub fn reset_io_accumulators() {
    DISK_IO_ACCUMULATOR.with(|acc| *acc.borrow_mut() = Duration::ZERO);
    NETWORK_IO_ACCUMULATOR.with(|acc| *acc.borrow_mut() = Duration::ZERO);
}

/// Add duration to disk I/O accumulator
pub fn add_disk_io_duration(duration: Duration) {
    DISK_IO_ACCUMULATOR.with(|acc| *acc.borrow_mut() += duration);
}

/// Add duration to network I/O accumulator
pub fn add_network_io_duration(duration: Duration) {
    NETWORK_IO_ACCUMULATOR.with(|acc| *acc.borrow_mut() += duration);
}

/// Get accumulated disk I/O duration
pub fn get_disk_io_duration() -> Duration {
    DISK_IO_ACCUMULATOR.with(|acc| *acc.borrow())
}

/// Get accumulated network I/O duration
pub fn get_network_io_duration() -> Duration {
    NETWORK_IO_ACCUMULATOR.with(|acc| *acc.borrow())
}

/// Measure a disk I/O operation
pub fn measure_disk_io<F, T>(f: F) -> T
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let result = f();
    add_disk_io_duration(start.elapsed());
    result
}

/// Measure a network I/O operation
pub fn measure_network_io<F, T>(f: F) -> T
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let result = f();
    add_network_io_duration(start.elapsed());
    result
}

// Keep backward compatibility
pub fn measure_io<F, T>(f: F) -> T { measure_disk_io(f) }
pub fn get_io_duration() -> Duration { get_disk_io_duration() }
```

### Step 2: Update `ToolTiming` struct

```rust
#[derive(Debug, Clone)]
pub struct ToolTiming {
    pub tool_name: String,
    pub fn_total_ms: f64,
    pub disk_io_ms: f64,     // Renamed from io_ms
    pub network_io_ms: f64,  // New field
    pub compute_ms: f64,
}

impl ToolTiming {
    pub fn output(&self) {
        let json = serde_json::json!({
            "tool": self.tool_name,
            "fn_total_ms": self.fn_total_ms,
            "disk_io_ms": self.disk_io_ms,
            "network_io_ms": self.network_io_ms,
            "compute_ms": self.compute_ms,
        });
        eprintln!("---TIMING---{}", json);
    }
}
```

### Step 3: Update `ToolTimer::finish()`

```rust
pub fn finish(self, tool_name: &str) -> ToolTiming {
    let elapsed = self.start.elapsed();
    let disk_io = get_disk_io_duration();
    let network_io = get_network_io_duration();

    let fn_total_ms = elapsed.as_secs_f64() * 1000.0;
    let disk_io_ms = disk_io.as_secs_f64() * 1000.0;
    let network_io_ms = network_io.as_secs_f64() * 1000.0;
    let compute_ms = (fn_total_ms - disk_io_ms - network_io_ms).max(0.0);

    ToolTiming {
        tool_name: tool_name.to_string(),
        fn_total_ms,
        disk_io_ms,
        network_io_ms,
        compute_ms,
    }
}
```

### Step 4: Update `builder.rs` Output

```rust
pub fn handle_tools_call(&self, name: &str, args: Value) -> Result<Value, String> {
    reset_io_accumulators();  // Reset both accumulators

    let result = self.registry.call(name, args)?;

    // Output separate I/O timings
    let disk_io_ms = get_disk_io_duration().as_secs_f64() * 1000.0;
    let network_io_ms = get_network_io_duration().as_secs_f64() * 1000.0;
    eprintln!("---DISK_IO---{:.3}", disk_io_ms);
    eprintln!("---NETWORK_IO---{:.3}", network_io_ms);

    // ... rest of the function
}
```

### Step 5: Update Server Tool Implementations

#### Filesystem tools (`servers/filesystem/src/tools.rs`)
Change all `measure_io` calls to `measure_disk_io`:
```rust
// Before
let content = measure_io(|| fs::read_to_string(path))?;

// After
let content = measure_disk_io(|| fs::read_to_string(path))?;
```

#### Git tools (`servers/git/src/tools.rs`)
Change all `measure_io` calls to `measure_disk_io`:
```rust
// All file operations use measure_disk_io
measure_disk_io(|| fs::read_to_string(&head_path))
measure_disk_io(|| fs::read(&loose_path))
measure_disk_io(|| File::open(idx_path))
```

#### Summarize tools (`servers/summarize/src/tools.rs`)
Change `measure_io` to `measure_network_io`:
```rust
// Before
let (status, content_str) = measure_io(|| {
    // HTTP request logic
})?;

// After
let (status, content_str) = measure_network_io(|| {
    // HTTP request logic
})?;
```

#### Fetch tools (`servers/fetch/src/tools.rs`) - if exists
Use `measure_network_io` for HTTP operations.

### Step 6: Update prelude exports

```rust
// In lib.rs prelude
pub use crate::timing::{
    measure_disk_io,
    measure_network_io,
    measure_io,  // backward compat
    ToolTimer
};
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `wasmmcp/src/timing.rs` | Add separate accumulators and measurement functions |
| `wasmmcp/src/builder.rs` | Output `---DISK_IO---` and `---NETWORK_IO---` |
| `wasmmcp/src/lib.rs` | Export new functions in prelude |
| `servers/filesystem/src/tools.rs` | `measure_io` -> `measure_disk_io` |
| `servers/git/src/tools.rs` | `measure_io` -> `measure_disk_io` |
| `servers/summarize/src/tools.rs` | `measure_io` -> `measure_network_io` |
| `servers/fetch/src/tools.rs` | `measure_io` -> `measure_network_io` |

---

## Expected Output Format

### Before
```
---IO---1.234
```

### After
```
---DISK_IO---1.234
---NETWORK_IO---0.000
```

or

```
---DISK_IO---0.000
---NETWORK_IO---45.678
```

---

## Python Profiling Script Changes

Parse the new output format:
```python
# Parse timing from stderr
for line in stderr.split('\n'):
    if line.startswith('---DISK_IO---'):
        disk_io_ms = float(line[13:])
    elif line.startswith('---NETWORK_IO---'):
        network_io_ms = float(line[16:])
```

Remove `io_type` classification - now measured directly.
