#!/usr/bin/env python3
"""
analyze_results.py - Analyze WASM MCP profiling results

Generates comparison tables and visualizations for:
- Cold start vs Warm start comparison
- Node-wise performance comparison
- Time decomposition analysis (cold_start, fn_total, io, compute)
- Input size scaling analysis
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import statistics

# Try to import pandas and matplotlib (optional)
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# Configuration
# =============================================================================

RESULTS_DIR = Path(__file__).parent.parent / "results"

NODE_ORDER = ["device-rpi", "edge-nuc", "edge-orin", "cloud"]

COLORS = {
    "device-rpi": "#e74c3c",    # Red
    "edge-nuc": "#3498db",      # Blue
    "edge-orin": "#2ecc71",     # Green
    "cloud": "#9b59b6",         # Purple
}


# =============================================================================
# Data Loading
# =============================================================================

def load_all_results(results_dir: Path = RESULTS_DIR) -> List[Dict]:
    """Load all measurement results from all nodes"""
    all_results = []

    for node_dir in results_dir.iterdir():
        if not node_dir.is_dir():
            continue

        node_name = node_dir.name

        for result_file in node_dir.glob("*.json"):
            try:
                with open(result_file) as f:
                    data = json.load(f)

                # Handle both single and array formats
                if isinstance(data, list):
                    all_results.extend(data)
                else:
                    all_results.append(data)
            except Exception as e:
                print(f"Warning: Failed to load {result_file}: {e}", file=sys.stderr)

    return all_results


def load_node_results(node_name: str, results_dir: Path = RESULTS_DIR) -> List[Dict]:
    """Load results for a specific node"""
    node_dir = results_dir / node_name
    if not node_dir.exists():
        return []

    results = []
    for result_file in node_dir.glob("*.json"):
        try:
            with open(result_file) as f:
                data = json.load(f)
            if isinstance(data, list):
                results.extend(data)
            else:
                results.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {result_file}: {e}", file=sys.stderr)

    return results


# =============================================================================
# Analysis Functions
# =============================================================================

def get_timing(r: Dict) -> Dict:
    """Get timing dict from result, supporting both timing_ms and timing keys"""
    return r.get("timing_ms") or r.get("timing", {})


def get_total_ms(r: Dict) -> float:
    """Get total_ms from result, supporting both field name formats"""
    timing = get_timing(r)
    return timing.get("total") or timing.get("total_ms", 0)


def summarize_by_tool(results: List[Dict]) -> Dict[str, Dict]:
    """Summarize results grouped by tool name"""
    by_tool = defaultdict(list)

    for r in results:
        tool_name = r.get("tool_name", "unknown")
        by_tool[tool_name].append(r)

    summary = {}
    for tool_name, measurements in by_tool.items():
        total_times = [get_total_ms(m) for m in measurements if get_timing(m)]

        summary[tool_name] = {
            "count": len(measurements),
            "mean_ms": statistics.mean(total_times) if total_times else 0,
            "std_ms": statistics.stdev(total_times) if len(total_times) > 1 else 0,
            "min_ms": min(total_times) if total_times else 0,
            "max_ms": max(total_times) if total_times else 0,
        }

    return summary


def summarize_by_node(results: List[Dict]) -> Dict[str, Dict]:
    """Summarize results grouped by node"""
    by_node = defaultdict(list)

    for r in results:
        node = r.get("node", "unknown")
        by_node[node].append(r)

    summary = {}
    for node, measurements in by_node.items():
        total_times = [get_total_ms(m) for m in measurements if get_timing(m)]

        summary[node] = {
            "count": len(measurements),
            "mean_ms": statistics.mean(total_times) if total_times else 0,
            "std_ms": statistics.stdev(total_times) if len(total_times) > 1 else 0,
            "tools_measured": len(set(m.get("tool_name", "") for m in measurements)),
        }

    return summary


def compare_cold_warm(results: List[Dict]) -> Dict[str, Dict]:
    """Compare cold start vs warm start performance"""
    comparison = {}

    # Group by tool and input size
    grouped = defaultdict(lambda: {"cold": [], "warm": []})

    for r in results:
        tool_name = r.get("tool_name", "unknown")
        size_label = r.get("input_size_label", "unknown")
        mode = r.get("mode", "cold")
        key = f"{tool_name}:{size_label}"

        if get_timing(r):
            grouped[key][mode].append(get_total_ms(r))

    for key, modes in grouped.items():
        cold_times = modes["cold"]
        warm_times = modes["warm"]

        if cold_times and warm_times:
            cold_mean = statistics.mean(cold_times)
            warm_mean = statistics.mean(warm_times)
            speedup = cold_mean / warm_mean if warm_mean > 0 else float('inf')

            comparison[key] = {
                "cold_mean_ms": cold_mean,
                "warm_mean_ms": warm_mean,
                "speedup": speedup,
                "cold_overhead_ms": cold_mean - warm_mean,
            }

    return comparison


def analyze_time_decomposition(results: List[Dict]) -> Dict[str, Dict]:
    """Analyze time decomposition (startup, json_parse, tool_exec, disk_io, network_io, compute)"""
    decomposition = {}

    for r in results:
        tool_name = r.get("tool_name", "unknown")
        size_label = r.get("input_size_label", "unknown")
        key = f"{tool_name}:{size_label}"

        # timing_ms 또는 timing 키 지원
        timing = r.get("timing_ms") or r.get("timing", {})
        if timing:
            # _ms 접미사 있는 키와 없는 키 모두 지원
            total = timing.get("total") or timing.get("total_ms", 0)
            startup = timing.get("startup") or timing.get("startup_ms", 0)
            json_parse = timing.get("json_parse") or timing.get("json_parse_ms", 0)
            tool_exec = timing.get("tool_exec") or timing.get("tool_exec_ms", 0)
            disk_io = timing.get("disk_io") or timing.get("disk_io_ms", 0)
            network_io = timing.get("network_io") or timing.get("network_io_ms", 0)
            compute = timing.get("compute") or timing.get("compute_ms", 0)

            if total > 0:
                decomposition[key] = {
                    "total_ms": total,
                    "startup_ms": startup,
                    "json_parse_ms": json_parse,
                    "tool_exec_ms": tool_exec,
                    "disk_io_ms": disk_io,
                    "network_io_ms": network_io,
                    "compute_ms": compute,
                    # Percentages relative to tool_exec only
                    "disk_io_pct": (disk_io / tool_exec) * 100 if tool_exec > 0 else 0,
                    "network_io_pct": (network_io / tool_exec) * 100 if tool_exec > 0 else 0,
                    "compute_pct": (compute / tool_exec) * 100 if tool_exec > 0 else 0,
                }

    return decomposition


def analyze_scaling(results: List[Dict], tool_name: str) -> Dict[str, Dict]:
    """Analyze how execution time scales with input size"""
    scaling = {}

    for r in results:
        if r.get("tool_name") != tool_name:
            continue

        size_label = r.get("input_size_label", "unknown")
        input_size = r.get("input_size", 0)
        node = r.get("node", "unknown")

        key = f"{node}:{size_label}"

        total_ms = get_total_ms(r)
        if total_ms > 0:
            scaling[key] = {
                "node": node,
                "size_label": size_label,
                "input_size_bytes": input_size,
                "total_ms": total_ms,
                "throughput_mbps": (input_size / 1048576) / (total_ms / 1000) if total_ms > 0 else 0,
            }

    return scaling


# =============================================================================
# Output Functions
# =============================================================================

def print_summary_table(summary: Dict[str, Dict], title: str):
    """Print a formatted summary table"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

    if not summary:
        print("No data available")
        return

    # Get column widths
    headers = list(next(iter(summary.values())).keys())
    col_width = max(len(str(k)) for k in summary.keys()) + 2
    val_widths = {h: max(len(h), 10) for h in headers}

    # Print header
    print(f"{'Name':<{col_width}}", end="")
    for h in headers:
        print(f"{h:>{val_widths[h]}}", end="  ")
    print()

    print("-" * (col_width + sum(val_widths.values()) + 2 * len(headers)))

    # Print rows
    for name, values in sorted(summary.items()):
        print(f"{name:<{col_width}}", end="")
        for h in headers:
            val = values.get(h, "N/A")
            if isinstance(val, float):
                print(f"{val:>{val_widths[h]}.2f}", end="  ")
            else:
                print(f"{str(val):>{val_widths[h]}}", end="  ")
        print()


def print_comparison_table(comparison: Dict[str, Dict]):
    """Print cold vs warm comparison table"""
    print(f"\n{'='*70}")
    print("Cold Start vs Warm Start Comparison")
    print(f"{'='*70}")

    if not comparison:
        print("No comparison data available (need both cold and warm measurements)")
        return

    print(f"{'Tool:Size':<30} {'Cold (ms)':>12} {'Warm (ms)':>12} {'Speedup':>10} {'Overhead (ms)':>14}")
    print("-" * 70)

    for key, values in sorted(comparison.items()):
        print(f"{key:<30} {values['cold_mean_ms']:>12.2f} {values['warm_mean_ms']:>12.2f} "
              f"{values['speedup']:>10.2f}x {values['cold_overhead_ms']:>14.2f}")


def print_decomposition_table(decomposition: Dict[str, Dict]):
    """Print time decomposition table"""
    print(f"\n{'='*130}")
    print("Execution Time Decomposition")
    print("  total = startup + json_parse + tool_exec")
    print("  tool_exec = disk_io + network_io + compute")
    print(f"{'='*130}")

    if not decomposition:
        print("No decomposition data available (internal timing not captured)")
        return

    print(f"{'Tool:Size':<28} {'Total':>8} {'Startup':>9} {'JSONParse':>10} {'ToolExec':>9} {'DiskIO':>12} {'NetIO':>12} {'Compute':>12}")
    print(f"{'':<28} {'(ms)':>8} {'(ms)':>9} {'(ms)':>10} {'(ms)':>9} {'(ms/%)':>12} {'(ms/%)':>12} {'(ms/%)':>12}")
    print("-" * 130)

    for key, values in sorted(decomposition.items()):
        print(f"{key:<28} {values['total_ms']:>8.1f} "
              f"{values['startup_ms']:>9.1f} "
              f"{values['json_parse_ms']:>10.1f} "
              f"{values['tool_exec_ms']:>9.1f} "
              f"{values['disk_io_ms']:>6.1f}/{values['disk_io_pct']:>4.0f}% "
              f"{values['network_io_ms']:>6.1f}/{values['network_io_pct']:>4.0f}% "
              f"{values['compute_ms']:>6.1f}/{values['compute_pct']:>4.0f}%")


def export_to_csv(results: List[Dict], output_path: Path):
    """Export results to CSV format"""
    if not HAS_PANDAS:
        print("pandas not installed, exporting as JSON instead")
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(results, f, indent=2)
        return

    # Flatten nested timing data
    flat_results = []
    for r in results:
        flat = {
            "tool_name": r.get("tool_name"),
            "node": r.get("node"),
            "mode": r.get("mode"),
            "input_size": r.get("input_size"),
            "input_size_label": r.get("input_size_label"),
            "output_size": r.get("output_size"),
            "runs": r.get("runs"),
            "timestamp": r.get("timestamp"),
        }

        timing = get_timing(r)
        if timing:
            for k, v in timing.items():
                flat[f"timing_{k}"] = v

        if "timing_std" in r:
            for k, v in r["timing_std"].items():
                flat[f"timing_std_{k}"] = v

        flat_results.append(flat)

    df = pd.DataFrame(flat_results)
    df.to_csv(output_path, index=False)
    print(f"Results exported to: {output_path}")


def plot_node_comparison(results: List[Dict], tool_name: str, output_path: Optional[Path] = None):
    """Plot node comparison for a specific tool"""
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed, skipping plot generation")
        return

    # Group by node and size
    data = defaultdict(lambda: defaultdict(list))

    for r in results:
        if r.get("tool_name") != tool_name:
            continue
        node = r.get("node", "unknown")
        size = r.get("input_size_label", "unknown")
        total_ms = get_total_ms(r)
        if total_ms > 0:
            data[node][size].append(total_ms)

    if not data:
        print(f"No data for tool: {tool_name}")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    sizes = sorted(set(s for node_data in data.values() for s in node_data.keys()))
    x = range(len(sizes))
    width = 0.8 / len(data)

    for i, (node, node_data) in enumerate(sorted(data.items())):
        means = [statistics.mean(node_data.get(s, [0])) for s in sizes]
        offset = (i - len(data)/2 + 0.5) * width
        color = COLORS.get(node, "#95a5a6")
        ax.bar([xi + offset for xi in x], means, width, label=node, color=color)

    ax.set_xlabel("Input Size")
    ax.set_ylabel("Execution Time (ms)")
    ax.set_title(f"Node Comparison: {tool_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


def plot_time_decomposition(results: List[Dict], tool_name: str, output_path: Optional[Path] = None):
    """Plot stacked bar chart of time decomposition"""
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed, skipping plot generation")
        return

    # Filter results for the tool
    filtered = [r for r in results if r.get("tool_name") == tool_name]

    if not filtered:
        print(f"No data for tool: {tool_name}")
        return

    # Prepare data
    labels = []
    startup_times = []
    json_parse_times = []
    disk_io_times = []
    network_io_times = []
    compute_times = []

    for r in filtered:
        label = f"{r.get('node', '')}:{r.get('input_size_label', '')}"
        labels.append(label)

        # timing_ms 또는 timing 키 지원
        timing = r.get("timing_ms") or r.get("timing", {})
        total = timing.get("total") or timing.get("total_ms", 1)  # Avoid division by zero

        # Get component times
        startup = timing.get("startup") or timing.get("startup_ms", 0)
        json_parse = timing.get("json_parse") or timing.get("json_parse_ms", 0)
        disk_io = timing.get("disk_io") or timing.get("disk_io_ms", 0)
        network_io = timing.get("network_io") or timing.get("network_io_ms", 0)
        compute = timing.get("compute") or timing.get("compute_ms", 0)

        # If no internal timing, attribute all to "compute" (unknown)
        if startup + json_parse + disk_io + network_io + compute == 0:
            compute = total

        startup_times.append(startup)
        json_parse_times.append(json_parse)
        disk_io_times.append(disk_io)
        network_io_times.append(network_io)
        compute_times.append(compute)

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(labels))
    # Stack: startup -> json_parse -> disk_io -> network_io -> compute
    ax.bar(x, startup_times, label='Startup (WASM load)', color='#e74c3c')
    bottom1 = startup_times
    ax.bar(x, json_parse_times, bottom=bottom1, label='JSON Parse', color='#f39c12')
    bottom2 = [a+b for a,b in zip(bottom1, json_parse_times)]
    ax.bar(x, disk_io_times, bottom=bottom2, label='Disk I/O', color='#3498db')
    bottom3 = [a+b for a,b in zip(bottom2, disk_io_times)]
    ax.bar(x, network_io_times, bottom=bottom3, label='Network I/O', color='#9b59b6')
    bottom4 = [a+b for a,b in zip(bottom3, network_io_times)]
    ax.bar(x, compute_times, bottom=bottom4, label='Compute', color='#2ecc71')

    ax.set_xlabel("Node:Size")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Time Decomposition: {tool_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze WASM MCP profiling results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show all summaries
  python analyze_results.py

  # Analyze specific node
  python analyze_results.py --node edge-nuc

  # Compare cold vs warm start
  python analyze_results.py --compare

  # Generate plots
  python analyze_results.py --plot --tool read_file

  # Export to CSV
  python analyze_results.py --export results.csv
"""
    )

    parser.add_argument("--node", "-n", type=str,
                        help="Analyze specific node only")
    parser.add_argument("--tool", "-t", type=str,
                        help="Analyze specific tool only")
    parser.add_argument("--compare", "-c", action="store_true",
                        help="Show cold vs warm comparison")
    parser.add_argument("--decomposition", "-d", action="store_true",
                        help="Show time decomposition analysis")
    parser.add_argument("--plot", "-p", action="store_true",
                        help="Generate plots")
    parser.add_argument("--export", "-e", type=Path,
                        help="Export results to CSV")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help="Results directory path")

    args = parser.parse_args()

    # Load results
    if args.node:
        results = load_node_results(args.node, args.results_dir)
        print(f"Loaded {len(results)} measurements for node: {args.node}")
    else:
        results = load_all_results(args.results_dir)
        print(f"Loaded {len(results)} total measurements")

    if not results:
        print("No results found!", file=sys.stderr)
        return

    # Filter by tool if specified
    if args.tool:
        results = [r for r in results if r.get("tool_name") == args.tool]
        print(f"Filtered to {len(results)} measurements for tool: {args.tool}")

    # Print summaries
    tool_summary = summarize_by_tool(results)
    print_summary_table(tool_summary, "Summary by Tool")

    node_summary = summarize_by_node(results)
    print_summary_table(node_summary, "Summary by Node")

    # Cold vs Warm comparison
    if args.compare:
        comparison = compare_cold_warm(results)
        print_comparison_table(comparison)

    # Time decomposition
    if args.decomposition:
        decomposition = analyze_time_decomposition(results)
        print_decomposition_table(decomposition)

    # Generate plots
    if args.plot:
        tool_name = args.tool or "read_file"
        plot_dir = args.results_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        plot_node_comparison(results, tool_name, plot_dir / f"node_comparison_{tool_name}.png")
        plot_time_decomposition(results, tool_name, plot_dir / f"decomposition_{tool_name}.png")

    # Export to CSV
    if args.export:
        export_to_csv(results, args.export)


if __name__ == "__main__":
    main()
