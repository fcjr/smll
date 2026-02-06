#!/usr/bin/env python3
"""Visualize smll benchmark results for blog post."""

import argparse
import json
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")


def load_results(filepath: str) -> dict:
    """Load benchmark results from JSON file."""
    with open(filepath) as f:
        return json.load(f)


def plot_content_types(results: dict, output_dir: Path):
    """Plot compression ratios by content type."""
    if not HAS_MATPLOTLIB:
        return

    data = results.get('content_types', {}).get('results', {})
    if not data:
        print("No content_types data found")
        return

    content_types = list(data.keys())
    smll_ratios = []
    gzip_ratios = []
    zstd_ratios = []

    for ct in content_types:
        smll_ratios.append(data[ct].get('smll', {}).get('ratio', 0) or 0)
        gzip_ratios.append(data[ct].get('gzip', {}).get('ratio', 0) or 0)
        zstd_ratios.append(data[ct].get('zstd', {}).get('ratio', 0) or 0)

    x = np.arange(len(content_types))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width, smll_ratios, width, label='SMLL (LLM)', color='#2ecc71')
    bars2 = ax.bar(x, gzip_ratios, width, label='GZIP', color='#3498db')
    bars3 = ax.bar(x + width, zstd_ratios, width, label='ZSTD', color='#9b59b6')

    ax.set_xlabel('Content Type')
    ax.set_ylabel('Compression Ratio (higher is better)')
    ax.set_title('Compression Ratio by Content Type')
    ax.set_xticks(x)
    ax.set_xticklabels(content_types, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}x',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'content_types.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'content_types.png'}")


def plot_text_length(results: dict, output_dir: Path):
    """Plot compression ratio vs text length."""
    if not HAS_MATPLOTLIB:
        return

    data = results.get('text_length', {}).get('results', {})
    if not data:
        print("No text_length data found")
        return

    lengths = sorted([int(k) for k in data.keys()])

    smll_bpc = []
    gzip_bpc = []
    zstd_bpc = []

    for length in lengths:
        d = data[str(length)]
        smll_bpc.append(d.get('smll', {}).get('bits_per_char', 0) or 0)
        gzip_bpc.append(d.get('gzip', {}).get('bits_per_char', 0) or 0)
        zstd_bpc.append(d.get('zstd', {}).get('bits_per_char', 0) or 0)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(lengths, smll_bpc, 'o-', label='SMLL (LLM)', color='#2ecc71', linewidth=2, markersize=8)
    ax.plot(lengths, gzip_bpc, 's-', label='GZIP', color='#3498db', linewidth=2, markersize=8)
    ax.plot(lengths, zstd_bpc, '^-', label='ZSTD', color='#9b59b6', linewidth=2, markersize=8)

    ax.set_xlabel('Text Length (characters)')
    ax.set_ylabel('Bits per Character (lower is better)')
    ax.set_title('Compression Efficiency vs Text Length')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'text_length.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'text_length.png'}")


def plot_model_sizes(results: dict, output_dir: Path):
    """Plot compression ratio by model size."""
    if not HAS_MATPLOTLIB:
        return

    data = results.get('model_sizes', {}).get('results', {})
    if not data:
        print("No model_sizes data found")
        return

    models = []
    ratios = []
    times = []

    for model, info in data.items():
        if 'error' not in info:
            models.append(model)
            ratios.append(info.get('ratio', 0))
            times.append(info.get('compress_time', 0))

    if not models:
        print("No valid model data")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Compression ratio
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
    bars1 = ax1.bar(models, ratios, color=colors)
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Compression Ratio')
    ax1.set_title('Compression Ratio by Model')
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    for bar, ratio in zip(bars1, ratios):
        ax1.annotate(f'{ratio:.2f}x',
                    xy=(bar.get_x() + bar.get_width() / 2, ratio),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    # Compression time
    bars2 = ax2.bar(models, times, color=colors)
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Compression Time (seconds)')
    ax2.set_title('Compression Time by Model')
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)

    for bar, t in zip(bars2, times):
        ax2.annotate(f'{t:.1f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, t),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_dir / 'model_sizes.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'model_sizes.png'}")


def plot_speed_comparison(results: dict, output_dir: Path):
    """Plot speed comparison."""
    if not HAS_MATPLOTLIB:
        return

    data = results.get('speed', {}).get('results', {})
    if not data:
        print("No speed data found")
        return

    sizes = sorted([int(k) for k in data.keys()])

    compress_throughput = []
    decompress_throughput = []

    for size in sizes:
        d = data[str(size)]
        compress_throughput.append(d.get('throughput_compress', 0))
        decompress_throughput.append(d.get('throughput_decompress', 0))

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(sizes))
    width = 0.35

    bars1 = ax.bar(x - width/2, compress_throughput, width, label='Compress', color='#2ecc71')
    bars2 = ax.bar(x + width/2, decompress_throughput, width, label='Decompress', color='#e74c3c')

    ax.set_xlabel('Text Size (characters)')
    ax.set_ylabel('Throughput (characters/second)')
    ax.set_title('SMLL Compression/Decompression Speed')
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'speed.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'speed.png'}")


def plot_head_to_head(results: dict, output_dir: Path):
    """Plot head-to-head comparison."""
    if not HAS_MATPLOTLIB:
        return

    data = results.get('head_to_head', {}).get('results', {})
    if not data:
        print("No head_to_head data found")
        return

    content_types = list(data.keys())
    smll_sizes = []
    gzip_sizes = []
    zstd_sizes = []
    winners = []

    for ct in content_types:
        smll_sizes.append(data[ct].get('smll') or 0)
        gzip_sizes.append(data[ct].get('gzip', 0))
        zstd_sizes.append(data[ct].get('zstd', 0))
        winners.append(data[ct].get('winner', ''))

    x = np.arange(len(content_types))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width, smll_sizes, width, label='SMLL', color='#2ecc71')
    bars2 = ax.bar(x, gzip_sizes, width, label='GZIP', color='#3498db')
    bars3 = ax.bar(x + width, zstd_sizes, width, label='ZSTD', color='#9b59b6')

    ax.set_xlabel('Content Type')
    ax.set_ylabel('Compressed Size (bytes, lower is better)')
    ax.set_title('Head-to-Head Compression Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(content_types, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'head_to_head.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'head_to_head.png'}")


def plot_impossible_compression(results: dict, output_dir: Path):
    """Plot bits per character for famous quotes."""
    if not HAS_MATPLOTLIB:
        return

    data = results.get('impossible', {}).get('results', {})
    if not data:
        print("No impossible data found")
        return

    quotes = list(data.keys())
    bpc = [data[q].get('bits_per_char', 0) for q in quotes]
    ratios = [data[q].get('ratio', 0) for q in quotes]

    # Truncate quote labels
    labels = [q[:20] + '...' if len(q) > 20 else q for q in quotes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bits per character
    colors = plt.cm.RdYlGn_r(np.array(bpc) / max(bpc) if max(bpc) > 0 else np.zeros(len(bpc)))
    bars1 = ax1.barh(labels, bpc, color=colors)
    ax1.set_xlabel('Bits per Character (lower = more predictable)')
    ax1.set_title('LLM Compression Efficiency on Famous Quotes')
    ax1.grid(axis='x', alpha=0.3)

    # Add 8 bpc reference line (uncompressed UTF-8)
    ax1.axvline(x=8, color='red', linestyle='--', label='Uncompressed (8 bpc)')
    ax1.legend()

    # Compression ratio
    colors2 = plt.cm.RdYlGn(np.array(ratios) / max(ratios) if max(ratios) > 0 else np.zeros(len(ratios)))
    bars2 = ax2.barh(labels, ratios, color=colors2)
    ax2.set_xlabel('Compression Ratio (higher is better)')
    ax2.set_title('Compression Ratio on Famous Quotes')
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'impossible_compression.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'impossible_compression.png'}")


def create_summary_table(results: dict, output_dir: Path):
    """Create a markdown summary table."""
    content_data = results.get('content_types', {}).get('results', {})

    lines = [
        "# SMLL Benchmark Results Summary",
        "",
        "## Compression Ratio by Content Type",
        "",
        "| Content Type | Original | SMLL | GZIP | ZSTD | Winner |",
        "|--------------|----------|------|------|------|--------|",
    ]

    for ct, data in content_data.items():
        orig = data.get('original_size', 0)
        smll_ratio = data.get('smll', {}).get('ratio', 0) or 0
        gzip_ratio = data.get('gzip', {}).get('ratio', 0) or 0
        zstd_ratio = data.get('zstd', {}).get('ratio', 0) or 0

        all_ratios = {'SMLL': smll_ratio, 'GZIP': gzip_ratio, 'ZSTD': zstd_ratio}
        winner = max(all_ratios, key=all_ratios.get) if any(all_ratios.values()) else '-'

        lines.append(f"| {ct} | {orig} | {smll_ratio:.2f}x | {gzip_ratio:.2f}x | {zstd_ratio:.2f}x | {winner} |")

    lines.extend([
        "",
        "## Key Findings",
        "",
        "- **LLM-generated text**: SMLL excels at compressing text similar to what LLMs produce",
        "- **Code**: High compression due to predictable syntax patterns",
        "- **Random/High-entropy**: Traditional compressors perform better",
        "- **Speed**: SMLL is slower but achieves competitive or better ratios on natural language",
        "",
    ])

    output_path = output_dir / 'summary.md'
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize smll benchmark results")
    parser.add_argument("results_file", help="JSON file with benchmark results")
    parser.add_argument("--output-dir", "-o", default="benchmark_charts",
                        help="Output directory for charts")

    args = parser.parse_args()

    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required. Install with: pip install matplotlib")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Loading results from: {args.results_file}")
    results = load_results(args.results_file)

    print(f"Generating charts in: {output_dir}")

    plot_content_types(results, output_dir)
    plot_text_length(results, output_dir)
    plot_model_sizes(results, output_dir)
    plot_speed_comparison(results, output_dir)
    plot_head_to_head(results, output_dir)
    plot_impossible_compression(results, output_dir)
    create_summary_table(results, output_dir)

    print("\nDone! Charts saved to:", output_dir)


if __name__ == "__main__":
    main()
