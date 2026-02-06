#!/usr/bin/env python3
"""Comprehensive benchmarks for smll LLM-based compression library."""

import argparse
import csv
import json
import os
import random
import string
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import bz2
import lzma
import zlib

try:
    import zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

import smll


# =============================================================================
# Sample Data Generation
# =============================================================================

SAMPLE_PYTHON_CODE = '''
def quicksort(arr):
    """Sorts an array using the quicksort algorithm."""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)


class BinarySearchTree:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

    def insert(self, value):
        if value < self.value:
            if self.left is None:
                self.left = BinarySearchTree(value)
            else:
                self.left.insert(value)
        else:
            if self.right is None:
                self.right = BinarySearchTree(value)
            else:
                self.right.insert(value)

    def search(self, value):
        if value == self.value:
            return True
        elif value < self.value and self.left:
            return self.left.search(value)
        elif value > self.value and self.right:
            return self.right.search(value)
        return False
'''

SAMPLE_JAVASCRIPT_CODE = '''
class EventEmitter {
  constructor() {
    this.events = {};
  }

  on(event, listener) {
    if (!this.events[event]) {
      this.events[event] = [];
    }
    this.events[event].push(listener);
    return this;
  }

  emit(event, ...args) {
    if (!this.events[event]) return false;
    this.events[event].forEach(listener => listener.apply(this, args));
    return true;
  }

  removeListener(event, listenerToRemove) {
    if (!this.events[event]) return this;
    this.events[event] = this.events[event].filter(
      listener => listener !== listenerToRemove
    );
    return this;
  }
}

async function fetchWithRetry(url, options = {}, retries = 3) {
  for (let i = 0; i < retries; i++) {
    try {
      const response = await fetch(url, options);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      if (i === retries - 1) throw error;
      await new Promise(r => setTimeout(r, 1000 * Math.pow(2, i)));
    }
  }
}
'''

SAMPLE_C_CODE = '''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Node {
    int data;
    struct Node* next;
} Node;

Node* createNode(int data) {
    Node* newNode = (Node*)malloc(sizeof(Node));
    if (newNode == NULL) {
        fprintf(stderr, "Memory allocation failed\\n");
        exit(1);
    }
    newNode->data = data;
    newNode->next = NULL;
    return newNode;
}

void insertAtHead(Node** head, int data) {
    Node* newNode = createNode(data);
    newNode->next = *head;
    *head = newNode;
}

void printList(Node* head) {
    Node* current = head;
    while (current != NULL) {
        printf("%d -> ", current->data);
        current = current->next;
    }
    printf("NULL\\n");
}

void freeList(Node* head) {
    Node* current = head;
    while (current != NULL) {
        Node* next = current->next;
        free(current);
        current = next;
    }
}
'''

SAMPLE_PROSE = '''
The old lighthouse stood sentinel on the rocky promontory, its weathered stones bearing
witness to countless storms that had battered the coastline over the centuries. Captain
Margaret Chen had seen it from her ship many times, but today was different. Today, she
would finally set foot on the island that had haunted her dreams since childhood.

The small rowboat pitched and rolled as her first mate navigated through the treacherous
waters. Jagged rocks lurked just beneath the surface, their dark shapes visible through
the crystal-clear water. One wrong move and they would be dashed against the ancient
stones that had claimed so many vessels before.

"Steady now," Margaret called out, her voice carrying over the crash of waves. "We're
almost through the worst of it."

The lighthouse keeper had stopped answering radio calls three days ago. The coast guard
had sent a team, but they never returned. Now it was up to Margaret and her crew to
discover what had happened on Thornwood Island, and whether the old legends about the
lighthouse were more than mere superstition.
'''

SAMPLE_WIKIPEDIA = '''
Quantum entanglement is a phenomenon that occurs when a group of particles are generated,
interact, or share spatial proximity in a way such that the quantum state of each particle
of the group cannot be described independently of the state of the others, including when
the particles are separated by a large distance.

The topic of quantum entanglement is at the heart of the disparity between classical and
quantum physics: entanglement is a primary feature of quantum mechanics not present in
classical mechanics. Measurements of physical properties such as position, momentum, spin,
and polarization performed on entangled particles can, in some cases, be found to be
perfectly correlated.

Albert Einstein famously derided entanglement as "spooky action at a distance." The
phenomenon challenged his belief that information could not travel faster than light.
However, modern understanding clarifies that while entangled particles exhibit correlated
measurements, no information is actually transmitted between them instantaneously.

Applications of quantum entanglement include quantum computing, quantum cryptography,
and quantum teleportation. These technologies leverage the unique properties of entangled
particles to perform computations and secure communications in ways impossible with
classical systems.
'''

SAMPLE_JSON = '''
{
  "users": [
    {
      "id": 1,
      "name": "Alice Johnson",
      "email": "alice@example.com",
      "roles": ["admin", "editor"],
      "settings": {
        "theme": "dark",
        "notifications": true,
        "language": "en-US"
      }
    },
    {
      "id": 2,
      "name": "Bob Smith",
      "email": "bob@example.com",
      "roles": ["viewer"],
      "settings": {
        "theme": "light",
        "notifications": false,
        "language": "en-GB"
      }
    },
    {
      "id": 3,
      "name": "Carol Williams",
      "email": "carol@example.com",
      "roles": ["editor", "moderator"],
      "settings": {
        "theme": "auto",
        "notifications": true,
        "language": "es-ES"
      }
    }
  ],
  "metadata": {
    "total": 3,
    "page": 1,
    "per_page": 10,
    "generated_at": "2024-01-15T10:30:00Z"
  }
}
'''

SAMPLE_XML = '''
<?xml version="1.0" encoding="UTF-8"?>
<catalog>
  <book id="bk101">
    <author>Gambardella, Matthew</author>
    <title>XML Developer's Guide</title>
    <genre>Computer</genre>
    <price>44.95</price>
    <publish_date>2000-10-01</publish_date>
    <description>An in-depth look at creating applications with XML.</description>
  </book>
  <book id="bk102">
    <author>Ralls, Kim</author>
    <title>Midnight Rain</title>
    <genre>Fantasy</genre>
    <price>5.95</price>
    <publish_date>2000-12-16</publish_date>
    <description>A former architect battles corporate zombies.</description>
  </book>
  <book id="bk103">
    <author>Corets, Eva</author>
    <title>Maeve Ascendant</title>
    <genre>Fantasy</genre>
    <price>5.95</price>
    <publish_date>2000-11-17</publish_date>
    <description>After the collapse of a nanotechnology society.</description>
  </book>
</catalog>
'''

SAMPLE_CSV = '''id,name,department,salary,hire_date,is_active
1,John Smith,Engineering,85000,2020-03-15,true
2,Jane Doe,Marketing,72000,2019-07-22,true
3,Bob Johnson,Engineering,92000,2018-01-10,true
4,Alice Brown,Sales,68000,2021-05-03,true
5,Charlie Wilson,Engineering,78000,2020-11-18,false
6,Diana Martinez,Marketing,75000,2019-09-30,true
7,Edward Lee,Sales,71000,2022-02-14,true
8,Fiona Garcia,Engineering,95000,2017-08-25,true
9,George Taylor,Marketing,69000,2021-04-12,true
10,Hannah Anderson,Sales,73000,2020-06-28,true
'''

# LLM-generated text (typical ChatGPT-style output)
SAMPLE_LLM_GENERATED = '''
I'd be happy to help you understand the concept of machine learning! Let me break it down
into simple terms.

Machine learning is a subset of artificial intelligence that enables computers to learn
and improve from experience without being explicitly programmed. Here are the key points:

1. **Data-Driven Learning**: Instead of following rigid rules, ML algorithms learn patterns
   from data. The more quality data they're exposed to, the better they perform.

2. **Types of Learning**:
   - Supervised Learning: The algorithm learns from labeled examples
   - Unsupervised Learning: The algorithm finds patterns in unlabeled data
   - Reinforcement Learning: The algorithm learns through trial and error

3. **Common Applications**:
   - Image recognition
   - Natural language processing
   - Recommendation systems
   - Fraud detection

The key insight is that machine learning automates the process of finding patterns. Rather
than a programmer writing specific rules, the algorithm discovers these rules from the data
itself. This makes ML particularly powerful for complex problems where manual rule-writing
would be impractical.

Would you like me to elaborate on any of these concepts?
'''

FAMOUS_QUOTES = [
    "To be, or not to be, that is the question.",
    "It was the best of times, it was the worst of times.",
    "All happy families are alike; each unhappy family is unhappy in its own way.",
    "Call me Ishmael.",
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
    "In the beginning God created the heaven and the earth.",
    "The only thing we have to fear is fear itself.",
    "I have a dream that one day this nation will rise up.",
]


def generate_random_text(length: int) -> str:
    """Generate random ASCII text."""
    return ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=length))


def generate_uuids(count: int) -> str:
    """Generate random UUIDs."""
    return '\n'.join(str(uuid.uuid4()) for _ in range(count))


def generate_base64(length: int) -> str:
    """Generate random base64-like text."""
    import base64
    random_bytes = bytes(random.getrandbits(8) for _ in range(length))
    return base64.b64encode(random_bytes).decode('ascii')


def generate_repetitive_text(pattern: str, count: int) -> str:
    """Generate highly repetitive text."""
    return (pattern + ' ') * count


# =============================================================================
# Benchmark Infrastructure
# =============================================================================

@dataclass
class CompressionResult:
    """Result of a compression operation."""
    original_size: int
    compressed_size: int
    compression_ratio: float
    compress_time: float
    decompress_time: Optional[float] = None
    verified: bool = False
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    name: str
    results: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


def compress_with_smll(compressor: smll.Compressor, text: str, verify: bool = True) -> CompressionResult:
    """Compress text using smll and optionally verify decompression."""
    original_bytes = text.encode('utf-8')
    original_size = len(original_bytes)

    try:
        start = time.perf_counter()
        compressed = compressor.compress(text)
        compress_time = time.perf_counter() - start

        compressed_size = len(compressed)

        decompress_time = None
        verified = False

        if verify:
            start = time.perf_counter()
            decompressed = compressor.decompress(compressed)
            decompress_time = time.perf_counter() - start
            verified = (decompressed == text)

        return CompressionResult(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=original_size / compressed_size if compressed_size > 0 else 0,
            compress_time=compress_time,
            decompress_time=decompress_time,
            verified=verified,
        )
    except Exception as e:
        return CompressionResult(
            original_size=original_size,
            compressed_size=0,
            compression_ratio=0,
            compress_time=0,
            error=str(e),
        )


def compress_with_traditional(text: str) -> dict:
    """Compress text using traditional algorithms."""
    original_bytes = text.encode('utf-8')
    original_size = len(original_bytes)

    results = {}

    # GZIP
    start = time.perf_counter()
    gzip_compressed = zlib.compress(original_bytes, level=9)
    gzip_time = time.perf_counter() - start
    results['gzip'] = CompressionResult(
        original_size=original_size,
        compressed_size=len(gzip_compressed),
        compression_ratio=original_size / len(gzip_compressed),
        compress_time=gzip_time,
        verified=True,
    )

    # BZ2
    start = time.perf_counter()
    bz2_compressed = bz2.compress(original_bytes, compresslevel=9)
    bz2_time = time.perf_counter() - start
    results['bz2'] = CompressionResult(
        original_size=original_size,
        compressed_size=len(bz2_compressed),
        compression_ratio=original_size / len(bz2_compressed),
        compress_time=bz2_time,
        verified=True,
    )

    # LZMA
    start = time.perf_counter()
    lzma_compressed = lzma.compress(original_bytes, preset=9)
    lzma_time = time.perf_counter() - start
    results['lzma'] = CompressionResult(
        original_size=original_size,
        compressed_size=len(lzma_compressed),
        compression_ratio=original_size / len(lzma_compressed),
        compress_time=lzma_time,
        verified=True,
    )

    # ZSTD
    if HAS_ZSTD:
        start = time.perf_counter()
        zstd_compressed = zstd.compress(original_bytes, 22)
        zstd_time = time.perf_counter() - start
        results['zstd'] = CompressionResult(
            original_size=original_size,
            compressed_size=len(zstd_compressed),
            compression_ratio=original_size / len(zstd_compressed),
            compress_time=zstd_time,
            verified=True,
        )

    return results


# =============================================================================
# Benchmark 1: Compression Ratio vs Content Type
# =============================================================================

def benchmark_content_types(compressor: smll.Compressor, verify: bool = False) -> BenchmarkResult:
    """Benchmark compression across different content types."""
    print("\n" + "=" * 70)
    print("BENCHMARK 1: Compression Ratio vs Content Type")
    print("=" * 70)

    content_types = {
        'Python Code': SAMPLE_PYTHON_CODE,
        'JavaScript Code': SAMPLE_JAVASCRIPT_CODE,
        'C Code': SAMPLE_C_CODE,
        'Natural Prose': SAMPLE_PROSE,
        'Wikipedia': SAMPLE_WIKIPEDIA,
        'JSON': SAMPLE_JSON,
        'XML': SAMPLE_XML,
        'CSV': SAMPLE_CSV,
        'LLM-Generated': SAMPLE_LLM_GENERATED,
        'Random Text': generate_random_text(500),
        'UUIDs': generate_uuids(20),
        'Base64': generate_base64(300),
        'Repetitive': generate_repetitive_text("hello world", 50),
    }

    results = {}

    for name, content in content_types.items():
        print(f"\n  Testing: {name} ({len(content)} chars)")

        # LLM compression
        smll_result = compress_with_smll(compressor, content, verify=verify)

        # Traditional compression
        trad_results = compress_with_traditional(content)

        results[name] = {
            'original_size': smll_result.original_size,
            'smll': {
                'size': smll_result.compressed_size,
                'ratio': smll_result.compression_ratio,
                'time': smll_result.compress_time,
                'verified': smll_result.verified,
                'error': smll_result.error,
            },
            **{k: {'size': v.compressed_size, 'ratio': v.compression_ratio, 'time': v.compress_time}
               for k, v in trad_results.items()}
        }

        # Print results
        print(f"    Original:  {smll_result.original_size:>6} bytes")
        if smll_result.error:
            print(f"    SMLL:      ERROR - {smll_result.error}")
        else:
            print(f"    SMLL:      {smll_result.compressed_size:>6} bytes ({smll_result.compression_ratio:.2f}x) [{smll_result.compress_time:.2f}s]")
        for algo, res in trad_results.items():
            print(f"    {algo.upper():10} {res.compressed_size:>6} bytes ({res.compression_ratio:.2f}x) [{res.compress_time:.4f}s]")

    return BenchmarkResult(name="content_types", results=results)


# =============================================================================
# Benchmark 2: Model Size vs Compression Tradeoff
# =============================================================================

# Common models to test (user should have these downloaded)
MODEL_CONFIGS = [
    ("QuantFactory/SmolLM2-135M-Instruct-GGUF", "*Q4_0.gguf", "SmolLM2-135M-Q4"),
    ("QuantFactory/SmolLM2-135M-Instruct-GGUF", "*Q8_0.gguf", "SmolLM2-135M-Q8"),
    ("QuantFactory/SmolLM2-360M-GGUF", "*Q4_0.gguf", "SmolLM2-360M-Q4"),
    ("QuantFactory/SmolLM2-360M-GGUF", "*Q8_0.gguf", "SmolLM2-360M-Q8"),
    ("QuantFactory/SmolLM2-1.7B-Instruct-GGUF", "*Q4_0.gguf", "SmolLM2-1.7B-Q4"),
    ("QuantFactory/SmolLM2-1.7B-Instruct-GGUF", "*Q8_0.gguf", "SmolLM2-1.7B-Q8"),
]


def benchmark_model_sizes(test_text: Optional[str] = None, verify: bool = False) -> BenchmarkResult:
    """Benchmark different model sizes and quantizations."""
    print("\n" + "=" * 70)
    print("BENCHMARK 2: Model Size vs Compression Tradeoff")
    print("=" * 70)

    if test_text is None:
        test_text = SAMPLE_PROSE[:500]

    print(f"\n  Test text: {len(test_text)} characters")

    results = {}

    for repo_id, filename, label in MODEL_CONFIGS:
        print(f"\n  Testing: {label}")
        try:
            with smll.Compressor.from_pretrained(repo_id=repo_id, filename=filename) as compressor:
                result = compress_with_smll(compressor, test_text, verify=verify)

                results[label] = {
                    'original_size': result.original_size,
                    'compressed_size': result.compressed_size,
                    'ratio': result.compression_ratio,
                    'compress_time': result.compress_time,
                    'decompress_time': result.decompress_time,
                    'verified': result.verified,
                }

                print(f"    Size: {result.compressed_size} bytes ({result.compression_ratio:.2f}x)")
                print(f"    Time: {result.compress_time:.2f}s compress", end="")
                if result.decompress_time:
                    print(f", {result.decompress_time:.2f}s decompress")
                else:
                    print()

        except Exception as e:
            print(f"    ERROR: {e}")
            results[label] = {'error': str(e)}

    return BenchmarkResult(name="model_sizes", results=results)


# =============================================================================
# Benchmark 3: Text Length Scaling
# =============================================================================

def benchmark_text_length(compressor: smll.Compressor, verify: bool = False) -> BenchmarkResult:
    """Benchmark compression ratio vs text length."""
    print("\n" + "=" * 70)
    print("BENCHMARK 3: Text Length Scaling")
    print("=" * 70)

    # Use prose as base text, repeat to get longer versions
    base_text = SAMPLE_PROSE + " " + SAMPLE_WIKIPEDIA

    lengths = [50, 100, 200, 500, 1000, 2000]
    results = {}

    for length in lengths:
        # Truncate or repeat to get desired length
        if length <= len(base_text):
            test_text = base_text[:length]
        else:
            repeats = (length // len(base_text)) + 1
            test_text = (base_text * repeats)[:length]

        print(f"\n  Testing: {length} characters")

        # LLM compression
        smll_result = compress_with_smll(compressor, test_text, verify=verify)

        # Traditional compression
        trad_results = compress_with_traditional(test_text)

        results[length] = {
            'original_size': smll_result.original_size,
            'smll': {
                'size': smll_result.compressed_size,
                'ratio': smll_result.compression_ratio,
                'time': smll_result.compress_time,
                'bits_per_char': (smll_result.compressed_size * 8) / length if smll_result.compressed_size > 0 else 0,
            },
            **{k: {
                'size': v.compressed_size,
                'ratio': v.compression_ratio,
                'time': v.compress_time,
                'bits_per_char': (v.compressed_size * 8) / length,
            } for k, v in trad_results.items()}
        }

        if smll_result.error:
            print(f"    SMLL: ERROR - {smll_result.error}")
        else:
            bpc = (smll_result.compressed_size * 8) / length
            print(f"    SMLL: {smll_result.compression_ratio:.2f}x ({bpc:.2f} bits/char)")

        for algo, res in trad_results.items():
            bpc = (res.compressed_size * 8) / length
            print(f"    {algo.upper()}: {res.compression_ratio:.2f}x ({bpc:.2f} bits/char)")

    return BenchmarkResult(name="text_length", results=results)


# =============================================================================
# Benchmark 4: Domain-Specific Models
# =============================================================================

DOMAIN_MODELS = [
    # General models
    ("QuantFactory/SmolLM2-360M-GGUF", "*Q4_0.gguf", "SmolLM2-360M (General)"),
    # Code models (user can add more)
    # ("TheBloke/CodeLlama-7B-GGUF", "*Q4_K_M.gguf", "CodeLlama-7B"),
    # ("TheBloke/starcoder-GGUF", "*Q4_K_M.gguf", "StarCoder"),
]


def benchmark_domain_models(verify: bool = False) -> BenchmarkResult:
    """Benchmark domain-specific models on different content."""
    print("\n" + "=" * 70)
    print("BENCHMARK 4: Domain-Specific Models")
    print("=" * 70)

    test_cases = {
        'Python Code': SAMPLE_PYTHON_CODE[:500],
        'Natural Prose': SAMPLE_PROSE[:500],
        'JSON': SAMPLE_JSON[:500],
    }

    results = {}

    for repo_id, filename, model_label in DOMAIN_MODELS:
        print(f"\n  Model: {model_label}")
        results[model_label] = {}

        try:
            with smll.Compressor.from_pretrained(repo_id=repo_id, filename=filename) as compressor:
                for content_type, content in test_cases.items():
                    result = compress_with_smll(compressor, content, verify=verify)

                    results[model_label][content_type] = {
                        'original_size': result.original_size,
                        'compressed_size': result.compressed_size,
                        'ratio': result.compression_ratio,
                        'time': result.compress_time,
                    }

                    if result.error:
                        print(f"    {content_type}: ERROR - {result.error}")
                    else:
                        print(f"    {content_type}: {result.compression_ratio:.2f}x")

        except Exception as e:
            print(f"    ERROR loading model: {e}")
            results[model_label] = {'error': str(e)}

    return BenchmarkResult(name="domain_models", results=results)


# =============================================================================
# Benchmark 5: Speed Benchmarks
# =============================================================================

def benchmark_speed(compressor: smll.Compressor, iterations: int = 3) -> BenchmarkResult:
    """Benchmark compression/decompression speed."""
    print("\n" + "=" * 70)
    print("BENCHMARK 5: Speed Benchmarks")
    print("=" * 70)

    test_sizes = [100, 500, 1000]
    base_text = SAMPLE_PROSE + " " + SAMPLE_WIKIPEDIA

    results = {}

    for size in test_sizes:
        if size <= len(base_text):
            test_text = base_text[:size]
        else:
            repeats = (size // len(base_text)) + 1
            test_text = (base_text * repeats)[:size]

        print(f"\n  Size: {size} characters ({iterations} iterations)")

        compress_times = []
        decompress_times = []
        compressed_data = None

        for i in range(iterations):
            # Compress
            start = time.perf_counter()
            compressed_data = compressor.compress(test_text)
            compress_times.append(time.perf_counter() - start)

            # Decompress
            start = time.perf_counter()
            _ = compressor.decompress(compressed_data)
            decompress_times.append(time.perf_counter() - start)

        avg_compress = sum(compress_times) / len(compress_times)
        avg_decompress = sum(decompress_times) / len(decompress_times)

        throughput_compress = size / avg_compress if avg_compress > 0 else 0
        throughput_decompress = size / avg_decompress if avg_decompress > 0 else 0

        results[size] = {
            'compress_time_avg': avg_compress,
            'compress_time_min': min(compress_times),
            'compress_time_max': max(compress_times),
            'decompress_time_avg': avg_decompress,
            'decompress_time_min': min(decompress_times),
            'decompress_time_max': max(decompress_times),
            'throughput_compress': throughput_compress,
            'throughput_decompress': throughput_decompress,
        }

        print(f"    Compress:   {avg_compress:.3f}s avg ({throughput_compress:.1f} chars/sec)")
        print(f"    Decompress: {avg_decompress:.3f}s avg ({throughput_decompress:.1f} chars/sec)")

        # Compare to traditional
        trad = compress_with_traditional(test_text)
        print(f"    GZIP:       {trad['gzip'].compress_time:.6f}s ({size / trad['gzip'].compress_time:.0f} chars/sec)")
        if HAS_ZSTD:
            print(f"    ZSTD:       {trad['zstd'].compress_time:.6f}s ({size / trad['zstd'].compress_time:.0f} chars/sec)")

    return BenchmarkResult(name="speed", results=results)


# =============================================================================
# Benchmark 6: Head-to-Head Comparison
# =============================================================================

def benchmark_head_to_head(compressor: smll.Compressor, verify: bool = False) -> BenchmarkResult:
    """Head-to-head comparison with edge cases."""
    print("\n" + "=" * 70)
    print("BENCHMARK 6: Head-to-Head Comparison")
    print("=" * 70)

    test_cases = {
        'Very Short (50 chars)': SAMPLE_PROSE[:50],
        'Short (100 chars)': SAMPLE_PROSE[:100],
        'Medium (500 chars)': SAMPLE_PROSE[:500],
        'Highly Repetitive': generate_repetitive_text("the quick brown fox ", 30),
        'Mixed Code+Comments': SAMPLE_PYTHON_CODE[:400],
        'Structured Data': SAMPLE_JSON[:400],
        'High Entropy': generate_random_text(200),
    }

    results = {}

    print(f"\n  {'Content Type':<25} {'Original':>8} {'SMLL':>10} {'GZIP':>10} {'ZSTD':>10} {'Winner':>10}")
    print("  " + "-" * 75)

    for name, content in test_cases.items():
        smll_result = compress_with_smll(compressor, content, verify=verify)
        trad_results = compress_with_traditional(content)

        # Determine winner
        all_results = {'smll': smll_result.compressed_size if not smll_result.error else float('inf')}
        all_results.update({k: v.compressed_size for k, v in trad_results.items()})

        winner = min(all_results, key=all_results.get)

        results[name] = {
            'original': smll_result.original_size,
            'smll': smll_result.compressed_size if not smll_result.error else None,
            'smll_ratio': smll_result.compression_ratio if not smll_result.error else None,
            **{k: v.compressed_size for k, v in trad_results.items()},
            'winner': winner,
        }

        smll_str = f"{smll_result.compressed_size}" if not smll_result.error else "ERR"
        gzip_size = trad_results['gzip'].compressed_size
        zstd_size = trad_results['zstd'].compressed_size if HAS_ZSTD else "-"

        print(f"  {name:<25} {smll_result.original_size:>8} {smll_str:>10} {gzip_size:>10} {zstd_size:>10} {winner.upper():>10}")

    return BenchmarkResult(name="head_to_head", results=results)


# =============================================================================
# Benchmark 7: "Impossible" Compression Demo
# =============================================================================

def benchmark_impossible_compression(compressor: smll.Compressor) -> BenchmarkResult:
    """Demonstrate extreme compression on well-known text."""
    print("\n" + "=" * 70)
    print("BENCHMARK 7: 'Impossible' Compression Demo")
    print("=" * 70)

    results = {}

    print("\n  Famous quotes and their compression:")
    print(f"  {'Quote (truncated)':<50} {'Orig':>6} {'SMLL':>6} {'Ratio':>7} {'BPC':>6}")
    print("  " + "-" * 80)

    for quote in FAMOUS_QUOTES:
        result = compress_with_smll(compressor, quote, verify=False)

        if not result.error:
            bpc = (result.compressed_size * 8) / len(quote)
            truncated = quote[:47] + "..." if len(quote) > 50 else quote

            results[quote[:30]] = {
                'original': result.original_size,
                'compressed': result.compressed_size,
                'ratio': result.compression_ratio,
                'bits_per_char': bpc,
            }

            print(f"  {truncated:<50} {result.original_size:>6} {result.compressed_size:>6} {result.compression_ratio:>6.2f}x {bpc:>5.2f}")

    # Also test some "impossible" cases
    print("\n  Predictable completions:")
    predictable = [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "SELECT * FROM users WHERE id = 1;",
        "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt",
        "The quick brown fox jumps over the lazy dog.",
    ]

    for text in predictable:
        result = compress_with_smll(compressor, text, verify=False)
        if not result.error:
            bpc = (result.compressed_size * 8) / len(text)
            truncated = text[:47].replace('\n', ' ') + "..." if len(text) > 50 else text.replace('\n', ' ')
            print(f"  {truncated:<50} {result.original_size:>6} {result.compressed_size:>6} {result.compression_ratio:>6.2f}x {bpc:>5.2f}")

    return BenchmarkResult(name="impossible_compression", results=results)


# =============================================================================
# Benchmark 8: Reproducibility Test
# =============================================================================

def benchmark_reproducibility(compressor: smll.Compressor, iterations: int = 5) -> BenchmarkResult:
    """Test that compression is deterministic."""
    print("\n" + "=" * 70)
    print("BENCHMARK 8: Reproducibility Test")
    print("=" * 70)

    test_texts = [
        ("Short text", "Hello, world!"),
        ("Medium text", SAMPLE_PROSE[:200]),
        ("Code", SAMPLE_PYTHON_CODE[:200]),
    ]

    results = {}
    all_passed = True

    for name, text in test_texts:
        print(f"\n  Testing: {name}")

        compressed_results = []
        for i in range(iterations):
            compressed = compressor.compress(text)
            compressed_results.append(compressed)

        # Check all results are identical
        is_deterministic = all(c == compressed_results[0] for c in compressed_results)

        results[name] = {
            'iterations': iterations,
            'deterministic': is_deterministic,
            'compressed_size': len(compressed_results[0]),
        }

        if is_deterministic:
            print(f"    PASS: All {iterations} compressions produced identical output ({len(compressed_results[0])} bytes)")
        else:
            print(f"    FAIL: Compression produced different results!")
            all_passed = False
            # Show differences
            sizes = [len(c) for c in compressed_results]
            print(f"    Sizes: {sizes}")

    print(f"\n  Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return BenchmarkResult(name="reproducibility", results=results, metadata={'all_passed': all_passed})


# =============================================================================
# Main
# =============================================================================

def run_all_benchmarks(
    model_repo: str = "QuantFactory/SmolLM2-360M-GGUF",
    model_file: str = "*Q4_0.gguf",
    verify: bool = False,
    output_file: Optional[str] = None,
):
    """Run all benchmarks and optionally save results."""
    print("=" * 70)
    print("SMLL COMPRESSION BENCHMARKS")
    print("=" * 70)
    print(f"\nLoading model: {model_repo} / {model_file}")

    all_results = {}

    with smll.Compressor.from_pretrained(repo_id=model_repo, filename=model_file) as compressor:
        print(f"Model loaded: {compressor.model_path}\n")

        # Run benchmarks
        all_results['content_types'] = benchmark_content_types(compressor, verify=verify)
        all_results['text_length'] = benchmark_text_length(compressor, verify=verify)
        all_results['speed'] = benchmark_speed(compressor)
        all_results['head_to_head'] = benchmark_head_to_head(compressor, verify=verify)
        all_results['impossible'] = benchmark_impossible_compression(compressor)
        all_results['reproducibility'] = benchmark_reproducibility(compressor)

    # Model comparison benchmarks (loads multiple models)
    print("\n" + "=" * 70)
    print("Running model comparison benchmarks (this may take a while)...")
    print("=" * 70)

    try:
        all_results['model_sizes'] = benchmark_model_sizes(verify=verify)
    except Exception as e:
        print(f"Model size benchmark failed: {e}")

    try:
        all_results['domain_models'] = benchmark_domain_models(verify=verify)
    except Exception as e:
        print(f"Domain models benchmark failed: {e}")

    # Save results
    if output_file:
        output_data = {
            name: {'results': r.results, 'metadata': r.metadata}
            for name, r in all_results.items()
        }
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nResults saved to: {output_file}")

    print("\n" + "=" * 70)
    print("BENCHMARKS COMPLETE")
    print("=" * 70)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run smll compression benchmarks")
    parser.add_argument("--model-repo", default="QuantFactory/SmolLM2-360M-GGUF",
                        help="Hugging Face model repository")
    parser.add_argument("--model-file", default="*Q4_0.gguf",
                        help="Model filename pattern")
    parser.add_argument("--verify", action="store_true",
                        help="Verify decompression (slower)")
    parser.add_argument("--output", "-o", type=str,
                        help="Output JSON file for results")
    parser.add_argument("--benchmark", "-b", type=str,
                        choices=['content', 'models', 'length', 'domain', 'speed', 'h2h', 'impossible', 'repro', 'all'],
                        default='all',
                        help="Run specific benchmark")

    args = parser.parse_args()

    if args.benchmark == 'all':
        run_all_benchmarks(
            model_repo=args.model_repo,
            model_file=args.model_file,
            verify=args.verify,
            output_file=args.output,
        )
    else:
        # Run individual benchmark
        print(f"Loading model: {args.model_repo} / {args.model_file}")

        with smll.Compressor.from_pretrained(repo_id=args.model_repo, filename=args.model_file) as compressor:
            if args.benchmark == 'content':
                benchmark_content_types(compressor, verify=args.verify)
            elif args.benchmark == 'length':
                benchmark_text_length(compressor, verify=args.verify)
            elif args.benchmark == 'speed':
                benchmark_speed(compressor)
            elif args.benchmark == 'h2h':
                benchmark_head_to_head(compressor, verify=args.verify)
            elif args.benchmark == 'impossible':
                benchmark_impossible_compression(compressor)
            elif args.benchmark == 'repro':
                benchmark_reproducibility(compressor)

        if args.benchmark == 'models':
            benchmark_model_sizes(verify=args.verify)
        elif args.benchmark == 'domain':
            benchmark_domain_models(verify=args.verify)


if __name__ == "__main__":
    main()
