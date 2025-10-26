#!/usr/bin/env python3
"""Example script demonstrating the smll compression library."""

import smll


def main():
    # Example data to compress
    text = "Hello, this is a test string for compression!"

    print(f"Original text: {text}")
    print(f"Loading model from Hugging Face...")
    print()

    # Use context manager for automatic cleanup (recommended)
    try:
        with smll.Compressor.from_pretrained(
            repo_id="QuantFactory/Meta-Llama-3-8B-GGUF",
            filename="*Q8_0.gguf",
        ) as compressor:
            print(f"Model loaded: {compressor.model_path}")
            print()
            # Compress the data
            print("Compressing...")
            compressed = compressor.compress(text)
            print(f"Compressed size: {len(compressed)} bytes")
            print(f"Compressed data: {compressed.hex()}")
            print()

            # Decompress the data
            if compressed:
                try:
                    print("Decompressing...")
                    decompressed = compressor.decompress(compressed)
                    print(f"Decompressed text: {decompressed}")
                    print()

                    # Verify round-trip
                    if decompressed == text:
                        print("✓ Round-trip successful!")
                    else:
                        print("✗ Round-trip failed - data mismatch")
                except RuntimeError as e:
                    print(f"Decompression error: {e}")
            else:
                print(
                    "Note: Compression returned empty bitstream (stub implementation)"
                )
                print(
                    "Replace the C++ stubs in include/smll.cpp with actual implementation"
                )
        # Model is automatically freed here
    except RuntimeError as e:
        print(f"Error: {e}")
        print("Make sure the model file exists at the specified path")

    print()
    print("Alternative usage patterns:")
    print("1. Without context manager:")
    print("   compressor = smll.Compressor('model.gguf')")
    print("   compressed = compressor.compress('text')")
    print("   # Model freed automatically when compressor goes out of scope")
    print()
    print("2. From Hugging Face:")
    print("   compressor = smll.Compressor.from_pretrained(")
    print("       'TheBloke/Mistral-7B-Instruct-v0.1-GGUF',")
    print("       filename='mistral-7b-instruct-v0.1.Q4_K_M.gguf'")
    print("   )")


if __name__ == "__main__":
    main()
