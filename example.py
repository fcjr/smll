#!/usr/bin/env python3
"""Example script demonstrating the smol compression library."""

import smol

def main():
    # Example data to compress
    text = "Hello, this is a test string for compression!"
    model_path = "models/gpt2.gguf"  # Path to your GGUF model file

    print(f"Original text: {text}")
    print(f"Using model: {model_path}")
    print()

    # Use context manager for automatic cleanup (recommended)
    try:
        with smol.Compressor(model_path) as compressor:
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
                print("Note: Compression returned empty bitstream (stub implementation)")
                print("Replace the C++ stubs in include/smol.cpp with actual implementation")
        # Model is automatically freed here
    except RuntimeError as e:
        print(f"Error: {e}")
        print("Make sure the model file exists at the specified path")

    print()
    print("Alternative usage without context manager:")
    print("compressor = smol.Compressor('model.gguf')")
    print("compressed = compressor.compress('text')")
    print("# Model freed automatically when compressor goes out of scope")

if __name__ == "__main__":
    main()
