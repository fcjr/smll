#!/usr/bin/env python3
import smol

# Use a tiny model for faster testing
compressor = smol.Compressor.from_pretrained(
    repo_id="QuantFactory/Meta-Llama-3-8B-GGUF",
    filename="*Q8_0.gguf"
)

# Test with very simple text
text = "Hi"
print(f"Original: '{text}'")

compressed = compressor.compress(text)
print(f"Compressed: {len(compressed)} bytes = {compressed.hex()}")

# Parse header
num_tokens = int.from_bytes(compressed[:4], 'big')
print(f"Num tokens: {num_tokens}")

decompressed = compressor.decompress(compressed)
print(f"Decompressed: '{decompressed}'")
print(f"Match: {text == decompressed}")
