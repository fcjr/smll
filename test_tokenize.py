#!/usr/bin/env python3
import smol
import ctypes

# Load model
compressor = smol.Compressor.from_pretrained(
    repo_id="QuantFactory/Meta-Llama-3-8B-GGUF",
    filename="*Q8_0.gguf"
)

# Access the C++ object to test tokenization directly
# We'll just use compress/decompress with a hack

text = "Hi"
print(f"Original: '{text}'")

# This will tokenize internally
compressed = compressor.compress(text)
num_tokens = int.from_bytes(compressed[:4], 'big')
print(f"Number of tokens: {num_tokens}")

# Try decompressing
decompressed = compressor.decompress(compressed)
print(f"Decompressed: '{decompressed}'")
print(f"Match: {text == decompressed}")

# Also test longer text
text2 = "Hello world"
print(f"\nOriginal: '{text2}'")
compressed2 = compressor.compress(text2)
num_tokens2 = int.from_bytes(compressed2[:4], 'big')
print(f"Number of tokens: {num_tokens2}")
decompressed2 = compressor.decompress(compressed2)
print(f"Decompressed: '{decompressed2}'")
print(f"Match: {text2 == decompressed2}")
