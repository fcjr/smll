#!/usr/bin/env python3
import smol

# Use a tiny model for faster testing
compressor = smol.Compressor.from_pretrained(
    repo_id="QuantFactory/Meta-Llama-3-8B-GGUF", filename="*Q8_0.gguf"
)

# Test with some text
text = """In information theory, data compression, source coding,[1] or bit-rate reduction is the process of encoding information using fewer bits than the original representation.[2] Any particular compression is either lossy or lossless. Lossless compression reduces bits by identifying and eliminating statistical redundancy. No information is lost in lossless compression. Lossy compression reduces bits by removing unnecessary or less important information.[3] Typically, a device that performs data compression is referred to as an encoder, and one that performs the reversal of the process (decompression) as a decoder.
        The process of reducing the size of a data file is often referred to as data compression. In the context of data transmission, it is called source coding: encoding is done at the source of the data before it is stored or transmitted.[4] Source coding should not be confused with channel coding, for error detection and correction or line coding, the means for mapping data onto a signal.
        Data compression algorithms present a space–time complexity trade-off between the bytes needed to store or transmit information, and the computational resources needed to perform the encoding and decoding. The design of data compression schemes involves balancing the degree of compression, the amount of distortion introduced (when using lossy data compression), and the computational resources or time required to compress and decompress the data.[5]"""
print(f"Original: '{text}'")

compressed = compressor.compress(text)
print(f"Compressed: {len(compressed)} bytes = {compressed.hex()}")

# Parse header
num_tokens = int.from_bytes(compressed[:4], "big")
print(f"Num tokens: {num_tokens}")

decompressed = compressor.decompress(compressed)
print(f"Decompressed: '{decompressed}'")
print(f"Match: {text == decompressed}")
