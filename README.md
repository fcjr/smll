# smll

LLM-powered text compression. Combines a language model with arithmetic coding to approach Shannon's theoretical entropy limit for text compression.

The text is tokenized and fed to the LLM, which outputs probability distributions for the next token. Arithmetic coding then converts these probabilities into bits proportional to `-log2(probability)`. Both the compressor and decompressor use the same model weights as a shared codebook.

For a deeper explanation, see the [blog post](https://www.frankchiarulli.com/blog/smll/).

## Installation

```bash
pip install pysmll
```

## Usage

```python
import smll

with smll.Compressor.from_pretrained(
    repo_id="QuantFactory/SmolLM2-360M-GGUF",
    filename="*Q4_0.gguf",
) as compressor:
    compressed = compressor.compress("Hello, world!")
    decompressed = compressor.decompress(compressed)
```

You can also load a local GGUF model directly:

```python
compressor = smll.Compressor("model.gguf")
compressed = compressor.compress("Hello, world!")
decompressed = compressor.decompress(compressed)
```

## Compression results

Using SmolLM2-360M on an Apple M4 Max (128GB):

| Data type | Compression ratio |
|---|---|
| LLM-generated text | 14.96x |
| Wikipedia | 14.83x |
| C code | 11.19x |
| Natural prose | 9.75x |
| Random data (UUIDs) | 0.94x |

At 1000 characters, smll achieves ~0.85 bits/character, near the estimated English entropy of 0.6-1.3 bits/char.

## Tradeoffs

- **Speed**: ~700 chars/sec (gzip: ~6.5M chars/sec). A 10KB document compresses in ~15 seconds.
- **Model size**: Both sides need the model weights (~200MB for SmolLM2-360M Q4).

## Building from source

Requires Python 3.12+, CMake 3.21+, and a C++11 compiler.

```bash
git clone --recurse-submodules https://github.com/fcjr/smll.git
cd smll
pip install .
```

## License

MIT
