"""Smol compression library."""

from typing import Union

try:
    from . import _smol
except ImportError:
    import _smol


class Compressor:
    """
    Compressor for text using language models.

    The compressor loads a model once and can be used for multiple
    compress/decompress operations. The model is automatically freed
    when the Compressor is garbage collected or when used as a context manager.

    Example:
        >>> # Explicit lifecycle
        >>> compressor = Compressor("model.gguf")
        >>> compressed = compressor.compress("Hello, world!")
        >>> text = compressor.decompress(compressed)

        >>> # With context manager (recommended)
        >>> with Compressor("model.gguf") as compressor:
        ...     compressed = compressor.compress("Hello, world!")
        ...     text = compressor.decompress(compressed)
    """

    def __init__(self, model_path: str):
        """
        Create a new Compressor with the specified model.

        Args:
            model_path: Path to the GGUF model file

        Raises:
            RuntimeError: If the model cannot be loaded
        """
        self._compressor = _smol.Compressor(model_path)

    def compress(self, data: str) -> bytes:
        """
        Compress a string and return a bitstream.

        Args:
            data: String data to compress

        Returns:
            Compressed data as bytes (bit stream)
        """
        bitstream = self._compressor.compress(data)
        return bytes(bitstream)

    def decompress(self, bitstream: Union[bytes, bytearray]) -> str:
        """
        Decompress a bitstream and return a string.

        Args:
            bitstream: Compressed data as bytes

        Returns:
            Decompressed string

        Raises:
            RuntimeError: If decompression fails
        """
        return self._compressor.decompress(list(bitstream))

    @property
    def model_path(self) -> str:
        """Get the path to the loaded model."""
        return self._compressor.model_path

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit context manager."""
        # C++ destructor will be called automatically when _compressor is freed
        return False


__all__ = ["Compressor"]
