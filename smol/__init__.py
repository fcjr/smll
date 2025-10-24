"""Smol compression library."""

from typing import Union, Optional

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

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        filename: Optional[str] = None,
        *,
        local_dir: Optional[str] = None,
        local_dir_use_symlinks: bool = True,
        cache_dir: Optional[str] = None,
        revision: Optional[str] = None,
    ) -> "Compressor":
        """
        Download and load a model from Hugging Face Hub.

        Args:
            repo_id: Hugging Face repository ID (e.g., "TheBloke/Mistral-7B-GGUF")
            filename: Specific model filename to download. If None, will try to auto-detect
            local_dir: Local directory to save the model (optional)
            local_dir_use_symlinks: Whether to use symlinks when downloading
            cache_dir: Optional cache directory for downloads
            revision: Git revision (branch, tag, or commit hash)

        Returns:
            Compressor instance with the downloaded model

        Example:
            >>> compressor = Compressor.from_pretrained(
            ...     "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
            ...     filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf"
            ... )
        """
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
        except ImportError:
            raise ImportError(
                "huggingface-hub is required to use from_pretrained. "
                "Install it with: uv add huggingface-hub"
            )

        # Auto-detect filename if not provided
        if filename is None:
            files = list_repo_files(repo_id, revision=revision)
            gguf_files = [f for f in files if f.endswith(".gguf")]

            if not gguf_files:
                raise ValueError(f"No GGUF files found in repository {repo_id}")

            if len(gguf_files) == 1:
                filename = gguf_files[0]
            else:
                raise ValueError(
                    f"Multiple GGUF files found in {repo_id}. "
                    f"Please specify one with the 'filename' parameter: {gguf_files}"
                )

        # Download the model file
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=local_dir_use_symlinks,
            cache_dir=cache_dir,
            revision=revision,
        )

        # Create and return Compressor instance
        return cls(model_path)


__all__ = ["Compressor"]
