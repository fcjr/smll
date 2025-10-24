"""Smol compression library."""

import json
import os
from pathlib import Path
from typing import Union, Optional, List

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
        additional_files: Optional[List[str]] = None,
    ) -> "Compressor":
        """
        Download and load a model from Hugging Face Hub.

        Args:
            repo_id: The model repo id (e.g., "TheBloke/Mistral-7B-GGUF")
            filename: A filename or glob pattern to match the model file in the repo
            additional_files: A list of filenames or glob patterns to match additional model files in the repo

        Returns:
            Compressor instance with the downloaded model

        Example:
            >>> compressor = Compressor.from_pretrained(
            ...     "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
            ...     filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf"
            ... )
        """
        try:
            from huggingface_hub import hf_hub_download, HfFileSystem
            from huggingface_hub.utils import validate_repo_id
            import fnmatch
        except ImportError:
            raise ImportError(
                "Compressor.from_pretrained requires the huggingface-hub package. "
                "You can install it with: uv add huggingface-hub"
            )

        validate_repo_id(repo_id)

        hffs = HfFileSystem()

        files = [
            file["name"] if isinstance(file, dict) else file
            for file in hffs.ls(repo_id, recursive=True)
        ]

        # split each file into repo_id, subfolder, filename
        file_list: List[str] = []
        for file in files:
            rel_path = Path(file).relative_to(repo_id)
            file_list.append(str(rel_path))

        # find the only/first shard file:
        matching_files = [file for file in file_list if fnmatch.fnmatch(file, filename)]  # type: ignore

        if len(matching_files) == 0:
            raise ValueError(
                f"No file found in {repo_id} that match {filename}\n\n"
                f"Available Files:\n{json.dumps(file_list)}"
            )

        if len(matching_files) > 1:
            raise ValueError(
                f"Multiple files found in {repo_id} matching {filename}\n\n"
                f"Available Files:\n{json.dumps(file_list)}"
            )

        (matching_file,) = matching_files

        subfolder = str(Path(matching_file).parent)
        filename = Path(matching_file).name

        # download the file
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder,
        )

        if additional_files:
            for additional_file_name in additional_files:
                # find the additional shard file:
                matching_additional_files = [
                    file for file in file_list if fnmatch.fnmatch(file, additional_file_name)
                ]

                if len(matching_additional_files) == 0:
                    raise ValueError(
                        f"No file found in {repo_id} that match {additional_file_name}\n\n"
                        f"Available Files:\n{json.dumps(file_list)}"
                    )

                if len(matching_additional_files) > 1:
                    raise ValueError(
                        f"Multiple files found in {repo_id} matching {additional_file_name}\n\n"
                        f"Available Files:\n{json.dumps(file_list)}"
                    )

                (matching_additional_file,) = matching_additional_files

                # download the additional file
                hf_hub_download(
                    repo_id=repo_id,
                    filename=matching_additional_file,
                    subfolder=subfolder,
                )

        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder,
            local_files_only=True,
        )

        # Create and return Compressor instance
        return cls(model_path)


__all__ = ["Compressor"]
