from abench.data_loader.data_loader import ABLoader,ABCrossvalExperiment
from typing import List, Optional, Tuple, Dict, Iterator
from PIL import Image
import os

class ImageLoader(ABLoader):
    """
    ABLoader-conform dataloader that loads and batches image data.

    Each iteration yields a batch of images (and optionally context, metadata).
    """

    def __init__(
        self,
        image_paths: List[str],
        context_data: Optional[List[Any]] = None,
        metadata: Optional[dict] = None,
        batch_size: int = 32,
        with_context: bool = True,
        with_metadata: bool = True,
        transform: Optional[Any] = None
    ):
        """
        Args:
            image_paths (List[str]): List of image file paths.
            context_data (List[Any], optional): Aligned context entries.
            metadata (dict, optional): Dataset-level metadata.
            batch_size (int): Number of samples per batch.
            with_context (bool): Include context in each batch.
            with_metadata (bool): Include metadata in each batch.
            transform (callable, optional): Image transform (e.g. resize, normalize).
        """
        assert batch_size > 0, "batch_size must be a positive integer"
        self.image_paths = image_paths
        self.context_data = context_data or [None] * len(image_paths)
        self.metadata = metadata or {}
        self.batch_size = batch_size
        self.transform = transform
        self.with_context = with_context
        self.with_metadata = with_metadata

    def __iter__(self) -> Iterator:
        """
        Yields:
            batches of (images, [contexts], [metadata])
        """
        batch_data = []
        batch_context = []

        for idx, img_path in enumerate(self.image_paths):
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            batch_data.append(img)

            if self.with_context:
                batch_context.append(self.context_data[idx])

            # When batch is full or it's the end
            if len(batch_data) == self.batch_size or idx == len(self.image_paths) - 1:
                output = [batch_data.copy()]
                if self.with_context:
                    output.append(batch_context.copy())
                if self.with_metadata:
                    output.append(self.metadata.copy())

                yield output[0] if len(output) == 1 else tuple(output)

                # reset batch
                batch_data.clear()
                batch_context.clear()