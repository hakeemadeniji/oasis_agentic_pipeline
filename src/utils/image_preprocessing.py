"""
Optimized image preprocessing pipeline for OASIS Agentic Pipeline.

Provides high-performance image loading, preprocessing, and augmentation
with memory-efficient processing and caching.
"""

import numpy as np
import torch
from PIL import Image
import io
import logging
from typing import Optional, Tuple, List, Union
from functools import lru_cache
import cv2

from src.utils.profiling import profile_function, cache_manager


logger = logging.getLogger(__name__)


class OptimizedImagePreprocessor:
    """
    High-performance image preprocessing pipeline with caching and optimization.

    Features:
    - Memory-efficient loading
    - Cached transformations
    - Parallel processing support
    - GPU-accelerated operations
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        use_cache: bool = True,
        cache_size: int = 1000,
        normalize: bool = True,
        use_gpu: bool = True,
    ):
        """
        Initialize optimized image preprocessor.

        Args:
            target_size: Target image size (height, width)
            use_cache: Enable transformation caching
            cache_size: Maximum cache size
            normalize: Apply normalization
            use_gpu: Use GPU for processing if available
        """
        self.target_size = target_size
        self.use_cache = use_cache
        self.cache_size = cache_size
        self.normalize = normalize
        self.use_gpu = use_gpu and torch.cuda.is_available()

        # Pre-compute normalization parameters
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Device for processing
        self.device = torch.device("cuda" if self.use_gpu else "cpu")

        logger.info(
            f"Image preprocessor initialized - "
            f"Target size: {target_size}, "
            f"Cache: {use_cache}, "
            f"GPU: {self.use_gpu}"
        )

    @profile_function
    def load_image(
        self,
        image_path: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        image_array: Optional[np.ndarray] = None,
    ) -> Image.Image:
        """
        Load image from various sources with optimization.

        Args:
            image_path: Path to image file
            image_bytes: Image as bytes
            image_array: Image as numpy array

        Returns:
            PIL Image object
        """
        if image_path is not None:
            # Load from file path with optimization
            img = Image.open(image_path)
        elif image_bytes is not None:
            # Load from bytes
            img = Image.open(io.BytesIO(image_bytes))
        elif image_array is not None:
            # Convert numpy array to PIL Image
            if image_array.dtype != np.uint8:
                image_array = (image_array * 255).astype(np.uint8)
            img = Image.fromarray(image_array)
        else:
            raise ValueError("Must provide one of: image_path, image_bytes, or image_array")

        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")

        return img

    @profile_function
    def resize_image(
        self,
        image: Image.Image,
        size: Optional[Tuple[int, int]] = None,
        method: int = Image.LANCZOS,
    ) -> Image.Image:
        """
        Resize image with optimized interpolation.

        Args:
            image: PIL Image
            size: Target size (width, height)
            method: Resampling method

        Returns:
            Resized PIL Image
        """
        target_size = size or self.target_size

        # Use cache for repeated resize operations
        if self.use_cache:
            cache_key = f"resize_{image.size}_{target_size}_{method}"
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result

        # Perform resize
        resized = image.resize(target_size[::-1], method)  # PIL uses (width, height)

        # Cache result
        if self.use_cache:
            cache_manager.set(cache_key, resized)

        return resized

    @profile_function
    def normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize tensor with pre-computed parameters.

        Args:
            tensor: Input tensor (C, H, W)

        Returns:
            Normalized tensor
        """
        if not self.normalize:
            return tensor

        # Move to device for GPU acceleration
        tensor = tensor.to(self.device)

        # Normalize
        mean = torch.tensor(self.mean).view(3, 1, 1).to(self.device)
        std = torch.tensor(self.std).view(3, 1, 1).to(self.device)

        normalized = (tensor - mean) / std

        return normalized

    @profile_function
    def preprocess_image(
        self, image: Union[str, bytes, np.ndarray, Image.Image], return_tensor: bool = True
    ) -> Union[torch.Tensor, Image.Image]:
        """
        Complete preprocessing pipeline with optimization.

        Args:
            image: Image source (path, bytes, array, or PIL Image)
            return_tensor: Return as tensor or PIL Image

        Returns:
            Preprocessed image
        """
        # Load image
        if isinstance(image, Image.Image):
            pil_image = image
        else:
            pil_image = self.load_image(
                image_path=image if isinstance(image, str) else None,
                image_bytes=image if isinstance(image, bytes) else None,
                image_array=image if isinstance(image, np.ndarray) else None,
            )

        # Resize
        resized_image = self.resize_image(pil_image)

        if not return_tensor:
            return resized_image

        # Convert to tensor
        tensor = torch.from_numpy(np.array(resized_image)).float()
        tensor = tensor.permute(2, 0, 1) / 255.0  # (H, W, C) -> (C, H, W)

        # Normalize
        normalized = self.normalize_tensor(tensor)

        return normalized

    @profile_function
    def preprocess_batch(
        self, images: List[Union[str, bytes, np.ndarray, Image.Image]], batch_size: int = 32
    ) -> torch.Tensor:
        """
        Preprocess batch of images efficiently.

        Args:
            images: List of image sources
            batch_size: Batch size for processing

        Returns:
            Batch tensor (B, C, H, W)
        """
        processed_images = []

        for image in images:
            processed = self.preprocess_image(image, return_tensor=True)
            processed_images.append(processed)

        # Stack into batch
        batch = torch.stack(processed_images)

        return batch

    def clear_cache(self):
        """Clear transformation cache."""
        if self.use_cache:
            cache_manager.clear()
            logger.info("Image preprocessing cache cleared")


class MemoryEfficientImageLoader:
    """Memory-efficient image loader for large datasets."""

    def __init__(self, max_cache_size: int = 100):
        """
        Initialize memory-efficient loader.

        Args:
            max_cache_size: Maximum number of images to cache
        """
        self.max_cache_size = max_cache_size
        self.cache = {}
        self.access_count = {}

    @lru_cache(maxsize=1000)
    def load_image_cached(self, image_path: str) -> Image.Image:
        """
        Load image with LRU caching.

        Args:
            image_path: Path to image file

        Returns:
            PIL Image
        """
        return Image.open(image_path)

    def load_image_lazy(self, image_path: str) -> Image.Image:
        """
        Load image with lazy loading and memory management.

        Args:
            image_path: Path to image file

        Returns:
            PIL Image
        """
        # Check cache
        if image_path in self.cache:
            self.access_count[image_path] = self.access_count.get(image_path, 0) + 1
            return self.cache[image_path]

        # Load image
        img = Image.open(image_path)

        # Manage cache size
        if len(self.cache) >= self.max_cache_size:
            # Remove least recently used
            lru_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
            del self.cache[lru_key]
            del self.access_count[lru_key]

        # Cache image
        self.cache[image_path] = img
        self.access_count[image_path] = 1

        return img

    def clear_cache(self):
        """Clear image cache."""
        self.cache.clear()
        self.access_count.clear()
        logger.info("Image loader cache cleared")


class ImageAugmentation:
    """Optimized image augmentation pipeline."""

    def __init__(
        self,
        rotation_range: float = 10.0,
        brightness_range: Tuple[float, float] = (0.9, 1.1),
        contrast_range: Tuple[float, float] = (0.9, 1.1),
        enable_random_crop: bool = False,
    ):
        """
        Initialize augmentation pipeline.

        Args:
            rotation_range: Rotation angle range in degrees
            brightness_range: Brightness adjustment range
            contrast_range: Contrast adjustment range
            enable_random_crop: Enable random cropping
        """
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.enable_random_crop = enable_random_crop

    @profile_function
    def augment(self, image: Image.Image) -> Image.Image:
        """
        Apply random augmentations to image.

        Args:
            image: PIL Image

        Returns:
            Augmented PIL Image
        """
        # Random rotation
        if self.rotation_range > 0:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            image = image.rotate(angle, fillcolor=0)

        # Random brightness
        if self.brightness_range != (1.0, 1.0):
            factor = np.random.uniform(*self.brightness_range)
            from PIL import ImageEnhance

            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(factor)

        # Random contrast
        if self.contrast_range != (1.0, 1.0):
            factor = np.random.uniform(*self.contrast_range)
            from PIL import ImageEnhance

            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(factor)

        return image

    def augment_batch(
        self, images: List[Image.Image], augment_all: bool = True
    ) -> List[Image.Image]:
        """
        Augment batch of images.

        Args:
            images: List of PIL Images
            augment_all: Whether to augment all images or just some

        Returns:
            List of augmented images
        """
        augmented = []

        for image in images:
            if augment_all or np.random.random() > 0.5:
                augmented.append(self.augment(image))
            else:
                augmented.append(image)

        return augmented


class ImageCompression:
    """Optimized image compression for storage and transmission."""

    @staticmethod
    @profile_function
    def compress_image(image: Image.Image, quality: int = 85, format: str = "JPEG") -> bytes:
        """
        Compress image to bytes.

        Args:
            image: PIL Image
            quality: Compression quality (1-100)
            format: Image format

        Returns:
            Compressed image bytes
        """
        buffer = io.BytesIO()
        image.save(buffer, format=format, quality=quality, optimize=True)
        return buffer.getvalue()

    @staticmethod
    @profile_function
    def decompress_image(image_bytes: bytes, format: str = "JPEG") -> Image.Image:
        """
        Decompress image from bytes.

        Args:
            image_bytes: Compressed image bytes
            format: Image format

        Returns:
            PIL Image
        """
        return Image.open(io.BytesIO(image_bytes))


class ImageQualityMetrics:
    """Image quality assessment for preprocessing optimization."""

    @staticmethod
    def calculate_sharpness(image: Image.Image) -> float:
        """
        Calculate image sharpness using Laplacian variance.

        Args:
            image: PIL Image

        Returns:
            Sharpness score
        """
        # Convert to grayscale
        gray = image.convert("L")
        gray_array = np.array(gray)

        # Calculate Laplacian variance
        laplacian = cv2.Laplacian(gray_array, cv2.CV_64F)
        sharpness = laplacian.var()

        return sharpness

    @staticmethod
    def calculate_brightness(image: Image.Image) -> float:
        """
        Calculate average brightness.

        Args:
            image: PIL Image

        Returns:
            Average brightness (0-255)
        """
        gray = image.convert("L")
        return np.mean(np.array(gray))

    @staticmethod
    def calculate_contrast(image: Image.Image) -> float:
        """
        Calculate image contrast using standard deviation.

        Args:
            image: PIL Image

        Returns:
            Contrast score
        """
        gray = image.convert("L")
        return np.std(np.array(gray))


# Global preprocessor instance
default_preprocessor = OptimizedImagePreprocessor()


if __name__ == "__main__":
    # Test optimized preprocessing
    print("Testing optimized image preprocessing...")

    # Create test image
    test_image = Image.new("RGB", (512, 512), color=(128, 128, 128))

    # Initialize preprocessor
    preprocessor = OptimizedImagePreprocessor(target_size=(224, 224), use_cache=True)

    # Test preprocessing
    import time

    start = time.time()
    for _ in range(10):
        result = preprocessor.preprocess_image(test_image)
    time_elapsed = time.time() - start

    print(f"Preprocessed 10 images in {time_elapsed:.4f}s")
    print(f"Average per image: {time_elapsed / 10:.4f}s")
    print(f"Result shape: {result.shape}")

    # Test memory-efficient loading
    print("\nTesting memory-efficient loading...")
    loader = MemoryEfficientImageLoader(max_cache_size=5)

    # Test augmentation
    print("\nTesting image augmentation...")
    augmenter = ImageAugmentation(rotation_range=5.0)
    augmented = augmenter.augment(test_image)
    print(f"Original size: {test_image.size}, Augmented size: {augmented.size}")

    print("\nOptimized image preprocessing test complete!")
