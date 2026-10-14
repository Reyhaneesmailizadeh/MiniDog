"""Dog-breed WebDataset loaders: raw image shards and precomputed-latent shards."""
import io
import itertools
import json
import numpy as np
import random
import tarfile
import torch
import webdataset as wds
from dataclasses import dataclass, field
from pathlib import Path
from torch.utils.data import IterableDataset
from torchvision import transforms
from typing import Optional



def _not_none(sample):
    return sample is not None


def _filter_valid_samples(sample):
    return sample[0] is not None


class _ReshufflingShardList(IterableDataset):
    """Shard list that reshuffles with seed + pass_index each time it is iterated.

    Used with an infinite (`.repeat()`) pipeline so that loader workers and the shuffle buffer
    survive across epochs while every epoch still sees a fresh shard order.
    """

    def __init__(self, urls, seed: int):
        self.urls, self.seed, self.passes = list(urls), seed, 0

    def __iter__(self):
        urls = list(self.urls)
        random.Random(self.seed + self.passes).shuffle(urls)
        self.passes += 1
        for url in urls:
            yield dict(url=url)


class DogsWebDataset:
    """
    WebDataset wrapper for the dog breed dataset.

    Expects tar shards where each
    sample contains a .jpg image and a .txt caption.
    Returns (image_tensor, caption_string) pairs.
    """

    def __init__(
        self,
        data_dir: str,
        transform: Optional[transforms.Compose] = None,
        image_size: int = 256,
        shuffle_buffer: int = 5000,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

        tar_files = sorted(self.data_dir.glob("*.tar"))
        if not tar_files:
            raise ValueError(f"No tar shards found in {data_dir}.")

        self._shard_urls = [str(f) for f in tar_files]
        self._num_shards = len(self._shard_urls)
        self._num_samples = sum(_shard_sample_counts(self.data_dir, tar_files, IMAGE_SUFFIXES).values())

        self.transform = transform or transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

    @property
    def estimated_size(self) -> int:
        return self._num_samples

    @property
    def num_shards(self) -> int:
        return self._num_shards

    def _decode_sample(self, sample):
        image = sample.get("jpg") or sample.get("png") or sample.get("jpeg") or sample.get("webp")

        caption_raw = sample.get("txt", b"")
        caption = caption_raw.decode("utf-8").strip() if isinstance(caption_raw, bytes) else str(caption_raw).strip()

        if image is not None:
            image = self.transform(image)

        return image, caption

    def create_pipeline(self, shuffle: bool = True):
        """Training (shuffle=True): infinite stream, shards reshuffled every pass, sample shuffle buffer.
        Eval (shuffle=False): one ordered pass over all shards."""
        if shuffle:
            return wds.DataPipeline(
                _ReshufflingShardList(self._shard_urls, self.seed),
                wds.split_by_node,
                wds.split_by_worker,
                wds.tarfile_to_samples(handler=wds.ignore_and_continue),
                wds.shuffle(self.shuffle_buffer, initial=self.shuffle_buffer // 2),
                wds.decode("pil", handler=wds.ignore_and_continue),
                wds.map(self._decode_sample, handler=wds.ignore_and_continue),
                wds.select(_filter_valid_samples),
            ).repeat()
        return (
            wds.WebDataset(self._shard_urls, nodesplitter=None, shardshuffle=False)
            .decode("pil", handler=wds.ignore_and_continue)
            .map(self._decode_sample, handler=wds.ignore_and_continue)
            .select(_filter_valid_samples)
        )


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


def _count_tar_samples(path: Path, suffixes) -> int:
    """Count samples in a shard as the number of members ending in one of `suffixes`.

    Reads only tar headers, but scanning still costs ~1-3s per GB of shard, so callers
    cache the result (see _shard_sample_counts) rather than call this on every load.
    """
    with tarfile.open(path) as tf:
        return sum(1 for name in tf.getnames() if name.endswith(suffixes))


def _shard_sample_counts(data_dir: Path, tar_files: list, suffixes=(".latent.npy",)) -> dict:
    """Get exact sample counts per shard, cached in a sidecar file keyed by (mtime, size).

    A full tar scan takes seconds to minutes depending on shard size, and every
    torchrun rank instantiates its own DogsLatentsWebDataset, so without caching this
    cost is paid repeatedly on every training launch. The cache is invalidated per-file
    if its mtime/size changes (e.g. shards regenerated).
    """
    cache_path = data_dir / ".shard_sample_counts.json"
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            cache = {}

    counts = {}
    dirty = False
    for f in tar_files:
        stat = f.stat()
        entry = cache.get(f.name)
        if entry and entry.get("mtime") == stat.st_mtime and entry.get("size") == stat.st_size:
            counts[f.name] = entry["count"]
            continue
        count = _count_tar_samples(f, suffixes)
        counts[f.name] = count
        cache[f.name] = {"mtime": stat.st_mtime, "size": stat.st_size, "count": count}
        dirty = True

    if dirty:
        try:
            cache_path.write_text(json.dumps(cache))
        except OSError:
            pass  # e.g. read-only data dir; caching is an optimization, not required for correctness

    return counts


class DogsLatentsWebDataset:
    """
    Reads pre-computed latent shards produced by python -m minidog.precompute_latents.

    Each tar sample contains:
        latent.npy      [C, H, W]              float16 → float32
        tokens.npy      [seq_len, dim]          float16 → float32
        attn_mask.npy   [seq_len]               bool
        dinov2.npy      [num_patches, dim]      float16 → float32 (DINOv2 ViT-B/14 REPA target)
        dinov3.npy      [num_patches, dim]      float16 → float32 (DINOv3 ViT-B/16 REPA target)
        pe_spatial.npy  [num_patches, dim]      float16 → float32 (PE-Spatial-B16 REPA target)
        eupe.npy        [num_patches, dim]      float16 → float32 (EUPE ViT-B/16 REPA target)

    A directory holds exactly one of dinov2.npy, dinov3.npy, pe_spatial.npy or
    eupe.npy,
    depending on which precompute script produced it.

    Returns (latent, tokens, attn_mask, dino_features) tuples where dino_features is a
    zero tensor (shape [1]) when neither is present, so WebLoader can collate
    homogeneously.
    """

    def __init__(self, data_dir: str, shuffle_buffer: int = 5000, seed: int = 42):
        self.data_dir = Path(data_dir)
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

        tar_files = sorted(self.data_dir.glob("*.tar"))
        if not tar_files:
            raise ValueError(f"No tar shards found in {data_dir}. Run python -m minidog.precompute_latents first.")
        self._shard_urls = [str(f) for f in tar_files]
        self._num_shards = len(self._shard_urls)
        self._num_samples = sum(_shard_sample_counts(self.data_dir, tar_files).values())

    @property
    def estimated_size(self) -> int:
        return self._num_samples

    @property
    def num_shards(self) -> int:
        return self._num_shards

    def _decode_sample(self, sample):
        try:
            latent = torch.from_numpy(np.load(io.BytesIO(sample["latent.npy"])).astype(np.float32))
            tokens = torch.from_numpy(np.load(io.BytesIO(sample["tokens.npy"])).astype(np.float32))
            attn_mask = torch.from_numpy(np.load(io.BytesIO(sample["attn_mask.npy"])))
            # DINO patch features used as the REPA target.
            # dinov2.npy is written by minidog.precompute_latents (DINOv2 ViT-B/14).
            # dinov3.npy is written by minidog.precompute_latents_dinov3 (DINOv3 ViT-B/16).
            # A shard directory contains one or the other, never both.
            if "dinov2.npy" in sample:
                dino_features = torch.from_numpy(np.load(io.BytesIO(sample["dinov2.npy"])).astype(np.float32))
            elif "dinov3.npy" in sample:
                dino_features = torch.from_numpy(np.load(io.BytesIO(sample["dinov3.npy"])).astype(np.float32))
            elif "pe_spatial.npy" in sample:
                dino_features = torch.from_numpy(np.load(io.BytesIO(sample["pe_spatial.npy"])).astype(np.float32))
            elif "eupe.npy" in sample:
                dino_features = torch.from_numpy(np.load(io.BytesIO(sample["eupe.npy"])).astype(np.float32))
            else:
                dino_features = torch.zeros(1)
            return latent, tokens, attn_mask, dino_features
        except Exception:
            return None

    def create_pipeline(self, shuffle: bool = True):
        """Training (shuffle=True): infinite stream, shards reshuffled every pass, sample shuffle buffer.
        Eval (shuffle=False): one ordered pass over all shards."""
        if shuffle:
            return wds.DataPipeline(
                _ReshufflingShardList(self._shard_urls, self.seed),
                wds.split_by_node,
                wds.split_by_worker,
                wds.tarfile_to_samples(handler=wds.ignore_and_continue),
                wds.shuffle(self.shuffle_buffer, initial=self.shuffle_buffer // 2),
                wds.map(self._decode_sample, handler=wds.ignore_and_continue),
                wds.select(_not_none),
            ).repeat()
        return (
            wds.WebDataset(self._shard_urls, nodesplitter=None, shardshuffle=False)
            .map(self._decode_sample, handler=wds.ignore_and_continue)
            .select(_not_none)
        )


@dataclass
class DataloaderResult:
    """Loader plus the bookkeeping the training loop needs (steps per epoch, epoch hooks)."""
    loader: wds.WebLoader
    dataset_size: int
    infinite: bool = False          # training loaders stream forever; each "epoch" is `len(self)` batches
    _batch_size: int = 1
    _world_size: int = 1
    virtual_epoch_steps: Optional[int] = None
    _iterator: Optional[object] = field(default=None, repr=False)

    def set_epoch(self, epoch: int):
        """Kept for API symmetry; the infinite stream reshuffles shards itself on every pass."""

    def __len__(self) -> int:
        if self.virtual_epoch_steps is not None:
            return self.virtual_epoch_steps
        return self.dataset_size // (self._batch_size * self._world_size)

    def __iter__(self):
        if not self.infinite:
            return iter(self.loader)
        if self._iterator is None:  # created once: loader workers and the shuffle buffer stay warm across epochs
            self._iterator = iter(self.loader)
        return itertools.islice(self._iterator, len(self))


def prepare_unified_dataloader(
    config: dict,
    image_size: int,
    batch_size: int,
    num_workers: int,
    rank: int,
    world_size: int,
    transform: Optional[transforms.Compose] = None,
    condition_type: str = "text",
    shuffle: bool = True,
    virtual_epoch_steps: Optional[int] = None,
) -> DataloaderResult:
    """
    Dataloader factory for the dog-breed WebDataset shards.

    Args:
        config: Dataset configuration dict with structure:
            {
                'target': 'dogs',
                'type': 'wds' | 'latents',   # raw image shards or precomputed latents
                'data_dir': str,
                'shuffle_buffer': int,
                'seed': int,
            }
        image_size: Target image resolution (ignored for precomputed latents)
        batch_size: Per-GPU batch size
        num_workers: Number of dataloader workers
        rank: Distributed training rank (unused; WebDataset splits shards by node)
        world_size: Total number of GPUs
        transform: Optional custom transform (raw image shards only)
        condition_type: Kept for interface compatibility; the dog data is text-conditioned
        shuffle: Whether to shuffle data (False for eval)
        virtual_epoch_steps: Optional fixed number of steps per epoch

    Returns:
        DataloaderResult with unified interface
    """
    target = config.get("target", "dogs")
    if target != "dogs":
        raise ValueError(f"Unsupported dataset target {target!r}; MiniDog only supports 'dogs'.")

    if config.get("type", "wds") == "latents":
        result = _prepare_dogs_latents_loader(
            config, batch_size, num_workers, world_size, shuffle
        )
    else:
        result = _prepare_dogs_loader(
            config, image_size, batch_size, num_workers, world_size, transform, shuffle
        )
    result.virtual_epoch_steps = virtual_epoch_steps
    return result


def _prepare_dogs_loader(
    config: dict,
    image_size: int,
    batch_size: int,
    num_workers: int,
    world_size: int,
    transform: Optional[transforms.Compose],
    shuffle: bool = True,
) -> DataloaderResult:
    """Prepare Dogs WebDataset loader."""

    data_dir = config.get("data_dir", "./data/dogs_wds")
    shuffle_buffer = config.get("shuffle_buffer", 5000)
    seed = config.get("seed", 42)

    wds_pipeline = DogsWebDataset(
        data_dir=data_dir,
        transform=transform,
        image_size=image_size,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
    )

    dataset = wds_pipeline.create_pipeline(shuffle=shuffle)
    total_samples = wds_pipeline.estimated_size

    # For eval (shuffle=False) skip node splitting so all shards are visible on each rank
    actual_num_workers = num_workers if shuffle else 0
    loader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        num_workers=actual_num_workers,
        pin_memory=True,
        multiprocessing_context="spawn" if actual_num_workers > 0 else None,
    )
    return DataloaderResult(
        loader=loader,
        dataset_size=total_samples,
        infinite=shuffle,
        _batch_size=batch_size,
        _world_size=world_size,
    )


def _prepare_dogs_latents_loader(
    config: dict,
    batch_size: int,
    num_workers: int,
    world_size: int,
    shuffle: bool = True,
) -> DataloaderResult:
    """Prepare Dogs pre-computed latents WebDataset loader."""

    data_dir = config.get("data_dir", "./data/dogs_latents_wds")
    shuffle_buffer = config.get("shuffle_buffer", 5000)
    seed = config.get("seed", 42)

    wds_pipeline = DogsLatentsWebDataset(
        data_dir=data_dir,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
    )

    dataset = wds_pipeline.create_pipeline(shuffle=shuffle)
    total_samples = wds_pipeline.estimated_size

    actual_num_workers = num_workers if shuffle else 0
    loader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        num_workers=actual_num_workers,
        pin_memory=True,
        multiprocessing_context="spawn" if actual_num_workers > 0 else None,
    )
    return DataloaderResult(
        loader=loader,
        dataset_size=total_samples,
        infinite=shuffle,
        _batch_size=batch_size,
        _world_size=world_size,
    )
