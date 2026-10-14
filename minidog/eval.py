"""FID / Inception Score evaluation during training."""
import logging
import numpy as np
import scipy.linalg
import sys
import tarfile
import torch
import torch.distributed as dist
from dataclasses import dataclass
from pathlib import Path
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset, Subset
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from tqdm import tqdm
from typing import Dict, Optional


def fid_from_moments(mu1, sigma1, mu2, sigma2) -> float:
    """Frechet distance between two Gaussians given their means and covariances."""
    mu1, mu2 = np.asarray(mu1, dtype=np.float64), np.asarray(mu2, dtype=np.float64)
    sigma1, sigma2 = np.asarray(sigma1, dtype=np.float64), np.asarray(sigma2, dtype=np.float64)
    diff = mu1 - mu2
    covmean = scipy.linalg.sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):  # numerical noise
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(max(fid, 0.0))


def inception_score(logits, num_splits=10) -> float:
    """Inception Score (mean over splits) from raw InceptionV3 logits."""
    probs = torch.nn.functional.softmax(logits.float(), dim=1).numpy()
    probs = probs[np.random.RandomState(0).permutation(len(probs))]
    split_size = len(probs) // num_splits
    scores = []
    for i in range(num_splits):
        part = probs[i * split_size : (i + 1) * split_size]
        p_y = part.mean(axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-10) - np.log(p_y + 1e-10))
        scores.append(float(np.exp(kl.sum(axis=1).mean())))
    return float(np.mean(scores))


def create_eval_dataloader(dataset, rank: int, world_size: int, num_samples: int, batch_size: int) -> DataLoader:
    """Shard the eval dataset across ranks and return a DataLoader for this rank's slice."""
    n = min(len(dataset), num_samples)
    chunk = n // world_size
    start = rank * chunk
    end = n if rank == world_size - 1 else (rank + 1) * chunk
    return DataLoader(Subset(dataset, list(range(start, end))), batch_size=batch_size, shuffle=False, num_workers=0)


logger = logging.getLogger(__name__)


class ListDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


@dataclass
class EvalDatasetInfo:
    dataset: Dataset          # captions only
    reference_npz: str        # precomputed InceptionV3 mu/sigma of the real images


def read_captions(shard_dir: str) -> list:
    """Captions of every sample in a folder of jpg+txt WebDataset shards, read straight from the tars."""
    captions = []
    for shard in sorted(Path(shard_dir).glob("*.tar")):
        with tarfile.open(shard) as tf:
            for member in tf:
                if member.name.endswith(".txt"):
                    captions.append(tf.extractfile(member).read().decode("utf-8").strip())
    return captions


def prepare_eval_datasets(eval_datasets_config: Dict[str, dict]) -> Dict[str, EvalDatasetInfo]:
    """For each eval.datasets entry, collect the captions (generation needs no images) and the FID reference."""
    eval_datasets = {}
    for ds_name, ds_cfg in eval_datasets_config.items():
        if 'reference_npz' not in ds_cfg:
            raise ValueError(f"eval.datasets.{ds_name} missing 'reference_npz', required for FID")
        captions = read_captions(ds_cfg['data_dir'])
        if not captions:
            raise ValueError(f"No captions found in shards under {ds_cfg['data_dir']}")
        eval_datasets[ds_name] = EvalDatasetInfo(dataset=ListDataset(captions), reference_npz=ds_cfg['reference_npz'])
        logger.info(f"Eval dataset loaded: {ds_name}, {len(captions)} captions")
    return eval_datasets


@torch.no_grad()
def evaluate_generation_distributed(
    model_fn,
    sample_fn,
    latent_size,
    additional_model_kwargs,
    use_guidance: bool,
    vae,
    val_dataset,
    num_samples: int,
    batch_size: int,
    rank: int,
    world_size: int,
    device: torch.device,
    global_step: int,
    autocast_kwargs: dict,
    reference_npz_path: str,
    text_encoder,
) -> Optional[Dict[str, float]]:
    """Generate one image per caption in val_dataset and compute FID and Inception Score.

    Each rank samples its shard and extracts InceptionV3 features inline; the small feature
    vectors are gathered and the metrics computed on rank 0. Returns the metrics dict on
    rank 0 and None on other ranks.
    """
    dist.barrier()
    loader = create_eval_dataloader(val_dataset, rank, world_size, num_samples, batch_size)
    inception = FeatureExtractorInceptionV3(name="inception-v3-compat", features_list=['2048', 'logits_unbiased']).to(device).eval()

    local_feats, local_logits = [], []
    iterator = tqdm(loader, desc=f"[Rank {rank}] Sampling", file=sys.stdout) if rank == 0 else loader
    with torch.inference_mode():
        for captions in iterator:
            captions = list(captions)
            n = len(captions)
            z = torch.randn(n, *latent_size, device=device)
            enc_out = text_encoder(captions)
            context, attn_mask = enc_out["tokens"], enc_out["attention_mask"]
            if use_guidance:
                z = torch.cat([z, z], dim=0)
                enc_null = text_encoder([""] * n)
                context = torch.cat([context, enc_null["tokens"]], dim=0)
                attn_mask = torch.cat([attn_mask, enc_null["attention_mask"]], dim=0)

            with autocast(**autocast_kwargs):
                samples = sample_fn(z, model_fn, context=context, attn_mask=attn_mask, **additional_model_kwargs)[-1]
                if use_guidance:
                    samples = samples.chunk(2, dim=0)[0]
                samples = vae.decode(samples).clamp(0, 1)

            feats, logits = inception(samples.mul(255).clamp(0, 255).to(dtype=torch.uint8))
            local_feats.append(feats.cpu())
            local_logits.append(logits.cpu())

    del inception
    torch.cuda.empty_cache()

    # all_gather needs equal sizes: pad the last rank's shard, trim after gathering
    local_feats = torch.cat(local_feats, dim=0)
    local_logits = torch.cat(local_logits, dim=0)
    max_chunk = -(-num_samples // world_size)
    pad = max_chunk - local_feats.shape[0]
    if pad > 0:
        local_feats = torch.cat([local_feats, local_feats.new_zeros(pad, local_feats.shape[1])])
        local_logits = torch.cat([local_logits, local_logits.new_zeros(pad, local_logits.shape[1])])
    local_feats, local_logits = local_feats.to(device), local_logits.to(device)
    gathered_feats = [torch.zeros_like(local_feats) for _ in range(world_size)]
    gathered_logits = [torch.zeros_like(local_logits) for _ in range(world_size)]
    dist.all_gather(gathered_feats, local_feats)
    dist.all_gather(gathered_logits, local_logits)

    metrics = None
    if rank == 0:
        all_feats = torch.cat(gathered_feats, dim=0)[:num_samples].cpu().double().numpy()
        all_logits = torch.cat(gathered_logits, dim=0)[:num_samples].cpu()
        ref = np.load(reference_npz_path)
        mu_ref = ref['mu'] if 'mu' in ref else ref['ref_mu']
        sigma_ref = ref['sigma'] if 'sigma' in ref else ref['ref_sigma']
        metrics = {
            'fid': fid_from_moments(all_feats.mean(axis=0), np.cov(all_feats, rowvar=False), mu_ref, sigma_ref),
            'inception_score': inception_score(all_logits),
        }
        print(f"[Eval] Step {global_step}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    dist.barrier()
    return metrics
