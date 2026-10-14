"""Reconstruction quality of a tokenizer on the pretraining set (Table 1 of the paper).

PSNR / SSIM / LPIPS between each image and its encode->decode reconstruction, plus rFID of the
reconstructions against the same InceptionV3 reference the generation FID uses. Preprocessing
matches precompute_latents.py exactly: bicubic resize to 256, centre crop, ToTensor -> [0, 1].

Usage (torchmetrics and lpips are only needed here, so pull them in for this run):
    uv run --with torchmetrics --with lpips python -m minidog.recon_eval --vae-type e2e-invae
    uv run --with torchmetrics --with lpips python -m minidog.recon_eval --vae-type e2e-vavae
"""
import argparse, glob, io, tarfile, numpy as np, torch
from PIL import Image
from torchvision import transforms
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from omegaconf import OmegaConf
from minidog.config import Stage2Config
from minidog.utils.model_utils import instantiate_from_config
from minidog.eval import fid_from_moments
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

ap = argparse.ArgumentParser()
ap.add_argument("--vae-type", required=True)
ap.add_argument("--limit", type=int, default=0, help="0 = all images")
ap.add_argument("--batch", type=int, default=32)
a = ap.parse_args()

SHARDS = sorted(glob.glob("data/dog-t2i-diffusion-data/dogs_recaptioned_wds/*.tar"))
REF    = "data/dog-t2i-diffusion-data/dogs_recaptioned_stats.npz"
dev    = torch.device("cuda")
tf = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

# only the tokenizer section of the config is used; --vae-type picks the tokenizer
cfg = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(Stage2Config),
                                          OmegaConf.load("configs/pretrain_e2e-invae_128tok_mse_irepa_eupe.yaml")))
cfg.stage_1.params["vae_type"] = a.vae_type
vae = instantiate_from_config(cfg.stage_1).to(dev).eval()

inception = FeatureExtractorInceptionV3("inception", ["2048", "logits_unbiased"]).to(dev).eval()
ssim  = StructuralSimilarityIndexMeasure(data_range=1.0).to(dev)
lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(dev)

def images():
    for s in SHARDS:
        with tarfile.open(s) as tf_:
            for m in tf_:
                if m.name.endswith(".jpg"):
                    yield tf(Image.open(io.BytesIO(tf_.extractfile(m).read())).convert("RGB"))

psnrs, ssims, lps, feats, n, buf = [], [], [], [], 0, []
for img in images():
    buf.append(img)
    if len(buf) < a.batch: continue
    x = torch.stack(buf).to(dev); buf = []
    with torch.no_grad():
        r = vae.decode(vae.encode(x)).clamp(0, 1)
        mse = ((x - r) ** 2).flatten(1).mean(1)
        psnrs += (-10 * torch.log10(mse.clamp_min(1e-12))).tolist()
        ssims.append(ssim(r, x).item() * x.size(0))
        lps.append(lpips(r, x).item() * x.size(0))
        f, _ = inception(r.mul(255).clamp(0, 255).to(torch.uint8))
        feats.append(f.cpu().numpy())
    n += x.size(0)
    if a.limit and n >= a.limit: break
    if n % 2560 == 0: print(f"  {n} images...", flush=True)

feats = np.concatenate(feats)
mu, sigma = feats.mean(0), np.cov(feats, rowvar=False)
ref = np.load(REF)
rfid = fid_from_moments(mu, sigma, ref["mu"], ref["sigma"])
print(f"\n=== {a.vae_type}  ({n} images) ===")
print(f"  PSNR  {np.mean(psnrs):8.3f} dB")
print(f"  SSIM  {sum(ssims)/n:8.4f}")
print(f"  LPIPS {sum(lps)/n:8.4f}")
print(f"  rFID  {rfid:8.3f}")
