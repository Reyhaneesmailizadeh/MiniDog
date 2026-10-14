"""Rectified-flow transport, Euler ODE sampler, classifier-free guidance, and text-conditioning helpers."""
from __future__ import annotations

import dataclasses
import torch
import torch.nn.functional as F
from copy import deepcopy
from functools import partial
from minidog.config import GuidanceConfig
from minidog.text_encoder import TextEncoder
from minidog.utils.dist_utils import main_process_first
from torch.cuda.amp import autocast


def apply_cfg_dropout(model_conds, model_conds_null, cfg_dropout_prob=0.1):
    batch_size = model_conds['context'].shape[0]
    mask = torch.rand(batch_size, device=model_conds['context'].device) < cfg_dropout_prob
    return {
        k: torch.where(mask.view(-1, *([1]*(v.ndim-1))), model_conds_null[k][:batch_size], v) if v is not None else None
        for k, v in model_conds.items()
    }, mask


def get_null_cond(text_encoder, batch_size):
    """Empty-caption conditioning, expanded to batch_size (used for CFG dropout)."""
    context, attn_mask = encode_text(text_encoder, [""])
    return dict(context=context.expand(batch_size, *context.shape[1:]),
                attn_mask=attn_mask.expand(batch_size, *attn_mask.shape[1:]))


def setup_text_encoder(config, rank, device):
    """Build the text encoder and set config.conditioning.context_dim from it."""
    with main_process_first(rank):
        text_encoder = TextEncoder(**dataclasses.asdict(config.conditioning.text_encoder)).to(device)
    config.conditioning.context_dim = text_encoder.feature_dim
    return text_encoder


def encode_text(text_encoder, y):
    """Encode captions. Returns (token_embeddings, attention_mask)."""
    with torch.no_grad():
        enc_out = text_encoder(y)
        return enc_out["tokens"], enc_out["attention_mask"]


def get_fixed_viz_batch_conditions(viz_fixed, y, text_encoder):
    """Capture the first batch's captions as fixed conditions for consistent visualization."""
    if viz_fixed['context'] is not None:
        return viz_fixed
    n = viz_fixed['zs'].shape[0]
    viz_fixed['prompts'] = list(y[:n])
    viz_fixed['context'], viz_fixed['attn_mask'] = encode_text(text_encoder, y[:n])
    return viz_fixed


def sample_and_decode(zs, context, attn_mask, eval_sampler, model_fn, sample_model_kwargs, vae,
                      use_guidance, text_encoder, autocast_kwargs):
    """Generate latents from noise and decode them; doubles the batch for CFG."""
    n = zs.shape[0]
    if use_guidance:
        zs = torch.cat([zs, zs], dim=0)
        context_null, attn_mask_null = encode_text(text_encoder, [""] * n)
        context = torch.cat([context, context_null], dim=0)
        attn_mask = torch.cat([attn_mask, attn_mask_null], dim=0)

    kwargs = deepcopy(sample_model_kwargs)
    kwargs.update(context=context, attn_mask=attn_mask)
    with autocast(**autocast_kwargs):
        samples = eval_sampler(zs, model_fn, **kwargs)[-1]
        if use_guidance:
            samples = samples.chunk(2, dim=0)[0]
    return vae.decode(samples).cpu().float()


def _expand_t(t, x):
    return t.view(t.size(0), *([1] * (len(x.size()) - 1)))


def get_time_sampler(time_dist_type: str):
    parts = time_dist_type.split("_")
    name = parts[0]
    if name == "logit-normal":
        assert len(parts) == 3, f"Expected 'logit-normal_MU_SIGMA', got '{time_dist_type}'"
        mu, sigma = float(parts[1]), float(parts[2])
        assert sigma > 0, "sigma must be > 0"
        return lambda bs: (torch.randn(bs) * sigma + mu).sigmoid()
    raise NotImplementedError(f"Unknown time distribution: {time_dist_type}")


class Transport:
    """Rectified-flow transport: x_t = (1 - t) x_1 + t x_0, with x_1 data and x_0 noise."""

    def __init__(self, prediction="velocity", time_dist_type="logit-normal_0_1",
                 time_dist_shift=1.0, time_dist_shift_eval=1.0, t_eps=0.05):
        assert prediction in ("velocity", "x"), f"Unknown prediction type: {prediction}"
        self.prediction = prediction
        self.time_dist_type = time_dist_type
        self.time_dist_shift = time_dist_shift
        self.time_dist_shift_eval = time_dist_shift_eval
        self.t_eps = t_eps
        self.time_sampler = get_time_sampler(time_dist_type)

    def sample(self, x1):
        x0 = torch.randn_like(x1)
        t = self.time_sampler(x1.shape[0]).to(x1)
        t = self.time_dist_shift * t / (1 + (self.time_dist_shift - 1) * t)
        return t, x0, x1

    def training_losses(self, model, x1, model_kwargs, model_kwargs_null,
                        z_clean=None, repa_coeff=None, repa_loss_type="mse", cfg_dropout_prob=0.1,
                        irepa=False, irepa_gamma=1.0):
        """Flow-matching loss, plus the REPA alignment loss when z_clean and repa_coeff are given."""
        model_kwargs, _ = apply_cfg_dropout(model_kwargs, model_kwargs_null, cfg_dropout_prob)

        t, x0, x1 = self.sample(x1)
        xt = (1 - _expand_t(t, x1)) * x1 + _expand_t(t, x1) * x0
        vt = (xt - x1) / _expand_t(t, xt).clamp_min(self.t_eps)

        enable_repa = z_clean is not None and repa_coeff is not None
        if enable_repa:
            model_output, zt_pred = model(xt, t, return_intermediate=True, **model_kwargs)
        else:
            model_output = model(xt, t, **model_kwargs)

        terms = {'loss': self.compute_loss(model_output, vt, xt, t)}
        if enable_repa:
            # "mse": what this repo has always used. "cosine": the objective the REPA
            # (arXiv 2410.06940) and iREPA (arXiv 2512.10794) papers actually specify --
            # per-token 1 - cos, averaged. Cosine divides out each vector's norm, so the
            # teacher's feature scale no longer sets the effective loss weight.
            if irepa:
                # iREPA Alg. 1: normalise the external representation over the SPATIAL
                # dimension (dim=1, the patch tokens), not over channels. This removes
                # each channel's offset and scale, leaving the pairwise structure
                # between patches -- which the paper argues is what actually transfers.
                z_clean = z_clean - irepa_gamma * z_clean.mean(dim=1, keepdim=True)
                z_clean = z_clean / (z_clean.std(dim=1, keepdim=True) + 1e-6)
            if repa_loss_type == "cosine":
                repa_term = (1.0 - F.cosine_similarity(zt_pred, z_clean, dim=-1)).mean()
            elif repa_loss_type == "mse":
                repa_term = F.mse_loss(zt_pred, z_clean)
            else:
                raise ValueError(f"unknown repa_loss_type: {repa_loss_type!r}")
            terms['loss_repa'] = repa_coeff * repa_term
        else:
            terms['loss_repa'] = torch.tensor(0.0, device=x1.device)
        return terms

    def convert_model_pred(self, output, xt, t):
        """Unify the model output to a velocity."""
        if self.prediction == "velocity":
            return output
        t_safe = _expand_t(t, xt).clamp_min(self.t_eps)
        return (xt - output) / t_safe

    def compute_loss(self, output, vt, xt, t):
        return (self.convert_model_pred(output, xt, t) - vt) ** 2

    def get_drift(self):
        def body_fn(x, t, h, model, **model_kwargs):
            return self.convert_model_pred(model(x, t, **model_kwargs), x, t)
        return body_fn


def create_transport(config, time_dist_shift=1.0, time_dist_shift_eval=1.0):
    """Create a Transport from a TransportConfig."""
    return Transport(prediction=config.prediction, time_dist_type=config.time_dist_type,
                     time_dist_shift=time_dist_shift, time_dist_shift_eval=time_dist_shift_eval, t_eps=config.t_eps)


class Sampler:
    def __init__(self, transport):
        self.transport = transport
        self.drift = self.transport.get_drift()

    def sample_ode(self, *, num_steps=50):
        t_grid = torch.linspace(1.0, 0.0, num_steps + 1)
        shift = self.transport.time_dist_shift_eval
        t_grid = shift * t_grid / (1 + (shift - 1) * t_grid)

        def sample_fn(x, model, **model_kwargs):
            device = x.device
            t_steps = t_grid.to(device)
            B = x.shape[0]

            model_kwargs_ = model_kwargs.copy()

            for i in range(num_steps):
                h = t_steps[i] - t_steps[i + 1]
                h_batch = torch.full((B,), h.item(), device=device)
                t_batch = torch.full((B,), t_steps[i].item(), device=device)
                d_cur = self.drift(x, t_batch, h_batch, model, **model_kwargs_)
                x = x - h * d_cur

            return x.unsqueeze(0)

        return sample_fn


def forward_with_cfg(model, x, t, cfg_scale, cfg_interval=(0, 1), **condition_kwargs):
    """Forward pass with classifier-free guidance. Expects a doubled batch [cond, uncond]."""
    half = x[: len(x) // 2]
    combined = torch.cat([half, half], dim=0)
    eps = model(combined, t, **condition_kwargs)
    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    guid_t_min, guid_t_max = cfg_interval
    assert guid_t_min < guid_t_max, "cfg_interval should be (min, max) with min < max"
    t = t[: len(t) // 2]
    half_eps = torch.where(
        ((t >= guid_t_min) & (t <= guid_t_max)).view(-1, *[1] * (cond_eps.ndim - 1)),
        uncond_eps + cfg_scale * (cond_eps - uncond_eps), cond_eps
    )
    return torch.cat([half_eps, half_eps], dim=0)


def get_model_forward_fn(model, guid_cfg: GuidanceConfig):
    """Return (model_fn, sample_kwargs): the CFG wrapper when guidance is active, else plain forward."""
    if guid_cfg.use_cfg:
        return partial(forward_with_cfg, model), dict(
            cfg_scale=guid_cfg.cfg.scale,
            cfg_interval=(guid_cfg.cfg.t_min, guid_cfg.cfg.t_max),
        )
    return model.forward, dict()
