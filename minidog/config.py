"""Config dataclasses. Load with OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(path)))."""
from __future__ import annotations

from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ModelConfig:
    """Generic model configuration for instantiate_from_config().
    Used for stage_1 (VAE) and stage_2 (DiT) model definitions.
    The params dict is passed as kwargs to the target class constructor.
    """
    target: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    ckpt: Optional[str] = None


@dataclass
class MiscConfig:
    """Miscellaneous model-related parameters."""
    latent_size: List[int] = field(default_factory=lambda: [768, 16, 16])  # [C, H, W]
    num_classes: int = 1000
    time_dist_shift_dim: int = 196608  # 16*16*768
    time_dist_shift_base: int = 4096
    time_dist_shift_base_eval: Optional[int] = None


@dataclass
class OptimizerConfig:
    """Optimizer configuration (shared across all training)."""
    type: str = "adamw"
    lr: float = 2.0e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    """LR scheduler configuration."""
    type: str = "cosine"  # "cosine" or "linear"
    warmup_epochs: float = 1.0
    warmup_steps: Optional[int] = None
    warmup_from_zero: bool = True
    decay_end_epoch: float = 16.0
    decay_end_steps: Optional[int] = None
    base_lr: float = 2.0e-4
    final_lr: float = 2.0e-5


@dataclass
class DatasetConfig:
    """Dataset configuration (shared across all training)."""
    target: str = "imagenet"
    type: str = "hf"  # ["hf", "wds"]
    data_dir: str = "./data"
    split: Any = "train"
    condition_type: Optional[str] = None  # "label" or "text"
    shared_tmpdir: str = "~/tmp"
    # WDS-specific
    shuffle_buffer: int = 10000
    seed: int = 42


@dataclass
class EvalConfig:
    """Evaluation configuration.
    eval.datasets.{name}.reference_npz, eval.datasets.{name}.metrics
    """
    eval_interval: int = 5000
    eval_model: bool = False  # Eval non-EMA model too
    eval_dir: str = MISSING  # directory for eval CSVs, e.g. "experiments/<user>/evals/stage2"
    datasets: Optional[Dict[str, Any]] = None


@dataclass
class TrainingConfig:
    """Base training configuration (shared across all)."""
    epochs: int = 16
    batch_size: int = 32
    global_batch_size: Optional[int] = None
    num_workers: int = 4
    global_seed: int = 0
    ema_decay: float = 0.9995
    clip_grad: Optional[float] = None
    log_interval: int = 100
    checkpoint_interval: int = 4
    sample_every: int = 2500
    virtual_epoch_steps: Optional[int] = None
    grad_accum_steps: int = 1
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: Optional[SchedulerConfig] = None
    image_size: int = 256


@dataclass
class TransportConfig:
    """Transport configuration for flow matching."""
    prediction: str = "velocity"  # "velocity" or "x"
    time_dist_type: str = "logit-normal_0_1"
    t_eps: float = 0.05


@dataclass
class SamplerConfig:
    """Sampler configuration for ODE Euler flow matching."""
    num_steps: int = 50


@dataclass
class CFGConfig:
    """CFG configuration for test-time guidance."""
    scale: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0


@dataclass
class GuidanceConfig:
    """Classifier-free guidance configuration for sampling."""
    cfg: Optional[CFGConfig] = field(default_factory=CFGConfig)

    @property
    def use_cfg(self):
        return self.cfg is not None and self.cfg.scale > 1.0


@dataclass
class RepaConfig:
    """REPA loss: align an intermediate DiT layer to frozen DINO patch features."""
    use_repa: bool = False
    repa_layer_depth: int = 8
    repa_coeff: float = 0.5
    repa_loss_type: str = "mse"  # "mse" (this repo's default) or "cosine" (REPA/iREPA papers)
    z_dim: Optional[int] = None  # set from the target encoder's embed_dim in train.py
    irepa: bool = False  # iREPA (arXiv:2512.10794): swap the linear projector for a
    # 3x3 conv over the patch grid, and spatially normalise the teacher features.
    # Their Algorithm 1 is: nn.Conv2d(D_in, D_out, kernel_size=3, padding=1), then
    #   x = x - gamma * x.mean(dim=1, keepdim=True)
    #   x = x / (x.std(dim=1, keepdim=True) + 1e-6)
    # with the statistics taken over the SPATIAL dimension, not the channels.
    irepa_gamma: float = 0.7  # the gamma in the centring term above; the paper's
    # Appendix G reports using gamma in [0.6, 0.8], i.e. the channel mean is only
    # partly removed -- that mean carries the global semantic component they argue
    # is NOT what transfers, so leaving some of it in is deliberate.
    encoder: str = "dinov2"  # "dinov2" (ViT-B/14), "dinov3" (ViT-B/16),
    # "pe_spatial" (PE-Spatial-B16) or "eupe" (EUPE ViT-B/16)
    dinov3_model: Optional[str] = None  # HF id, used when encoder == "dinov3";
    # None -> minidog.dinov3.DEFAULT_DINOV3_MODEL
    pe_spatial_model: Optional[str] = None  # PE config name, used when encoder == "pe_spatial";
    # None -> minidog.pe_spatial.DEFAULT_PE_SPATIAL_MODEL
    pe_spatial_layernorm: bool = True  # parameter-free LayerNorm on PE patch tokens;
    # PE-Spatial has no ln_post, and raw features are ~20x the DINOv2 scale
    eupe_model: Optional[str] = None  # hub name, used when encoder == "eupe";
    # None -> minidog.eupe.DEFAULT_EUPE_MODEL
    eupe_strip_norm_affine: bool = True  # drop the final norm's learned scale/shift,
    # matching the DINOv2/DINOv3 default


@dataclass
class ConditioningArchConfig:
    """In-context conditioning architecture configuration."""
    num_t_tokens: int = 4
    num_c_tokens: int = 8


@dataclass
class TextEncoderConfig:
    """Text encoder configuration."""
    model_name: str = "Qwen/Qwen3-0.6B"
    max_length: int = 256


@dataclass
class ConditioningConfig:
    """Text conditioning configuration."""
    type: str = "text"
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    cfg_dropout_prob: float = 0.1
    context_dim: Optional[int] = None  # initialized later in train.py
    arch: ConditioningArchConfig = field(default_factory=ConditioningArchConfig)


@dataclass
class Stage2Config:
    """Top-level configuration for Stage 2 training."""
    stage_1: ModelConfig = field(default_factory=ModelConfig)
    stage_2: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    repa: RepaConfig = field(default_factory=RepaConfig)
    misc: MiscConfig = field(default_factory=MiscConfig)
    eval: Optional[EvalConfig] = None

    def post_process(self):
        """Post-process the config to set certain runtime fields."""
        self.conditioning.arch.num_c_tokens = self.conditioning.text_encoder.max_length

    def prepare_model_params(self):
        """Populate stage_2.params from typed config fields for model construction.

        Call this after setting runtime fields (conditioning.text_feature_dim,
        conditioning.context_dim, repa.z_dim) and before instantiating the model.
        Uses setdefault for static values so YAML-specified params are never overwritten.
        """
        params = self.stage_2.params

        params.setdefault('context_dim', self.conditioning.context_dim)
        params.setdefault('cond_arch', self.conditioning.arch)

        if self.repa.use_repa:
            params.setdefault('enable_repa', True)
            params.setdefault('repa_layer_depth', self.repa.repa_layer_depth)
            params.setdefault('irepa', self.repa.irepa)
            if self.repa.z_dim is not None:
                params.setdefault('z_dim', self.repa.z_dim)
