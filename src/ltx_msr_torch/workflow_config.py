from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    checkpoint: str = "ltx-2.3-22b-distilled-1.1.safetensors"
    text_encoder: str = "gemma_3_12B_it.safetensors"
    lora: str = "LTX-2.3/LTX-2.3-Licon-MSR-V1.safetensors"
    lora_strength: float = 1.0


@dataclass(frozen=True)
class ReferenceConfig:
    width: int = 1920
    height: int = 1280
    frame_count: int = 41


@dataclass(frozen=True)
class LatentConfig:
    width: int = 1280
    height: int = 1920
    video_frames: int = 145
    audio_frames: int = 241
    frame_rate: int = 24
    batch_size: int = 1


@dataclass(frozen=True)
class SamplingConfig:
    seed: int = 337096718960207
    sampler: str = "euler"
    cfg: float = 1.0
    sigmas: tuple[float, ...] = (
        1.0,
        0.99375,
        0.9875,
        0.98125,
        0.975,
        0.909375,
        0.725,
        0.421875,
        0.0,
    )


@dataclass(frozen=True)
class NagConfig:
    scale: float = 11.0
    alpha: float = 0.25
    tau: float = 2.5
    inplace: bool = True


@dataclass(frozen=True)
class IcLoraGuideConfig:
    frame_idx: int = 0
    strength: float = 1.0
    latent_downscale_factor: float = 1.0
    crop: str = "center"
    use_tiled_encode: bool = False
    tile_size: int = 256
    tile_overlap: int = 64


@dataclass(frozen=True)
class PromptConfig:
    global_prompt: str = ""
    local_prompts: str = ""
    negative_prompt: str = (
        "subtitles, watermark, worst quality, blurry, jittery, distorted, "
        "inconsistent appearance"
    )
    segment_lengths: str = ""
    epsilon: float = 0.0022


@dataclass(frozen=True)
class OutputConfig:
    fps: int = 24
    filename_prefix: str = "LTX-2/MSR_verify_iclora_guide"
    format: str = "auto"
    codec: str = "auto"


@dataclass(frozen=True)
class WorkflowConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    latent: LatentConfig = field(default_factory=LatentConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    nag: NagConfig = field(default_factory=NagConfig)
    ic_lora_guide: IcLoraGuideConfig = field(default_factory=IcLoraGuideConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def default_workflow_config() -> WorkflowConfig:
    return WorkflowConfig()

