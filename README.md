# ltx-msr-torch

[English](README.md) | [简体中文](README.zh-CN.md)

Standalone PyTorch-oriented reconstruction of the ComfyUI LTX 2.3 MSR workflow.

This project is being converted in stages. It now contains local PyTorch
replacements for the workflow tensor preparation, text conditioning,
LTXAV model wiring, IC-LoRA guide injection, sampling, standalone video/audio
VAE, vocoder, and smoke video writing paths. ComfyUI remains useful as an
optional parity reference and for API-prompt comparison, but it is not required
by the generation runtime.

## Provenance And Licensing

This project was reconstructed from the behavior of a ComfyUI MSR workflow,
including the LTX 2.3 Multiple Subject Reference graph represented in
`sample_cases/LTX-2.3_MSR_sample_workflow_V2.json`. The goal is to provide a
standalone torch implementation that follows the same workflow semantics,
parameters, tensor preparation, conditioning, IC-LoRA guide handling, sampling,
and decode path without requiring ComfyUI at runtime.

The standalone VAE and vocoder modules include code adapted from ComfyUI commit
`dd17debce517f8818ae9910b437cb1ebaa673176` under GPL-3.0. Source paths and
modification notices are retained in each derived file; see `LICENSE` and
`THIRD_PARTY_NOTICES.md`. The project does not include model, LoRA, or text
encoder weights. Those assets remain subject to their own licenses and usage
terms.

The `tools/` directory contains parity/debug helpers that can compare this torch
path against ComfyUI during development; those helpers are separate from the
standalone torch runtime.

## Current Status

- Implemented: `LiconMSR` reference-video construction.
- Implemented: local PyTorch replacements for low-level workflow nodes:
  `INTConstant`, `ManualSigmas`, `RandomNoise`, `EmptyLTXVLatentVideo`, and
  `LTXVEmptyLatentAudio`.
- Implemented: local conditioning metadata replacement for `LTXVConditioning`.
- Implemented: metadata-only local inspection for `LTXICLoRALoaderModelOnly`
  with ComfyUI-style LoRA path resolution and `reference_downscale_factor`
  extraction.
- Implemented: local ComfyUI-style path resolution for checkpoint, text
  encoder, LoRA, and audio VAE checkpoint files.
- Implemented: workflow parity config for the sampled ComfyUI graph.
- Implemented: ComfyUI UI-workflow to API-prompt conversion for MSR case tests.
- Implemented: Gemma tokenizer/text model loading, text projection, PromptRelay
  token planning, and LTXAV text embedding connectors.
- Implemented: LTXAV transformer construction, streaming checkpoint load,
  LoRA application, LTXAV input/output projection, timestep/rope preparation,
  tuple Euler sampling, and VAE video/audio decode.
- Implemented: IC-LoRA video guide planning, real VideoVAE guide encode,
  keyframe/guide attention metadata injection, and decoded mp4 smoke output
  with AAC audio muxing.
- Implemented: standalone video VAE, audio VAE, audio patchifier, and vocoder;
  generation no longer imports any ComfyUI runtime module.
- Validated: the bundled `validition_v1_01` workflow case runs end to end in
  torch, and the first denoising step matches the ComfyUI reference dump
  bit-for-bit after aligning the ComfyUI DynamicVRAM/LowVramPatch LoRA path.

The remaining development path is:

1. Keep comparing parity-critical tensor shapes and metadata against ComfyUI.
2. Keep the debug/parity tools available for future workflow changes.
3. Keep validating standalone VAE and vocoder parity as upstream code evolves.

## Usage

### Quick Start

The following steps deploy a fresh checkout on Linux with Python 3.10 or newer.
An NVIDIA GPU with sufficient VRAM is required for the full 22B workflow, and
`ffmpeg` must be available on `PATH` to write video with audio.

```bash
git clone https://github.com/XINGSHEN2/ltx-msr-torch.git
cd ltx-msr-torch

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

No ComfyUI checkout, custom nodes, or ComfyUI server is required for generation.
ComfyUI is needed only when running the optional parity/debug helpers that
explicitly submit a prompt to a ComfyUI server.

### Model Weights

The default model root is `models/` in this repository. Download these three
externally distributed weights:

For a fresh deployment, the bundled resumable downloader installs all three
weights and the required Gemma tokenizer/configuration files:

```bash
bash scripts/download_models.sh
```

Set `LTX_MSR_MODEL_ROOT=/path/to/models` before running the script to store the
files on a separate model disk. `HF_ENDPOINT=https://hf-mirror.com` is also
supported.

| Required asset | Destination | Official source |
| --- | --- | --- |
| Integrated LTX-2.3 checkpoint | `models/checkpoints/ltx-2.3-22b-distilled-1.1.safetensors` | [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-22b-distilled-1.1.safetensors) |
| Gemma 3 12B text encoder | `models/text_encoders/gemma_3_12B_it.safetensors` | [Comfy-Org/ltx-2](https://huggingface.co/Comfy-Org/ltx-2/blob/main/split_files/text_encoders/gemma_3_12B_it.safetensors) |
| LTX-2.3 MSR LoRA | `models/loras/LTX-2.3/LTX-2.3-Licon-MSR-V1.safetensors` | [LiconStudio/LTX-2.3-Multiple-Subject-Reference](https://huggingface.co/LiconStudio/LTX-2.3-Multiple-Subject-Reference/blob/main/LTX-2.3-Licon-MSR-V1.safetensors) |

These commands use resumable downloads. Set `HF_ENDPOINT=https://hf-mirror.com`
before running them when the standard Hugging Face endpoint is unavailable.

```bash
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
mkdir -p models/checkpoints models/text_encoders models/loras/LTX-2.3

curl -L --fail -C - \
  -o models/checkpoints/ltx-2.3-22b-distilled-1.1.safetensors \
  "$HF_ENDPOINT/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-1.1.safetensors?download=true"

curl -L --fail -C - \
  -o models/text_encoders/gemma_3_12B_it.safetensors \
  "$HF_ENDPOINT/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it.safetensors?download=true"

curl -L --fail -C - \
  -o models/loras/LTX-2.3/LTX-2.3-Licon-MSR-V1.safetensors \
  "$HF_ENDPOINT/LiconStudio/LTX-2.3-Multiple-Subject-Reference/resolve/main/LTX-2.3-Licon-MSR-V1.safetensors?download=true"
```

The download script also installs `gemma3cfg.json`, `tokenizer.json`,
`tokenizer.model`, and `tokenizer_config.json` from a pinned upstream
ComfyUI-LTXVideo revision. No additional repository clone is required.

To reuse an existing model library instead, do not create these paths. Point
the application at directories with the same category layout:

```bash
export LTX_MSR_MODEL_ROOT=/path/to/models
export LTX_MSR_GEMMA_CONFIG_DIR=/path/to/gemma_configs
```

The integrated LTX checkpoint already contains the transformer, text
projection, video VAE, audio VAE/vocoder, and embedding connector weights.
Consequently, this project does **not** require separate downloads for
`ltx-2.3-22b-dev_transformer_only_fp8_scaled.safetensors`,
`ltx-2.3_text_projection_bf16.safetensors`, a VAE, or a vocoder. It also does
not use `gemma_3_12B_it_fp4_mixed.safetensors` or
`ltx-2.3-22b-dev.safetensors`. Those two names appear only in the sample
workflow's model-manager metadata; the effective workflow selections and the
Python defaults use the three weights listed above.

Verify that all required files resolve:

```bash
python scripts/verify_environment.py
```

The script checks Python and package imports, `ffmpeg`, all three model headers,
Gemma tokenizer files, and confirms that the bundled video and audio VAEs import
without a ComfyUI runtime. CUDA/NVIDIA is not inspected or required by default,
so the same preflight works on CPU-only servers. For a CUDA deployment, CUDA/BF16
can be required explicitly and a small test video can be generated:

```bash
python scripts/verify_environment.py --require-cuda --smoke --device cuda
```

When models are stored outside this checkout, either export
`LTX_MSR_MODEL_ROOT` and `LTX_MSR_GEMMA_CONFIG_DIR`, or pass them directly:

```bash
python scripts/verify_environment.py \
  --model-root /path/to/models \
  --gemma-config-dir /path/to/gemma_configs
```

### Running The Workflow

Run the bundled `validition_v1/01` case through the pure torch MSR path:

```bash
python -m ltx_msr_torch \
  generate-msr-case \
  --workflow sample_cases/LTX-2.3_MSR_sample_workflow_V2.json \
  --case-dir sample_cases/validition_v1_01 \
  --output-video outputs/msr_validation_v1_01_workflow_exact_lowvram_lora.mp4 \
  --dtype bf16 \
  --device cuda
```

To submit the same `validation_v1/01` case to a running standalone LTX MSR
service and download the result into a chosen directory:

```bash
bash scripts/submit_validation_v1_service.sh \
  --server http://127.0.0.1:9004 \
  --output-dir /path/to/output
```

The script reads the global/local prompts embedded in the workflow, preserves
the original PromptRelay, dimensions, frame counts, seed, negative prompt, and
full sampling settings, then downloads the MP4 after the task completes.

The companion `mx-services/ltx_msr` launcher supports a split-device resident
runtime: Gemma and the text connectors stay on one device, while LTX 22B, the
LoRA, and both VAEs stay on another. Its long-lived worker reuses
`PersistentMSRRuntime` instead of loading weights for every task.

For a quick wiring check, use fewer layers and only the first sampler step:

```bash
python -m ltx_msr_torch \
  generate-msr-case \
  --workflow sample_cases/LTX-2.3_MSR_sample_workflow_V2.json \
  --case-dir sample_cases/validition_v1_01 \
  --output-video outputs/msr_case_01_smoke.mp4 \
  --layers 1 \
  --max-sigmas 2 \
  --dtype bf16 \
  --device cuda
```

Create an MSR reference tensor from up to four subject images plus a background:

```bash
python -m ltx_msr_torch build-reference \
  --subject-1 /path/to/1.png \
  --subject-2 /path/to/2.png \
  --background /path/to/background.png \
  --output /tmp/msr_reference.pt
```

The saved tensor matches ComfyUI image tensor convention:

```text
[frames, height, width, channels], float32, range [0, 1]
```

Default dimensions and frame count match the inspected workflow:

```text
width=1920, height=1280, frame_count=41
```

Build a ComfyUI API prompt for a downloaded MSR validation case:

```bash
python -m ltx_msr_torch build-api-prompt \
  --case-dir sample_cases/validition_v1_01 \
  --output outputs/validition_v1_01_api_prompt.json \
  --output-prefix LTX-2/MSR_torch_parity_01
```

The project includes this small input case under
`sample_cases/validition_v1_01`.

The following optional development-only flow requires a separate ComfyUI
checkout and running ComfyUI server. To submit the project-local sample, expose
the project under the ComfyUI input folder first:

```bash
ln -sfn "$PWD" "$COMFYUI_ROOT/input/ltx-msr-torch"

python -m ltx_msr_torch build-api-prompt \
  --case-dir "$COMFYUI_ROOT/input/ltx-msr-torch/sample_cases/validition_v1_01" \
  --output outputs/project_sample_validition_v1_01_api_prompt.json \
  --output-prefix LTX-2/MSR_project_sample_01

python -m ltx_msr_torch submit-api-prompt \
  --prompt outputs/project_sample_validition_v1_01_api_prompt.json \
  --server 127.0.0.1:8188 \
  --wait
```

The local ComfyUI client bypasses environment HTTP proxies for `127.0.0.1`.

Inspect the local torch replacements and resolved workflow parameters:

```bash
python -m ltx_msr_torch inspect-local-state
```

This includes the IC-LoRA guide frame/index plan, target encode size, and
estimated conditioning token count before the model-dependent VAE encode.

Inspect safetensors headers without loading full model weights:

```bash
python -m ltx_msr_torch inspect-model-headers
```

Inspect checkpoint section key counts:

```bash
python -m ltx_msr_torch inspect-checkpoint
```

Inspect the loaded text embedding projection module:

```bash
python -m ltx_msr_torch inspect-text-projection
```

Inspect video/audio VAE checkpoint sections:

```bash
python -m ltx_msr_torch inspect-vae-sections
```

Inspect Gemma text encoder checkpoint/config sections:

```bash
python -m ltx_msr_torch inspect-text-encoder
```

Gemma tokenizer loading and PromptRelay token range planning are available from
`ltx_msr_torch.gemma_tokenizer`.

```bash
python -m ltx_msr_torch inspect-tokenizer --case-dir sample_cases/validition_v1_01
```

Inspect the workflow LoRA tensor pair manifest:

```bash
python -m ltx_msr_torch inspect-lora-manifest
```

The local LoRA utilities now also include the pure torch `B @ A` delta math and
raw-checkpoint target matching used before applying weights.

PromptRelay's deterministic segment planning is available from
`ltx_msr_torch.prompt_relay`; model patching and text encoder conditioning are
kept as separate replacement steps.

LTX2 NAG guidance math is available from `ltx_msr_torch.nag`, including the
normalized attention guidance formula and workflow patch target planning.

Euler sampler utilities are available from `ltx_msr_torch.sampler`; model
forward integration is intentionally separate from the deterministic step math.

Build the bundled validation case inputs locally and encode the IC-LoRA guide
with the real VideoVAE:

```bash
python -m ltx_msr_torch \
  smoke-case-inputs \
  --case-dir sample_cases/validition_v1_01 \
  --width 64 \
  --height 64 \
  --frame-count 9 \
  --latent-frames 8 \
  --device cpu
```

Run a minimal real-weight torch sampling smoke and write the decoded video with
audio:

```bash
python -m ltx_msr_torch \
  smoke-ltxav-sampling \
  --layers 1 \
  --device cpu \
  --dtype bf16 \
  --apply-lora \
  --decode \
  --output-video outputs/smoke_ltxav_sampling.mp4
```

This smoke intentionally uses one transformer layer and a 1x1 latent grid so it
verifies wiring, LoRA application, video/audio decode, AAC muxing, and mp4
output without attempting the full 22B workflow resolution.

The same command accepts prompt/image overrides. For example, to run the bundled
case with a custom output path:

```bash
python -m ltx_msr_torch \
  generate-msr-case \
  --workflow sample_cases/LTX-2.3_MSR_sample_workflow_V2.json \
  --case-dir sample_cases/validition_v1_01 \
  --output-video outputs/msr_case_01_torch.mp4
```

By default this reads the bundled workflow JSON and uses the workflow's width,
height, frame count, seed, sigma schedule, PromptRelay settings, IC-LoRA guide,
NAG settings, and LoRA strength.

## Parity Notes

The source ComfyUI workflow uses:

- checkpoint: `ltx-2.3-22b-distilled-1.1.safetensors`
- text encoder: `gemma_3_12B_it.safetensors`
- LoRA: `LTX-2.3/LTX-2.3-Licon-MSR-V1.safetensors`
- sampler: `euler`
- CFG: `1`
- NAG: scale `11`, alpha `0.25`, tau `2.5`, inplace `true`
- IC-LoRA guide: frame index `0`, strength `1`, latent downscale `1`, crop `center`
- sigmas: `1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0`
