# ltx-msr-torch

Standalone PyTorch-oriented reconstruction of the ComfyUI LTX 2.3 MSR workflow.

This project is being converted in stages. It now contains local PyTorch
replacements for the workflow tensor preparation, text conditioning,
LTXAV model wiring, IC-LoRA guide injection, sampling, VAE decode, and smoke
video writing paths. ComfyUI remains useful as the parity reference and for
API-prompt comparison, but the main reconstruction code runs locally in torch.

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
- Remaining: full-size end-to-end generation still needs practical GPU runtime
  validation.

The intended path is:

1. Keep comparing parity-critical tensor shapes and metadata against ComfyUI.
2. Run small torch smoke tests with real checkpoint sections after each module
   replacement.
3. Scale the same torch path to the full validation case once the runtime
   budget and GPU memory are available.

## Usage

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

To submit that project-local sample to ComfyUI, expose the project under the
ComfyUI input folder first:

```bash
ln -sfn /home/xingshen/yiwu/ltx-msr-torch /home/xingshen/ComfyUI/input/ltx-msr-torch

python -m ltx_msr_torch build-api-prompt \
  --case-dir /home/xingshen/ComfyUI/input/ltx-msr-torch/sample_cases/validition_v1_01 \
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

Run a minimal real-weight torch sampling smoke and write the decoded video with
audio:

```bash
PYTHONPATH=src /home/xingshen/ComfyUI/.venv/bin/python -m ltx_msr_torch \
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

## Parity Notes

The source ComfyUI workflow uses:

- checkpoint: `ltx-2.3-22b-distilled-1.1.safetensors`
- text encoder: `gemma_3_12B_it.safetensors`
- LoRA: `LTX-2.3\LTX-2.3-Licon-MSR-V1.safetensors`
- sampler: `euler`
- CFG: `1`
- NAG: scale `11`, alpha `0.25`, tau `2.5`, inplace `true`
- IC-LoRA guide: frame index `0`, strength `1`, latent downscale `1`, crop `center`
- sigmas: `1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0`
