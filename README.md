# ltx-msr-torch

Standalone PyTorch-oriented reconstruction of the ComfyUI LTX 2.3 MSR workflow.

This project is being converted in stages. The first stage extracts the stable,
non-model parts of the workflow into normal Python code and records the exact
ComfyUI node parameters needed for parity.

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
- Pending: direct LTX 2.3 model load, PromptRelay conditioning, IC-LoRA guide,
  NAG patch, sampler, VAE decode, and mp4 writing.

The intended path is:

1. Build and verify tensors that are independent of ComfyUI.
2. Add a compatibility runner that calls the existing ComfyUI node classes.
3. Replace each compatibility call with local PyTorch code, one module at a time.

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

Inspect safetensors headers without loading full model weights:

```bash
python -m ltx_msr_torch inspect-model-headers
```

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
