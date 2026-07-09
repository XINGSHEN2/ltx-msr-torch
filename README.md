# ltx-msr-torch

Standalone PyTorch-oriented reconstruction of the ComfyUI LTX 2.3 MSR workflow.

This project is being converted in stages. It now contains local PyTorch
replacements for the workflow tensor preparation, text conditioning,
LTXAV model wiring, IC-LoRA guide injection, sampling, VAE decode, and smoke
video writing paths. ComfyUI remains useful as the parity reference and for
API-prompt comparison, but the main reconstruction code runs locally in torch.

## Provenance And Licensing

This project was reconstructed from the behavior of a ComfyUI MSR workflow,
including the LTX 2.3 Multiple Subject Reference graph represented in
`sample_cases/LTX-2.3_MSR_sample_workflow_V2.json`. The goal is to provide a
standalone torch implementation that follows the same workflow semantics,
parameters, tensor preparation, conditioning, IC-LoRA guide handling, sampling,
and decode path without requiring ComfyUI at runtime.

The implementation is not intended to vendor ComfyUI, custom node, model, LoRA,
or text encoder weights. Those external components remain subject to their own
licenses and usage terms. In particular, users should review the licenses for
the referenced ComfyUI/custom-node projects and the LTX-2.3, Gemma text encoder,
and LTX-2.3-Licon-MSR-V1 model assets before redistribution or commercial use.

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
- Validated: the bundled `validition_v1_01` workflow case runs end to end in
  torch, and the first denoising step matches the ComfyUI reference dump
  bit-for-bit after aligning the ComfyUI DynamicVRAM/LowVramPatch LoRA path.

The remaining development path is:

1. Keep comparing parity-critical tensor shapes and metadata against ComfyUI.
2. Keep the debug/parity tools available for future workflow changes.
3. Reduce assumptions that are currently tied to a local ComfyUI checkout, such
   as the Gemma config file location.

## Usage

The command examples below use the ComfyUI virtualenv because this project was
developed against the same torch/AV stack. If you use a separate virtualenv,
install equivalent dependencies and replace `/home/xingshen/ComfyUI/.venv/bin/python`
with that interpreter. From a source checkout, run commands with `PYTHONPATH=src`;
if the package is installed with `pip install -e .`, `PYTHONPATH=src` can be
omitted.

### Model Weights

By default the code resolves model files from the standard local ComfyUI model
layout:

```text
/home/xingshen/ComfyUI/models/checkpoints/ltx-2.3-22b-distilled-1.1.safetensors
/home/xingshen/ComfyUI/models/text_encoders/gemma_3_12B_it.safetensors
/home/xingshen/ComfyUI/models/loras/LTX-2.3/LTX-2.3-Licon-MSR-V1.safetensors
```

The LTX checkpoint contains the transformer, video VAE, and audio VAE sections
used by this reconstruction, so no separate VAE checkpoint is needed for the
bundled workflow.

Download the matching files with resumable `curl` downloads:

```bash
mkdir -p /home/xingshen/ComfyUI/models/checkpoints
mkdir -p /home/xingshen/ComfyUI/models/text_encoders
mkdir -p /home/xingshen/ComfyUI/models/loras/LTX-2.3

curl -L --fail -C - \
  -o /home/xingshen/ComfyUI/models/checkpoints/ltx-2.3-22b-distilled-1.1.safetensors \
  "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-1.1.safetensors?download=true"

curl -L --fail -C - \
  -o /home/xingshen/ComfyUI/models/text_encoders/gemma_3_12B_it.safetensors \
  "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it.safetensors?download=true"

curl -L --fail -C - \
  -o /home/xingshen/ComfyUI/models/loras/LTX-2.3/LTX-2.3-Licon-MSR-V1.safetensors \
  "https://huggingface.co/LiconStudio/LTX-2.3-Multiple-Subject-Reference/resolve/main/LTX-2.3-Licon-MSR-V1.safetensors?download=true"
```

If the weights already exist elsewhere, symlinks are fine:

```bash
ln -sfn /path/to/ltx-2.3-22b-distilled-1.1.safetensors \
  /home/xingshen/ComfyUI/models/checkpoints/ltx-2.3-22b-distilled-1.1.safetensors
ln -sfn /path/to/gemma_3_12B_it.safetensors \
  /home/xingshen/ComfyUI/models/text_encoders/gemma_3_12B_it.safetensors
ln -sfn /path/to/LTX-2.3-Licon-MSR-V1.safetensors \
  /home/xingshen/ComfyUI/models/loras/LTX-2.3/LTX-2.3-Licon-MSR-V1.safetensors
```

The Gemma tokenizer/config files are currently read from the ComfyUI-LTXVideo
custom node config directory:

```text
/home/xingshen/ComfyUI/custom_nodes/ComfyUI-LTXVideo/gemma_configs
```

That directory must contain `gemma3cfg.json`, `tokenizer.json`,
`tokenizer.model`, and `tokenizer_config.json`.

Verify that all required files resolve:

```bash
PYTHONPATH=src /home/xingshen/ComfyUI/.venv/bin/python -m ltx_msr_torch \
  inspect-model-headers

PYTHONPATH=src /home/xingshen/ComfyUI/.venv/bin/python -m ltx_msr_torch \
  inspect-text-encoder
```

### Running The Workflow

Run the bundled `validition_v1/01` case through the pure torch MSR path:

```bash
PYTHONPATH=src /home/xingshen/ComfyUI/.venv/bin/python -m ltx_msr_torch \
  generate-msr-case \
  --workflow sample_cases/LTX-2.3_MSR_sample_workflow_V2.json \
  --case-dir sample_cases/validition_v1_01 \
  --output-video outputs/msr_validation_v1_01_workflow_exact_lowvram_lora.mp4 \
  --dtype bf16 \
  --device cuda
```

For a quick wiring check, use fewer layers and only the first sampler step:

```bash
PYTHONPATH=src /home/xingshen/ComfyUI/.venv/bin/python -m ltx_msr_torch \
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

Build the bundled validation case inputs locally and encode the IC-LoRA guide
with the real VideoVAE:

```bash
PYTHONPATH=src /home/xingshen/ComfyUI/.venv/bin/python -m ltx_msr_torch \
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

The same command accepts prompt/image overrides. For example, to run the bundled
case with a custom output path:

```bash
PYTHONPATH=src /home/xingshen/ComfyUI/.venv/bin/python -m ltx_msr_torch \
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
- LoRA: `LTX-2.3\LTX-2.3-Licon-MSR-V1.safetensors`
- sampler: `euler`
- CFG: `1`
- NAG: scale `11`, alpha `0.25`, tau `2.5`, inplace `true`
- IC-LoRA guide: frame index `0`, strength `1`, latent downscale `1`, crop `center`
- sigmas: `1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0`
