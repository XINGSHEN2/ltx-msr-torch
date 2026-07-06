# Conversion Plan

The target is a standalone PyTorch inference project that reproduces the
ComfyUI LTX 2.3 MSR workflow.

## Stage 1: Extract Stable Inputs

Status: started.

- `LiconMSR` reference-video construction is local code.
- Workflow settings can be extracted from the ComfyUI JSON.
- The generated reference tensor follows ComfyUI image tensor convention:
  `[frames, height, width, channels]`, `float32`, range `[0, 1]`.

## Stage 2: Compatibility Runner

Status: started.

Add a runner that calls ComfyUI node classes directly from this project:

- `LowVRAMCheckpointLoader`
- `LTXAVTextEncoderLoader`
- `LTXICLoRALoaderModelOnly`
- `PromptRelayEncode`
- `CLIPTextEncode`
- `EmptyLTXVLatentVideo`
- `LTXVEmptyLatentAudio`
- `LTXAddVideoICLoRAGuide`
- `LTX2_NAG`
- `CFGGuider`
- `SamplerCustomAdvanced`
- `VAEDecode`
- `CreateVideo` / `SaveVideo`

This is the parity baseline.

The first compatibility step uses ComfyUI API prompt JSON:

```bash
PYTHONPATH=src /home/xingshen/ComfyUI/.venv/bin/python -m ltx_msr_torch build-api-prompt \
  --case-dir /mnt/AINAS0/user/xingshen/LTX-2.3-Multiple-Subject-Reference/examples-hf/validition_v1/01 \
  --output outputs/validition_v1_01_api_prompt.json \
  --output-prefix LTX-2/MSR_torch_parity_01
```

If a ComfyUI server is already running, submit it with:

```bash
PYTHONPATH=src /home/xingshen/ComfyUI/.venv/bin/python -m ltx_msr_torch submit-api-prompt \
  --prompt outputs/validition_v1_01_api_prompt.json \
  --server 127.0.0.1:8188
```

## Stage 3: Replace Nodes With Local Code

Status: started.

Replace one node family at a time and compare intermediate outputs:

1. Local low-level nodes:
   `INTConstant`, `ManualSigmas`, `RandomNoise`, `EmptyLTXVLatentVideo`, and
   `LTXVEmptyLatentAudio`.
2. Metadata-only `LTXICLoRALoaderModelOnly` inspection:
   LoRA path resolution and `reference_downscale_factor` extraction.
3. `LTXVConditioning` conditioning metadata update.
4. Model path resolution for checkpoint, text encoder, LoRA, and audio VAE
   checkpoint files.
5. Safetensors header inspection for checkpoint, text encoder, and LoRA.
6. IC-LoRA guide shape planning, frame index resolution, causal fix, and
   token-count parity before model-dependent VAE encoding.
7. PromptRelay local prompt splitting, segment length distribution, token range
   mapping, and relay segment math before model patching.
8. Checkpoint loading.
9. LoRA weight application.
10. PromptRelay text encoder conditioning and model patching.
11. IC-LoRA guide VAE encode and conditioning injection.
12. NAG model patch.
13. Euler sampler with manual sigmas.
14. VAE decode and video/audio mux.

## Parity Rule

For every replaced module, compare:

- tensor shape
- dtype
- value range
- key metadata fields
- fixed-seed output drift

Do not replace multiple model-affecting nodes in one step.
