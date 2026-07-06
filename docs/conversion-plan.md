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

## Stage 3: Replace Nodes With Local Code

Replace one node family at a time and compare intermediate outputs:

1. Model path resolution and checkpoint loading.
2. LoRA application.
3. PromptRelay text conditioning.
4. Empty video/audio latent creation.
5. IC-LoRA guide encode and conditioning injection.
6. NAG model patch.
7. Euler sampler with manual sigmas.
8. VAE decode and video/audio mux.

## Parity Rule

For every replaced module, compare:

- tensor shape
- dtype
- value range
- key metadata fields
- fixed-seed output drift

Do not replace multiple model-affecting nodes in one step.

