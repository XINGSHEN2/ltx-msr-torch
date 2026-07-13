#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_root="${LTX_MSR_MODEL_ROOT:-${project_root}/models}"
hf_endpoint="${HF_ENDPOINT:-https://huggingface.co}"
gemma_config_base="${LTX_MSR_GEMMA_CONFIG_BASE_URL:-https://raw.githubusercontent.com/Lightricks/ComfyUI-LTXVideo/aceeae9635f6d493f2893ba3c411a1c36031788a/gemma_configs}"

download() {
  local url="$1"
  local output="$2"
  mkdir -p "$(dirname "$output")"
  curl -L --fail --retry 5 --retry-delay 2 -C - -o "$output" "$url"
}

download \
  "${hf_endpoint}/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-1.1.safetensors?download=true" \
  "${model_root}/checkpoints/ltx-2.3-22b-distilled-1.1.safetensors"

download \
  "${hf_endpoint}/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it.safetensors?download=true" \
  "${model_root}/text_encoders/gemma_3_12B_it.safetensors"

download \
  "${hf_endpoint}/LiconStudio/LTX-2.3-Multiple-Subject-Reference/resolve/main/LTX-2.3-Licon-MSR-V1.safetensors?download=true" \
  "${model_root}/loras/LTX-2.3/LTX-2.3-Licon-MSR-V1.safetensors"

for filename in gemma3cfg.json tokenizer.json tokenizer.model tokenizer_config.json; do
  download "${gemma_config_base}/${filename}" "${model_root}/gemma_configs/${filename}"
done

echo "Models and Gemma configuration files are ready under: ${model_root}"
