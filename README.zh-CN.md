# ltx-msr-torch

[English](README.md) | [简体中文](README.zh-CN.md)

面向 PyTorch 的 ComfyUI LTX 2.3 MSR 工作流独立重构实现。

本项目正在分阶段转换。目前已经包含工作流张量准备、文本条件、LTXAV
模型连接、IC-LoRA 引导注入、采样、独立视频/音频 VAE、Vocoder 及冒烟视频
写入等环节的本地 PyTorch 实现。ComfyUI 仍可作为一致性参考和 API prompt
对比工具，但生成运行时已经不再依赖 ComfyUI。

## 来源与许可证

本项目依据 ComfyUI MSR 工作流的行为进行重构，其中包括
`sample_cases/LTX-2.3_MSR_sample_workflow_V2.json` 表示的 LTX 2.3
Multiple Subject Reference 工作流图。目标是提供独立的 torch 实现，在不依赖
ComfyUI 运行时的情况下，遵循相同的工作流语义、参数、张量准备、条件处理、
IC-LoRA 引导处理、采样及解码流程。

独立 VAE 和 Vocoder 模块包含依据 GPL-3.0 从 ComfyUI commit
`dd17debce517f8818ae9910b437cb1ebaa673176` 修改的代码。每个衍生文件均保留
来源路径和修改说明，详见 `LICENSE` 与 `THIRD_PARTY_NOTICES.md`。项目不内置
模型、LoRA 或文本编码器权重，这些资源仍分别受其自身许可证和使用条款约束。

`tools/` 目录包含开发期间用于对比本 torch 路径和 ComfyUI 的一致性及调试
工具；这些工具与独立 torch 运行时相互分离。

## 当前状态

- 已实现：`LiconMSR` 参考视频构建。
- 已实现：底层工作流节点的本地 PyTorch 替代，包括 `INTConstant`、
  `ManualSigmas`、`RandomNoise`、`EmptyLTXVLatentVideo` 和
  `LTXVEmptyLatentAudio`。
- 已实现：`LTXVConditioning` 的本地条件元数据替代。
- 已实现：`LTXICLoRALoaderModelOnly` 的本地纯元数据检查，支持 ComfyUI 风格
  的 LoRA 路径解析和 `reference_downscale_factor` 提取。
- 已实现：checkpoint、文本编码器、LoRA 和音频 VAE checkpoint 文件的本地
  ComfyUI 风格路径解析。
- 已实现：样例 ComfyUI 工作流图的一致性配置。
- 已实现：用于 MSR 样例测试的 ComfyUI UI 工作流到 API prompt 转换。
- 已实现：Gemma tokenizer/文本模型加载、文本投影、PromptRelay token 规划和
  LTXAV 文本嵌入连接器。
- 已实现：LTXAV transformer 构建、checkpoint 流式加载、LoRA 应用、LTXAV
  输入/输出投影、timestep/rope 准备、元组 Euler 采样及 VAE 视频/音频解码。
- 已实现：IC-LoRA 视频引导规划、真实 VideoVAE 引导编码、关键帧/引导注意力
  元数据注入，以及带 AAC 音频混流的解码 mp4 冒烟输出。
- 已实现：独立视频 VAE、音频 VAE、音频 Patchifier 与 Vocoder；生成流程不再
  导入任何 ComfyUI 运行时模块。
- 已验证：内置的 `validition_v1_01` 工作流样例可以通过 torch 端到端运行；
  对齐 ComfyUI DynamicVRAM/LowVramPatch LoRA 路径后，第一个去噪步骤与
  ComfyUI 参考 dump 逐 bit 一致。

后续开发方向：

1. 继续对比与一致性有关的关键张量形状和元数据。
2. 保留调试和一致性工具，以支持后续工作流变更。
3. 在上游实现变化时继续验证独立 VAE 与 Vocoder 的一致性。

## 使用方法

### 快速开始

以下步骤用于在 Linux 上部署全新代码库，需要 Python 3.10 或更高版本。完整
22B 工作流需要具有足够显存的 NVIDIA GPU；如需写入带音频的视频，`PATH` 中
还必须存在 `ffmpeg`。

```bash
git clone https://github.com/XINGSHEN2/ltx-msr-torch.git
cd ltx-msr-torch

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

生成时不需要克隆 ComfyUI、不需要安装自定义节点，也不需要启动 ComfyUI
服务。只有显式向 ComfyUI 服务提交 prompt 的可选一致性/调试工具才需要单独的
ComfyUI 环境。

### 模型权重

默认模型根目录是本仓库中的 `models/`。需要下载以下三个外部分发的权重：

全新服务器可以直接运行内置的断点续传下载脚本。脚本会下载三个权重以及所需的
Gemma tokenizer/config 文件：

```bash
bash scripts/download_models.sh
```

如需存放到独立模型盘，请在运行前设置
`LTX_MSR_MODEL_ROOT=/path/to/models`；脚本也支持
`HF_ENDPOINT=https://hf-mirror.com`。

| 必需资源 | 保存位置 | 官方来源 |
| --- | --- | --- |
| LTX-2.3 集成 checkpoint | `models/checkpoints/ltx-2.3-22b-distilled-1.1.safetensors` | [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-22b-distilled-1.1.safetensors) |
| Gemma 3 12B 文本编码器 | `models/text_encoders/gemma_3_12B_it.safetensors` | [Comfy-Org/ltx-2](https://huggingface.co/Comfy-Org/ltx-2/blob/main/split_files/text_encoders/gemma_3_12B_it.safetensors) |
| LTX-2.3 MSR LoRA | `models/loras/LTX-2.3/LTX-2.3-Licon-MSR-V1.safetensors` | [LiconStudio/LTX-2.3-Multiple-Subject-Reference](https://huggingface.co/LiconStudio/LTX-2.3-Multiple-Subject-Reference/blob/main/LTX-2.3-Licon-MSR-V1.safetensors) |

以下命令支持断点续传。如果无法访问 Hugging Face 标准站点，请在运行前设置
`HF_ENDPOINT=https://hf-mirror.com`。

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

下载脚本还会从固定版本的 ComfyUI-LTXVideo 下载 `gemma3cfg.json`、
`tokenizer.json`、`tokenizer.model` 和 `tokenizer_config.json`，无需再克隆其他
代码仓库。

如果需要复用已有模型库，则无需创建上述模型路径。请通过环境变量指向具有相同
分类目录结构的位置：

```bash
export LTX_MSR_MODEL_ROOT=/path/to/models
export LTX_MSR_GEMMA_CONFIG_DIR=/path/to/gemma_configs
```

LTX 集成 checkpoint 已包含 transformer、文本投影、视频 VAE、音频
VAE/vocoder 和嵌入连接器权重。因此本项目**不需要**单独下载
`ltx-2.3-22b-dev_transformer_only_fp8_scaled.safetensors`、
`ltx-2.3_text_projection_bf16.safetensors`、VAE 或 vocoder；也不使用
`gemma_3_12B_it_fp4_mixed.safetensors` 或 `ltx-2.3-22b-dev.safetensors`。
后两个名称只存在于样例工作流的模型管理器元数据中；实际工作流选项和 Python
默认配置使用上表中的三个权重。

验证所有必需文件是否可以解析：

```bash
python scripts/verify_environment.py
```

该脚本会检查 Python 和依赖导入、`ffmpeg`、三个模型文件的 safetensors 头、
Gemma tokenizer 文件，并确认项目内置的视频/音频 VAE 在没有 ComfyUI 运行时的
情况下可以导入。默认不会检查 CUDA/NVIDIA，因此也适用于没有 NVIDIA 显卡的
服务器。如果部署目标明确使用 CUDA，可以显式要求 CUDA/BF16 并生成小尺寸测试视频：

```bash
python scripts/verify_environment.py --require-cuda --smoke --device cuda
```

如果模型不在仓库的 `models` 目录下，可以导出
`LTX_MSR_MODEL_ROOT`、`LTX_MSR_GEMMA_CONFIG_DIR`，也可以直接传参：

```bash
python scripts/verify_environment.py \
  --model-root /path/to/models \
  --gemma-config-dir /path/to/gemma_configs
```

### 运行工作流

通过纯 torch MSR 路径运行内置的 `validition_v1/01` 样例：

```bash
python -m ltx_msr_torch \
  generate-msr-case \
  --workflow sample_cases/LTX-2.3_MSR_sample_workflow_V2.json \
  --case-dir sample_cases/validition_v1_01 \
  --output-video outputs/msr_validation_v1_01_workflow_exact_lowvram_lora.mp4 \
  --dtype bf16 \
  --device cuda
```

通过已启动的独立 LTX MSR 服务提交同一个 `validation_v1/01` 样例，并把成片
下载到指定目录：

```bash
bash scripts/submit_validation_v1_service.sh \
  --server http://127.0.0.1:9004 \
  --output-dir /path/to/output
```

脚本读取工作流内置的 global/local prompt，保持原始 PromptRelay、尺寸、帧数、
seed、negative prompt 和完整采样配置，并在任务完成后下载 MP4。

配套 `mx-services/ltx_msr` 支持双卡常驻 Runtime：Gemma 与文本连接器放在一张
卡，LTX 22B、LoRA 和两个 VAE 放在另一张卡。常驻 Worker 直接复用
`PersistentMSRRuntime`，不会为每个任务重新加载权重。

如需快速检查模型连接，可以减少层数，并且只运行第一个采样步骤：

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

使用最多四张主体图片和一张背景图片创建 MSR 参考张量：

```bash
python -m ltx_msr_torch build-reference \
  --subject-1 /path/to/1.png \
  --subject-2 /path/to/2.png \
  --background /path/to/background.png \
  --output /tmp/msr_reference.pt
```

保存的张量遵循 ComfyUI 图像张量规范：

```text
[frames, height, width, channels], float32, range [0, 1]
```

默认尺寸和帧数与已检查的工作流一致：

```text
width=1920, height=1280, frame_count=41
```

为已下载的 MSR 验证样例构建 ComfyUI API prompt：

```bash
python -m ltx_msr_torch build-api-prompt \
  --case-dir sample_cases/validition_v1_01 \
  --output outputs/validition_v1_01_api_prompt.json \
  --output-prefix LTX-2/MSR_torch_parity_01
```

项目在 `sample_cases/validition_v1_01` 中内置了这个小型输入样例。

下面是仅用于开发调试的可选流程，需要另行准备并启动 ComfyUI。要把项目内
样例提交给 ComfyUI，先在 ComfyUI 输入目录中暴露本项目：

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

本地 ComfyUI 客户端会绕过 `127.0.0.1` 的环境 HTTP 代理。

检查本地 torch 替代实现以及解析后的工作流参数：

```bash
python -m ltx_msr_torch inspect-local-state
```

输出包括 IC-LoRA 引导帧/索引规划、目标编码尺寸，以及依赖模型的 VAE 编码前
估算的条件 token 数量。

在不加载完整模型权重的情况下检查 safetensors header：

```bash
python -m ltx_msr_torch inspect-model-headers
```

检查 checkpoint 各分区的 key 数量：

```bash
python -m ltx_msr_torch inspect-checkpoint
```

检查已加载的文本嵌入投影模块：

```bash
python -m ltx_msr_torch inspect-text-projection
```

检查视频/音频 VAE checkpoint 分区：

```bash
python -m ltx_msr_torch inspect-vae-sections
```

检查 Gemma 文本编码器 checkpoint 和配置分区：

```bash
python -m ltx_msr_torch inspect-text-encoder
```

Gemma tokenizer 加载和 PromptRelay token 范围规划由
`ltx_msr_torch.gemma_tokenizer` 提供。

```bash
python -m ltx_msr_torch inspect-tokenizer --case-dir sample_cases/validition_v1_01
```

检查工作流 LoRA 张量对清单：

```bash
python -m ltx_msr_torch inspect-lora-manifest
```

本地 LoRA 工具还包括纯 torch `B @ A` 增量计算，以及应用权重前使用的原始
checkpoint 目标匹配。

PromptRelay 的确定性分段规划由 `ltx_msr_torch.prompt_relay` 提供；模型 patch
和文本编码器条件处理作为独立替代步骤保留。

LTX2 NAG 引导计算由 `ltx_msr_torch.nag` 提供，包括归一化注意力引导公式和
工作流 patch 目标规划。

Euler 采样器工具由 `ltx_msr_torch.sampler` 提供；模型 forward 集成有意与
确定性的采样步骤计算分离。

在本地构建内置验证样例输入，并使用真实 VideoVAE 编码 IC-LoRA 引导：

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

运行使用真实权重的最小 torch 采样冒烟测试，并写入解码后的音视频：

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

该冒烟测试有意只使用一个 transformer 层和 1x1 latent 网格，因此无需尝试
完整 22B 工作流分辨率，即可验证模型连接、LoRA 应用、视频/音频解码、AAC
混流和 mp4 输出。

同一命令也接受 prompt/图像覆盖参数。例如，使用自定义输出路径运行内置样例：

```bash
python -m ltx_msr_torch \
  generate-msr-case \
  --workflow sample_cases/LTX-2.3_MSR_sample_workflow_V2.json \
  --case-dir sample_cases/validition_v1_01 \
  --output-video outputs/msr_case_01_torch.mp4
```

默认情况下，该命令读取内置工作流 JSON，并使用其中的宽度、高度、帧数、随机
种子、sigma 调度、PromptRelay 设置、IC-LoRA 引导、NAG 设置和 LoRA 强度。

## 一致性说明

源 ComfyUI 工作流使用：

- checkpoint：`ltx-2.3-22b-distilled-1.1.safetensors`
- 文本编码器：`gemma_3_12B_it.safetensors`
- LoRA：`LTX-2.3/LTX-2.3-Licon-MSR-V1.safetensors`
- 采样器：`euler`
- CFG：`1`
- NAG：scale `11`、alpha `0.25`、tau `2.5`、inplace `true`
- IC-LoRA 引导：frame index `0`、strength `1`、latent downscale `1`、crop `center`
- sigmas：`1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0`
