

**Horizon-LM:** 


**Single-GPU Training of Hundreds-of-Billions Parameter Language Models with Mixed BF16/FP32 Precision**

## Quick Start

```bash
# Install
git clone https://github.com/DLYuanGod/Horizon-LM.git
cd Horizon-LM
pip install -e .

# Train
python examples/train.py --config examples/configs/qwen_32b.yaml
```

## Features

- ✅ **Single GPU Training**: Train 120B+ models on one GPU
- ✅ **CPU-Backed Parameters**: FP32 master weights on CPU, BF16 working copy on GPU
- ✅ **YAML Configuration**: Easy model/dataset/hyperparameter configuration
- ✅ **Flash Attention**: Memory-efficient attention computation
- ✅ **DeepSpeed CPUAdam**: 5-7x faster optimizer
- ✅ **Auto CUDA Extension**: Automatically builds optimized CUDA kernels

## Supported Models

| Model Family | Model Sizes | Status |
|--------------|-------------|--------|
| **Qwen2/Qwen2.5** | 0.5B/1.5B/3B/7B/14B/32B/72B | ✅ |
| Qwen3 | 0.6B/1.7B/4B/8B/14B/32B/80B/235B | 🔄 Pending |
| Llama 3/3.1/3.2/3.3 | 1B/3B/8B/70B | 🔄 Pending |
| Llama 4 | 109B/402B | 🔄 Pending |
| DeepSeek (LLM/Code/MoE) | 7B/16B/67B/236B | 🔄 Pending |
| DeepSeek 3 | 236B/671B | 🔄 Pending |
| DeepSeek R1 | 1.5B/7B/8B/14B/32B/70B/671B | 🔄 Pending |
| Mistral/Mixtral | 7B/8x7B/8x22B | 🔄 Pending |
| Phi-4 | 3.8B/14B | 🔄 Pending |
| GPT-OSS | 20B/120B | 🔄 Pending |
| GLM-4/GLM-4.5 | 9B/32B/106B/355B | 🔄 Pending |
| InternLM 2/3 | 7B/8B/20B | 🔄 Pending |
| Gemma 2/3 | 2B/7B/9B/27B | 🔄 Pending |
| Yi | 6B/9B/34B | 🔄 Pending |
| Baichuan 2 | 7B/13B | 🔄 Pending |
| ChatGLM 3/4 | 6B/9B | 🔄 Pending |

**Legend:**
- ✅ Fully supported and tested
- 🔄 Pending - Coming soon

Currently, **Qwen2/Qwen2.5** models are fully supported. Other models will be added progressively.

## Usage

### 1. Configure Training

Edit `examples/configs/qwen_32b.yaml`:

```yaml
model:
  name: "Qwen/Qwen2.5-32B-Instruct"
  dtype: "bfloat16"

dataset:
  path: "/path/to/your/dataset"
  max_seq_len: 1024

training:
  batch_size: 96
  num_steps: 1000
  learning_rate: 1.0e-5

optimizer:
  type: "deepspeed_adam"  # or "adamw"
```

### 2. Train

```bash
# Use config file
python examples/train.py --config examples/configs/qwen_7b.yaml

# Override parameters
python examples/train.py \
    --config examples/configs/qwen_7b.yaml \
    --batch-size 148 \
    --num-steps 500
```

### 3. Use in Python

```python
from infinity import CPUMasterModel, CPUMasterConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model
config = CPUMasterConfig(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    dataset_path="/path/to/dataset",
    batch_size=148,
)

tokenizer = AutoTokenizer.from_pretrained(config.model_name)
hf_model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    torch_dtype=torch.bfloat16,
    device_map="cpu"
)

model = CPUMasterModel(hf_model, config)

# Train...
loss, tokens, timing = model.forward_and_backward(
    input_ids, attention_mask, labels
)
```

## Available Configs

| Config | Model | Batch Size | Seq Len |
|--------|-------|------------|---------|
| `qwen_32b.yaml` | Qwen 2.5 32B | 96 | 1024 |
| `qwen_7b.yaml` | Qwen 2.5 7B | 148 | 1024 |

See `examples/configs/README.md` for detailed configuration guide.

## Requirements

- **GPU**: NVIDIA GPU with 40GB+ VRAM (A100, H100, GH200)
- **CPU RAM**: 256GB+ for 32B models
- **CUDA**: 11.8+
- **PyTorch**: 2.0+
- **Python**: 3.9+


**Key Techniques:**
- Double buffering for overlapped weight transfer
- Gradient checkpointing every K layers
- Async gradient collection with slab pool
- Manual gradient computation (no autograd overhead)

## Performance

Training Qwen 2.5 32B on single H100:
- **Memory**:  ~415GB CPU
- **Throughput**: ~259 TFLOPS
- **Batch Size**: 96 (seq_len=1024)

## Installation Details

When you run `pip install -e .`:
1. Installs Python dependencies (PyTorch, Transformers, etc.)
2. Builds CUDA extension automatically (if CUDA available)
3. Sets up `infinity` library for import

Optional dependencies:
```bash
# Flash Attention (recommended)
pip install flash-attn

# DeepSpeed CPUAdam (5-7x faster optimizer)
pip install deepspeed
```

## Troubleshooting

**Out of Memory?**
- Reduce `batch_size` in config
- Increase `checkpoint_interval`
- Reduce `max_seq_len`

**Slow Training?**
- Use `deepspeed_adam` optimizer
- Install Flash Attention
- Increase `num_workers` for data loading

**CUDA Extension Failed?**
- Training still works without it (slightly slower)
- Check CUDA version: `nvcc --version`
- Manually build: `cd infinity/cuda_pipeline && python setup.py install`

## Citation

If you use Horizon-LM in your research, please cite:

```bibtex
@software{horizon-lm,
  title = {Horizon-LM: Single-GPU Training of Hundreds-of-Billions Parameter Language Models with Mixed BF16/FP32 Precision},
  author = {Zhengqing Yuan, Lichao Sun, Yanfang (Fanny) Ye},
  year = {2026},
  url = {https://github.com/DLYuanGod/Horizon-LM}
}
```

