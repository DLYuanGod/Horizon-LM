# Configuration Guide

This directory contains YAML configuration files for training different models with Infinity.

## Quick Start

### Using a Configuration File

```bash
# Train with Qwen 32B configuration
python train_cpu_master_v10.py --config configs/qwen_32b.yaml

# Train with Qwen 7B configuration
python train_cpu_master_v10.py --config configs/qwen_7b.yaml

# Train with LLaMA 3 8B configuration
python train_cpu_master_v10.py --config configs/llama3_8b.yaml
```

### Overriding Configuration Parameters

You can override specific parameters from the command line:

```bash
# Override model name
python train_cpu_master_v10.py --config configs/qwen_32b.yaml --model-name "Qwen/Qwen2.5-14B-Instruct"

# Override dataset path
python train_cpu_master_v10.py --config configs/qwen_32b.yaml --dataset-path "/path/to/your/dataset"

# Override batch size and number of steps
python train_cpu_master_v10.py --config configs/qwen_32b.yaml --batch-size 64 --num-steps 500
```

## Configuration File Structure

### Model Configuration

```yaml
model:
  name: "Qwen/Qwen2.5-32B-Instruct"  # HuggingFace model identifier
  dtype: "bfloat16"                   # Data type: bfloat16, float16, float32
  device: 0                           # CUDA device index
```

### Dataset Configuration

```yaml
dataset:
  path: "/path/to/dataset"            # Path to HuggingFace dataset
  max_seq_len: 1024                   # Maximum sequence length
  num_workers: 2                      # DataLoader workers
```

### Training Hyperparameters

```yaml
training:
  batch_size: 96                      # Batch size per step
  gradient_accumulation_steps: 1      # Gradient accumulation steps
  num_steps: 100                      # Total training steps
  learning_rate: 1.0e-5               # Learning rate
  weight_decay: 0.01                  # Weight decay
  max_grad_norm: 1.0                  # Gradient clipping norm
  seed: 42                            # Random seed
```

### Optimizer Configuration

```yaml
optimizer:
  type: "deepspeed_adam"              # Options: deepspeed_adam, adamw
  beta1: 0.9                          # Adam beta1
  beta2: 0.999                        # Adam beta2
  eps: 1.0e-8                         # Adam epsilon
```

**Note**: `deepspeed_adam` requires DeepSpeed to be installed. If not available, the script will fall back to PyTorch's AdamW.

### Memory Management

```yaml
memory:
  checkpoint_interval: 4              # Layers between checkpoints
  num_grad_slabs: 12                  # Gradient slab pool size
```

**Recommendation**: Set `num_grad_slabs >= 2 * checkpoint_interval` to avoid gradient slab starvation.

### Logging

```yaml
logging:
  log_interval: 1                     # Steps between logging
  enable_timing: true                 # Enable CUDA timing (adds sync overhead)
```

## Available Configurations

### qwen_32b.yaml
- **Model**: Qwen 2.5 32B Instruct
- **Batch Size**: 96
- **Sequence Length**: 1024
- **Optimized for**: Single H100 GPU
- **Use Case**: Production training of 32B models

### qwen_7b.yaml
- **Model**: Qwen 2.5 7B Instruct
- **Batch Size**: 128
- **Sequence Length**: 2048
- **Optimized for**: Development and testing
- **Use Case**: Faster iteration with smaller models

### llama3_8b.yaml
- **Model**: LLaMA 3 8B Instruct
- **Batch Size**: 128
- **Sequence Length**: 2048
- **Optimized for**: LLaMA-style models
- **Use Case**: Training LLaMA family models

### train_config.yaml
- **Template configuration** with detailed comments
- Use this as a starting point for custom configurations

## Creating Custom Configurations

1. Copy an existing configuration file:
   ```bash
   cp configs/qwen_32b.yaml configs/my_config.yaml
   ```

2. Edit the configuration:
   ```yaml
   model:
     name: "your-org/your-model"

   dataset:
     path: "/path/to/your/dataset"

   training:
     batch_size: 64
     num_steps: 1000
   ```

3. Run training:
   ```bash
   python train_cpu_master_v10.py --config configs/my_config.yaml
   ```

## Tips for Configuration

### Memory Optimization

- **Reduce batch_size**: Lower GPU memory usage
- **Increase checkpoint_interval**: Trade memory for recomputation
- **Adjust num_grad_slabs**: Balance between memory and performance

### Performance Optimization

- **Use deepspeed_adam**: 5-7x faster than PyTorch AdamW
- **Increase num_workers**: Faster data loading
- **Disable enable_timing**: Remove synchronization overhead (but lose timing info)

### Training Stability

- **Lower learning_rate**: More stable training (1e-6 to 1e-5)
- **Increase gradient_accumulation_steps**: Larger effective batch size
- **Adjust max_grad_norm**: Prevent gradient explosion (0.5 to 2.0)

## Troubleshooting

### Out of Memory (OOM)

1. Reduce `batch_size`
2. Increase `checkpoint_interval`
3. Reduce `max_seq_len`
4. Reduce `num_grad_slabs`

### Slow Training

1. Use `deepspeed_adam` optimizer
2. Increase `num_workers`
3. Disable `enable_timing`
4. Reduce `checkpoint_interval` (if memory allows)

### Loss Not Decreasing

1. Increase `learning_rate`
2. Reduce `weight_decay`
3. Check dataset quality
4. Verify `max_grad_norm` is not too small

## Example Workflows

### Quick Test Run

```bash
# Use small model with few steps for testing
python train_cpu_master_v10.py --config configs/qwen_7b.yaml --num-steps 10
```

### Production Training

```bash
# Full training run with 32B model
python train_cpu_master_v10.py --config configs/qwen_32b.yaml --num-steps 10000
```

### Hyperparameter Sweep

```bash
# Try different learning rates
for lr in 1e-6 5e-6 1e-5 5e-5; do
    python train_cpu_master_v10.py --config configs/qwen_32b.yaml --learning-rate $lr
done
```

## Additional Resources

- [Infinity Documentation](../../README.md)
- [CPUMasterModel API](../../infinity/model/cpu_master.py)
- [Configuration Schema](../../infinity/config/training.py)
