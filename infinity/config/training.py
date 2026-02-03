"""Training configuration for CPU Master model."""

from dataclasses import dataclass
import torch


@dataclass
class CPUMasterConfig:
    """Configuration for CPUMasterModel training.

    Args:
        model_name: HuggingFace model identifier
        max_seq_len: Maximum sequence length
        batch_size: Batch size per training step
        gradient_accumulation_steps: Number of steps to accumulate gradients
        num_steps: Total number of training steps
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        beta1: Adam beta1 parameter
        beta2: Adam beta2 parameter
        eps: Adam epsilon parameter
        max_grad_norm: Maximum gradient norm for clipping
        device: CUDA device index
        dtype: Data type for GPU computations (bfloat16 or float16)
        seed: Random seed for reproducibility
        log_interval: Steps between logging
        checkpoint_interval: Layers between checkpoints (for gradient checkpointing)
        dataset_path: Path to training dataset
        enable_timing: Enable CUDA timing (adds sync overhead)
        num_grad_slabs: Number of gradient slab buffers (>= 2 * checkpoint_interval recommended)
    """

    model_name: str = "Qwen/Qwen2.5-32B-Instruct"
    max_seq_len: int = 1024
    batch_size: int = 96
    gradient_accumulation_steps: int = 1
    num_steps: int = 100
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    device: int = 0
    dtype: torch.dtype = torch.bfloat16
    seed: int = 42
    log_interval: int = 1
    checkpoint_interval: int = 4
    dataset_path: str = ""
    enable_timing: bool = True
    num_grad_slabs: int = 12

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.num_grad_slabs < 2 * self.checkpoint_interval:
            import warnings
            warnings.warn(
                f"num_grad_slabs ({self.num_grad_slabs}) < 2 * checkpoint_interval "
                f"({2 * self.checkpoint_interval}). This may cause gradient slab starvation."
            )

        if not self.dataset_path:
            raise ValueError("dataset_path must be specified")

    @classmethod
    def qwen_32b_preset(cls, dataset_path: str, **kwargs):
        """Preset configuration for Qwen 2.5 32B model.

        Args:
            dataset_path: Path to training dataset
            **kwargs: Additional configuration overrides
        """
        return cls(
            model_name="Qwen/Qwen2.5-32B-Instruct",
            max_seq_len=1024,
            batch_size=96,
            checkpoint_interval=4,
            num_grad_slabs=12,
            dataset_path=dataset_path,
            **kwargs
        )
