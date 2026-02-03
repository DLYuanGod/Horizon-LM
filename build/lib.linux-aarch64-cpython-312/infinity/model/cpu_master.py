"""CPU Master Model with explicit recompute and async pipeline.

This module implements a CPU-backed training system for large language models that
exceed GPU memory capacity. Key features:
- FP32 master parameters stored on CPU
- Double-buffered GPU layer execution
- Async weight transfer and gradient collection
- K-slab gradient pool for memory efficiency
- Manual gradient computation without autograd overhead
"""

import logging
import copy
import gc
import threading
import queue
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# Try to import flash-attn CrossEntropyLoss
try:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss as FlashCrossEntropyLoss
    FLASH_CE_AVAILABLE = True
except ImportError:
    FLASH_CE_AVAILABLE = False


class CPUMasterModel:
    """CPU master with explicit recompute and async pipeline.
    
    This model keeps FP32 master parameters on CPU and uses double-buffered GPU
    execution with async weight transfer and gradient collection.
    
    Args:
        hf_model: HuggingFace model to wrap
        config: CPUMasterConfig instance
    """

    def __init__(self, hf_model, config):
        self.config = config
        self.device = torch.device(f"cuda:{config.device}")

        model = hf_model.model if hasattr(hf_model, 'model') else hf_model

        self.vocab_size = hf_model.config.vocab_size
        self.hidden_size = hf_model.config.hidden_size

        cfg = hf_model.config
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads

        # CPU master modules
        self.embedding = model.embed_tokens.cpu()
        self.norm = model.norm.cpu() if hasattr(model, 'norm') else None
        self.lm_head = hf_model.lm_head.cpu()

        # Detect weight tying
        self.tied_lm_head = False
        if hasattr(self.lm_head, "weight") and hasattr(self.embedding, "weight"):
            self.tied_lm_head = (self.lm_head.weight is self.embedding.weight)
            if self.tied_lm_head:
                logger.info("Detected tied lm_head and embedding weights")

        self.rotary_emb = model.rotary_emb.cpu() if hasattr(model, 'rotary_emb') else None

        # CPU master layers
        self.cpu_layers = [layer.cpu() for layer in model.layers]

        # Replace with Flash Attention
        from infinity.ops.attention import FlashAttentionLayer
        logger.info("Replacing attention with Flash Attention...")
        for layer in self.cpu_layers:
            layer.self_attn = FlashAttentionLayer(layer.self_attn)

        # GPU modules (created once, reused)
        logger.info("Creating GPU modules...")
        self.emb_gpu = copy.deepcopy(self.embedding).to(self.device)
        self.norm_gpu = copy.deepcopy(self.norm).to(self.device) if self.norm else None
        self.lm_head_gpu = copy.deepcopy(self.lm_head).to(self.device)

        # Restore weight tying on GPU
        if self.tied_lm_head and hasattr(self.lm_head_gpu, "weight"):
            self.lm_head_gpu.weight = self.emb_gpu.weight
            logger.info("Restored weight tying on GPU")

        self.rotary_gpu = copy.deepcopy(self.rotary_emb).to(self.device) if self.rotary_emb else None

        # Flatten layer parameters for efficient H2D copy
        logger.info("Creating flattened parameter buffers...")
        self.layer_param_shapes = []
        self.layer_param_numel = []
        self.layer_cpu_params = []

        for layer in self.cpu_layers:
            shapes = [p.shape for p in layer.parameters()]
            numel = [p.numel() for p in layer.parameters()]
            cpu_params = list(layer.parameters())
            self.layer_param_shapes.append(shapes)
            self.layer_param_numel.append(numel)
            self.layer_cpu_params.append(cpu_params)

        self.layer_total_numel = sum(self.layer_param_numel[0])

        # Calculate head and embedding sizes
        self.head_total_numel = sum(p.numel() for p in self.lm_head.parameters())
        if self.norm:
            self.head_total_numel += sum(p.numel() for p in self.norm.parameters())

        self.embed_total_numel = sum(p.numel() for p in self.embedding.parameters())

        # Double-buffered CPU flat buffers
        self.cpu_flat_buffers = [
            torch.empty(self.layer_total_numel, dtype=config.dtype).pin_memory(),
            torch.empty(self.layer_total_numel, dtype=config.dtype).pin_memory()
        ]

        # Double-buffered GPU flat params
        self.gpu_flat_buffers = [
            torch.empty(self.layer_total_numel, dtype=config.dtype, device=self.device),
            torch.empty(self.layer_total_numel, dtype=config.dtype, device=self.device)
        ]

        # Double-buffered GPU layer templates
        logger.info("Creating double-buffered GPU layer templates...")
        self.gpu_layer_buffers = [
            copy.deepcopy(self.cpu_layers[0]).to(self.device),
            copy.deepcopy(self.cpu_layers[0]).to(self.device)
        ]

        # CUDA streams
        self.compute_stream = torch.cuda.current_stream(device=self.device)
        self.weight_stream = torch.cuda.Stream(device=self.device)
        self.grad_stream = torch.cuda.Stream(device=self.device)

        # CUDA events for synchronization
        self.weight_ready_events = [torch.cuda.Event(enable_timing=False) for _ in range(2)]
        self.h2d_done_events = [torch.cuda.Event(enable_timing=False) for _ in range(2)]
        self.backward_done_events = [torch.cuda.Event(enable_timing=False) for _ in range(2)]
        self.buffer_busy_events = [torch.cuda.Event(enable_timing=False) for _ in range(2)]
        self.buffer_free_events = [torch.cuda.Event(enable_timing=False) for _ in range(2)]
        self.param_sync_event = torch.cuda.Event(enable_timing=False)
        self.loss_backward_done = torch.cuda.Event(enable_timing=False)
        self.embedding_backward_done = torch.cuda.Event(enable_timing=False)

        # K-slab pool for gradient D2H
        logger.info(f"Creating gradient slab pools...")
        self.layer_grad_slabs = [
            torch.empty(self.layer_total_numel, dtype=config.dtype, device='cpu', pin_memory=True)
            for _ in range(config.num_grad_slabs)
        ]
        self.layer_slab_events = [
            torch.cuda.Event(enable_timing=False) for _ in range(config.num_grad_slabs)
        ]
        self.layer_slab_free_list = queue.Queue()
        for i in range(config.num_grad_slabs):
            self.layer_slab_free_list.put(i)

        # Head slab
        self.head_grad_slab = torch.empty(self.head_total_numel, dtype=config.dtype, device='cpu', pin_memory=True)
        self.head_slab_event = torch.cuda.Event(enable_timing=False)
        self.head_slab_free = threading.Event()
        self.head_slab_free.set()

        # Embedding slab
        self.embed_grad_slab = torch.empty(self.embed_total_numel, dtype=config.dtype, device='cpu', pin_memory=True)
        self.embed_slab_event = torch.cuda.Event(enable_timing=False)
        self.embed_slab_free = threading.Event()
        self.embed_slab_free.set()

        # CPU worker thread for async gradient accumulation
        self.grad_task_queue = queue.Queue()
        self.worker_stop = threading.Event()
        self.worker_thread = threading.Thread(target=self._grad_worker, daemon=True)
        self.worker_thread.start()

        # Initialize buffer state events
        logger.info("Initializing buffer state events...")
        current_stream = torch.cuda.current_stream(self.device)
        for i in range(2):
            self.buffer_free_events[i].record(current_stream)
            self.h2d_done_events[i].record(current_stream)
        self.param_sync_event.record(current_stream)
        current_stream.synchronize()

        logger.info(f"Model: {len(self.cpu_layers)} layers, checkpoint every {config.checkpoint_interval}")
        logger.info(f"Flattened param size per layer: {self.layer_total_numel * config.dtype.itemsize / 1024**2:.2f} MB")
        logger.info(f"Gradient slab pools:")
        logger.info(f"  - Layer slabs: {config.num_grad_slabs} × {self.layer_total_numel * config.dtype.itemsize / 1024**2:.2f} MB")
        logger.info(f"  - Head slab: {self.head_total_numel * config.dtype.itemsize / 1024**2:.2f} MB")
        logger.info(f"  - Embed slab: {self.embed_total_numel * config.dtype.itemsize / 1024**2:.2f} MB")

        # Flash-attn CrossEntropyLoss
        if FLASH_CE_AVAILABLE:
            self.ce_loss = FlashCrossEntropyLoss(inplace_backward=True, ignore_index=-100, reduction='none')
            logger.info("Using flash-attn CrossEntropyLoss")
        else:
            self.ce_loss = None
            logger.info("Using standard PyTorch CE")


    def _grad_worker(self):
        """CPU worker thread: wait for D2H completion, accumulate gradients, return slab to pool."""
        while not self.worker_stop.is_set():
            try:
                task = self.grad_task_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            slab_type, slab_idx, cpu_params, shapes, numels = task

            if slab_type == 'layer':
                event = self.layer_slab_events[slab_idx]
                slab_flat = self.layer_grad_slabs[slab_idx]
            elif slab_type == 'head':
                event = self.head_slab_event
                slab_flat = self.head_grad_slab
            else:  # 'embed'
                event = self.embed_slab_event
                slab_flat = self.embed_grad_slab

            # Wait for D2H copy to complete
            event.synchronize()

            # Unflatten and accumulate gradients to CPU params
            offset = 0
            for p_cpu, shape, numel in zip(cpu_params, shapes, numels):
                grad_view = slab_flat[offset:offset + numel].view(shape)
                if p_cpu.grad is None:
                    p_cpu.grad = torch.empty_like(grad_view, device='cpu')
                    p_cpu.grad.copy_(grad_view)
                else:
                    p_cpu.grad.add_(grad_view)
                offset += numel

            # Return slab to free list
            if slab_type == 'layer':
                self.layer_slab_free_list.put(slab_idx)
            elif slab_type == 'head':
                self.head_slab_free.set()
            else:  # 'embed'
                self.embed_slab_free.set()

            self.grad_task_queue.task_done()

    def _sync_params_to_gpu(self):
        """Sync CPU master params to GPU modules (call after optimizer step)."""
        # Sync embedding
        for p_gpu, p_cpu in zip(self.emb_gpu.parameters(), self.embedding.parameters()):
            p_gpu.data.copy_(p_cpu.data, non_blocking=True)

        if self.norm_gpu:
            for p_gpu, p_cpu in zip(self.norm_gpu.parameters(), self.norm.parameters()):
                p_gpu.data.copy_(p_cpu.data, non_blocking=True)

        # Sync lm_head (skip if tied)
        if not self.tied_lm_head:
            for p_gpu, p_cpu in zip(self.lm_head_gpu.parameters(), self.lm_head.parameters()):
                p_gpu.data.copy_(p_cpu.data, non_blocking=True)

        if self.rotary_gpu:
            for p_gpu, p_cpu in zip(self.rotary_gpu.parameters(), self.rotary_emb.parameters()):
                p_gpu.data.copy_(p_cpu.data, non_blocking=True)

        # Record event: parameter sync is queued
        self.param_sync_event.record(torch.cuda.current_stream(self.device))

    def _load_layer_to_buffer_async(self, layer_idx, buffer_idx):
        """Load CPU layer params to GPU buffer asynchronously."""
        # Wait for previous H2D on this CPU buffer to complete
        self.h2d_done_events[buffer_idx].synchronize()

        # Wait for buffer to be free
        self.weight_stream.wait_event(self.buffer_free_events[buffer_idx])

        # Pack layer params into CPU flat buffer
        cpu_flat = self.cpu_flat_buffers[buffer_idx]
        layer = self.cpu_layers[layer_idx]
        offset = 0
        for p in layer.parameters():
            numel = p.numel()
            cpu_flat[offset:offset + numel].copy_(p.data.flatten())
            offset += numel

        # Single H2D copy on weight_stream
        with torch.cuda.stream(self.weight_stream):
            self.gpu_flat_buffers[buffer_idx].copy_(cpu_flat, non_blocking=True)
            self.weight_ready_events[buffer_idx].record(self.weight_stream)
            self.h2d_done_events[buffer_idx].record(self.weight_stream)

    def _unflatten_to_layer(self, buffer_idx):
        """Unflatten GPU buffer to layer parameters."""
        flat = self.gpu_flat_buffers[buffer_idx]
        gpu_layer = self.gpu_layer_buffers[buffer_idx]

        offset = 0
        for p in gpu_layer.parameters():
            numel = p.numel()
            p.data.copy_(flat[offset:offset + numel].view(p.shape))
            offset += numel

    def _collect_layer_grads_async(self, layer_idx, buffer_idx):
        """Collect GPU buffer grads to CPU layer using K-slab flat buffer pool."""
        # Get a free layer slab
        slab_idx = self.layer_slab_free_list.get()
        slab_flat = self.layer_grad_slabs[slab_idx]

        # Wait for backward to complete on compute_stream
        self.grad_stream.wait_event(self.backward_done_events[buffer_idx])

        gpu_layer = self.gpu_layer_buffers[buffer_idx]

        # Queue all D2H copies on grad_stream
        with torch.cuda.stream(self.grad_stream):
            offset = 0
            for p_gpu in gpu_layer.parameters():
                if p_gpu.grad is not None:
                    numel = p_gpu.grad.numel()
                    slab_flat[offset:offset + numel].copy_(p_gpu.grad.flatten(), non_blocking=True)
                    p_gpu.grad.record_stream(self.grad_stream)
                    p_gpu.grad = None
                    offset += numel

            # Record D2H completion event
            self.layer_slab_events[slab_idx].record(self.grad_stream)
            self.buffer_free_events[buffer_idx].record(self.grad_stream)

        # Queue task for CPU worker
        self.grad_task_queue.put((
            'layer',
            slab_idx,
            self.layer_cpu_params[layer_idx],
            self.layer_param_shapes[layer_idx],
            self.layer_param_numel[layer_idx]
        ))

    def _accumulate_grads_batch(self):
        """Wait for CPU worker to finish all gradient accumulation tasks."""
        self.grad_task_queue.join()


    def forward_and_backward(self, input_ids, attention_mask, labels):
        """Forward and backward pass with explicit recompute.
        
        Args:
            input_ids: Input token IDs [B, T]
            attention_mask: Attention mask [B, T]
            labels: Target labels [B, T]
            
        Returns:
            Tuple of (loss_value, num_tokens, timing_dict)
        """
        B, T = input_ids.shape

        # Wait for parameter sync from previous step
        self.compute_stream.wait_event(self.param_sync_event)

        # Timing events
        start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)
        start.record()

        # === FORWARD (no graph, checkpoint every K layers) ===

        input_ids_gpu = input_ids.to(self.device)
        hidden = self.emb_gpu(input_ids_gpu)
        del input_ids_gpu

        # Cache position for RoPE
        cache_position = torch.arange(T, device=self.device)

        # Compute position_embeddings manually
        if self.rotary_gpu:
            dummy = torch.empty((1, 1, T, self.head_dim), device=self.device, dtype=torch.float32)
            position_ids = torch.arange(T, device=self.device).unsqueeze(0).expand(B, -1)
            cos, sin = self.rotary_gpu(dummy, position_ids[:1])
            position_embeddings = (cos.to(self.config.dtype), sin.to(self.config.dtype))
            del dummy, position_ids
        else:
            position_embeddings = None

        # Use 2D attention mask for FlashAttention
        mask = attention_mask.to(self.device)

        # Checkpoints: store on GPU
        checkpoints = {}

        with torch.no_grad():
            # Preload first layer into buffer 0
            self._load_layer_to_buffer_async(0, 0)
            self.weight_stream.synchronize()
            self._unflatten_to_layer(0)

            for i in range(len(self.cpu_layers)):
                buffer_idx = i % 2
                next_buffer_idx = (i + 1) % 2

                # Checkpoint before layer
                if i % self.config.checkpoint_interval == 0:
                    checkpoints[i] = hidden.detach()

                # Async prefetch next layer
                if i + 1 < len(self.cpu_layers):
                    self._load_layer_to_buffer_async(i + 1, next_buffer_idx)

                # Wait for current layer's weights to be ready
                self.compute_stream.wait_event(self.weight_ready_events[buffer_idx])

                # Unflatten weights to layer
                with torch.cuda.stream(self.compute_stream):
                    self._unflatten_to_layer(buffer_idx)
                    self.buffer_busy_events[buffer_idx].record(self.compute_stream)

                    # Compute current layer
                    out = self.gpu_layer_buffers[buffer_idx](
                        hidden, attention_mask=mask, cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        use_cache=False, output_attentions=False
                    )
                    hidden = out[0] if isinstance(out, tuple) else out

        # Final checkpoint
        checkpoints[len(self.cpu_layers)] = hidden.detach()

        # Norm
        if self.norm_gpu:
            hidden = self.norm_gpu(hidden)

        fwd_end.record()

        # === LOSS + BACKWARD ===

        labels_gpu = labels.to(self.device)
        B, T = hidden.shape[0], hidden.shape[1]
        H = self.hidden_size
        V = self.vocab_size
        chunk_size = 128

        # Recompute norm with gradient enabled
        hidden_before_norm = checkpoints[len(self.cpu_layers)].requires_grad_(True)
        if self.norm_gpu:
            hidden_after_norm = self.norm_gpu(hidden_before_norm)
        else:
            hidden_after_norm = hidden_before_norm

        # Compute loss with autograd graph intact
        total_loss = torch.zeros((), device=self.device, dtype=torch.float32)
        total_valid_tokens = 0

        for t_start in range(0, T - 1, chunk_size):
            t_end = min(t_start + chunk_size, T - 1)

            h = hidden_after_norm[:, t_start:t_end, :]
            y = labels_gpu[:, t_start+1:t_end+1]

            # Compute logits
            logits = self.lm_head_gpu(h)

            flat_y = y.reshape(-1)
            flat_logits = logits.reshape(-1, V)

            if self.ce_loss is not None:
                per_tok = self.ce_loss(flat_logits, flat_y)
                valid = (flat_y != -100)
                loss_chunk = per_tok[valid].sum()
                total_valid_tokens += int(valid.sum().item())
            else:
                loss_chunk = nn.functional.cross_entropy(
                    flat_logits, flat_y, ignore_index=-100, reduction='sum'
                )
                total_valid_tokens += int((flat_y != -100).sum().item())

            total_loss = total_loss + loss_chunk
            del logits, loss_chunk

        # Validation: Ensure we have valid tokens
        if total_valid_tokens == 0:
            logger.warning("No valid tokens in batch! Skipping...")
            return 0.0, B * T, {'forward': 0.0, 'backward': 0.0}

        # Final loss
        loss = total_loss / total_valid_tokens
        loss_val = loss.item()

        # Validation: Check for NaN or Inf
        if not torch.isfinite(torch.tensor(loss_val)):
            logger.error(f"Loss is {loss_val}! Training may be unstable.")

        # Single backward
        loss.backward()

        # Record event: loss backward complete
        self.loss_backward_done.record(self.compute_stream)

        # Collect gradients from norm and lm_head
        grad_hidden = hidden_before_norm.grad.detach()

        # Wait for head slab to be free
        if not self.head_slab_free.wait(timeout=30.0):
            raise RuntimeError("head slab wait timeout: worker may be stalled")
        self.head_slab_free.clear()
        slab_flat = self.head_grad_slab

        # Copy lm_head/norm grads on grad_stream
        with torch.cuda.stream(self.grad_stream):
            self.grad_stream.wait_event(self.loss_backward_done)

            offset = 0
            # Copy lm_head grads (skip if tied)
            if not self.tied_lm_head:
                for p_gpu in self.lm_head_gpu.parameters():
                    if p_gpu.grad is not None:
                        numel = p_gpu.grad.numel()
                        slab_flat[offset:offset + numel].copy_(p_gpu.grad.flatten(), non_blocking=True)
                        p_gpu.grad = None
                        offset += numel

            # Copy norm grads
            if self.norm_gpu:
                for p_gpu in self.norm_gpu.parameters():
                    if p_gpu.grad is not None:
                        numel = p_gpu.grad.numel()
                        slab_flat[offset:offset + numel].copy_(p_gpu.grad.flatten(), non_blocking=True)
                        p_gpu.grad = None
                        offset += numel

            self.head_slab_event.record(self.grad_stream)

        # Queue task for CPU worker
        cpu_params = []
        if not self.tied_lm_head:
            cpu_params.extend(self.lm_head.parameters())
        if self.norm_gpu:
            cpu_params.extend(self.norm.parameters())
        shapes = [p.shape for p in cpu_params]
        numels = [p.numel() for p in cpu_params]
        self.grad_task_queue.put(('head', None, cpu_params, shapes, numels))

        del labels_gpu, hidden_after_norm, hidden_before_norm, total_loss

        # Backward through layers (block-wise, checkpoints from CPU)
        num_blocks = (len(self.cpu_layers) + self.config.checkpoint_interval - 1) // self.config.checkpoint_interval

        for block_idx in range(num_blocks - 1, -1, -1):
            block_start = block_idx * self.config.checkpoint_interval
            block_end = min((block_idx + 1) * self.config.checkpoint_interval, len(self.cpu_layers))

            # Load checkpoint from GPU
            current_checkpoint = checkpoints[block_start]

            # Recompute entire block, cache on GPU
            recompute_cache = {}
            hidden_recompute = current_checkpoint

            with torch.no_grad():
                for j in range(block_start, block_end):
                    buffer_idx = j % 2

                    # Async load
                    self._load_layer_to_buffer_async(j, buffer_idx)

                    # Wait for weights
                    self.compute_stream.wait_event(self.weight_ready_events[buffer_idx])

                    # Unflatten and compute
                    with torch.cuda.stream(self.compute_stream):
                        self._unflatten_to_layer(buffer_idx)
                        self.buffer_busy_events[buffer_idx].record(self.compute_stream)

                        out = self.gpu_layer_buffers[buffer_idx](
                            hidden_recompute, attention_mask=mask, cache_position=cache_position,
                            position_embeddings=position_embeddings,
                            use_cache=False, output_attentions=False
                        )
                        hidden_recompute = out[0] if isinstance(out, tuple) else out

                    recompute_cache[j] = hidden_recompute.detach()
                    del out

            # Backward through block
            for i in range(block_end - 1, block_start - 1, -1):
                buffer_idx = i % 2

                # Get input from cache
                if i == block_start:
                    layer_input = current_checkpoint.detach().requires_grad_(True)
                else:
                    layer_input = recompute_cache[i - 1].requires_grad_(True)

                # Load layer i weights
                self._load_layer_to_buffer_async(i, buffer_idx)

                # Wait for weights
                self.compute_stream.wait_event(self.weight_ready_events[buffer_idx])

                # Unflatten, forward and backward with autograd.grad
                with torch.cuda.stream(self.compute_stream):
                    self._unflatten_to_layer(buffer_idx)
                    self.buffer_busy_events[buffer_idx].record(self.compute_stream)

                    gpu_layer = self.gpu_layer_buffers[buffer_idx]
                    out = gpu_layer(
                        layer_input, attention_mask=mask, cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        use_cache=False, output_attentions=False
                    )
                    layer_output = out[0] if isinstance(out, tuple) else out

                    # Use autograd.grad for explicit gradient computation
                    grads = torch.autograd.grad(
                        outputs=layer_output,
                        inputs=(layer_input, *gpu_layer.parameters()),
                        grad_outputs=grad_hidden,
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=False,
                    )
                    grad_hidden = grads[0].detach()
                    param_grads = grads[1:]

                    # Write param_grads to GPU layer's .grad for async D2H
                    for p, g in zip(gpu_layer.parameters(), param_grads):
                        p.grad = g

                    self.backward_done_events[buffer_idx].record(self.compute_stream)

                # Collect grads asynchronously
                self._collect_layer_grads_async(i, buffer_idx)

                # Release cache
                if i in recompute_cache:
                    del recompute_cache[i]

                del layer_input, layer_output, out

            recompute_cache.clear()

        # === BACKWARD THROUGH EMBEDDING ===

        # Recompute embedding with gradient enabled
        input_ids_gpu = input_ids.to(self.device)
        emb_out = self.emb_gpu(input_ids_gpu)

        # Backward through embedding
        emb_out.backward(grad_hidden)

        # Record event: embedding backward complete
        self.embedding_backward_done.record(self.compute_stream)

        # Wait for embed slab to be free
        if not self.embed_slab_free.wait(timeout=30.0):
            raise RuntimeError("embed slab wait timeout: worker may be stalled")
        self.embed_slab_free.clear()
        slab_flat = self.embed_grad_slab

        # Collect embedding grads on grad_stream
        with torch.cuda.stream(self.grad_stream):
            self.grad_stream.wait_event(self.embedding_backward_done)

            offset = 0
            for p_gpu in self.emb_gpu.parameters():
                if p_gpu.grad is not None:
                    numel = p_gpu.grad.numel()
                    slab_flat[offset:offset + numel].copy_(p_gpu.grad.flatten(), non_blocking=True)
                    p_gpu.grad = None
                    offset += numel

            self.embed_slab_event.record(self.grad_stream)

        # Queue task for CPU worker
        cpu_params = list(self.embedding.parameters())
        shapes = [p.shape for p in cpu_params]
        numels = [p.numel() for p in cpu_params]
        self.grad_task_queue.put(('embed', None, cpu_params, shapes, numels))

        del input_ids_gpu, emb_out
        del mask, cache_position, position_embeddings, grad_hidden
        checkpoints.clear()

        # Batch accumulate all gradients on CPU
        self._accumulate_grads_batch()

        # Timing
        bwd_end.record()
        torch.cuda.synchronize()
        fwd_time = start.elapsed_time(fwd_end) / 1000.0
        bwd_time = fwd_end.elapsed_time(bwd_end) / 1000.0
        total_time = start.elapsed_time(bwd_end) / 1000.0

        return loss_val, B * T, {
            'forward': fwd_time,
            'backward': bwd_time,
            'total': total_time,
        }


    def get_parameters(self):
        """Get all parameters, deduplicated by object id to avoid double-optimizing tied weights."""
        seen = set()
        params = []

        for p in self.embedding.parameters():
            if id(p) not in seen:
                params.append(p)
                seen.add(id(p))

        for layer in self.cpu_layers:
            for p in layer.parameters():
                if id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))

        if self.norm is not None:
            for p in self.norm.parameters():
                if id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))

        for p in self.lm_head.parameters():
            if id(p) not in seen:
                params.append(p)
                seen.add(id(p))

        return params

    def zero_grad(self):
        """Zero all gradients."""
        for p in self.get_parameters():
            if p.grad is not None:
                p.grad.zero_()

    def cleanup(self):
        """Stop worker thread and cleanup resources."""
        self.worker_stop.set()
        self.worker_thread.join(timeout=5.0)
