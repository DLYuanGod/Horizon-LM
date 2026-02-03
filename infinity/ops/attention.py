"""Attention operations for Infinity.

This module provides Flash Attention integration for memory-efficient attention computation.
"""

import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from transformers.integrations.flash_attention import flash_attention_forward


class FlashAttentionLayer(nn.Module):
    """Flash Attention wrapper that uses HuggingFace's official flash_attention_forward implementation.

    This wrapper replaces standard attention with Flash Attention for improved memory efficiency
    and performance. It maintains compatibility with HuggingFace's attention interface.

    Args:
        hf_attn: HuggingFace attention module to wrap
    """

    def __init__(self, hf_attn):
        super().__init__()
        # Keep original projection layers
        self.q_proj = hf_attn.q_proj
        self.k_proj = hf_attn.k_proj
        self.v_proj = hf_attn.v_proj
        self.o_proj = hf_attn.o_proj

        # Copy config and attributes
        self.config = hf_attn.config
        self.layer_idx = hf_attn.layer_idx if hasattr(hf_attn, 'layer_idx') else None
        self.is_causal = getattr(hf_attn, 'is_causal', True)
        self.attention_dropout = hf_attn.attention_dropout if hasattr(hf_attn, 'attention_dropout') else 0.0
        self.scaling = hf_attn.scaling if hasattr(hf_attn, 'scaling') else None
        self.head_dim = hf_attn.head_dim

        # Get head configuration
        if hasattr(hf_attn, 'num_heads'):
            self.num_heads = hf_attn.num_heads
        elif hasattr(hf_attn, 'config'):
            self.num_heads = hf_attn.config.num_attention_heads
        else:
            raise ValueError("Cannot find num_heads in attention module")

        if hasattr(hf_attn, 'num_key_value_heads'):
            self.num_kv_heads = hf_attn.num_key_value_heads
        elif hasattr(hf_attn, 'config'):
            self.num_kv_heads = getattr(hf_attn.config, 'num_key_value_heads', self.num_heads)
        else:
            self.num_kv_heads = self.num_heads

        # Sliding window for Qwen2
        self.sliding_window = hf_attn.sliding_window if hasattr(hf_attn, 'sliding_window') else None

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        cache_position=None,
        position_embeddings=None,
        use_cache=False,
        output_attentions=False,
        **kwargs,
    ):
        """Forward pass with Flash Attention.

        Args:
            hidden_states: Input hidden states [B, T, H]
            attention_mask: Attention mask [B, T]
            cache_position: Cache position for RoPE
            position_embeddings: Precomputed position embeddings (cos, sin)
            use_cache: Whether to use KV cache (not supported)
            output_attentions: Whether to output attention weights

        Returns:
            Tuple of (attention_output, attention_weights)
        """
        # Follow HF Qwen2Attention implementation exactly
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Q/K/V projection and reshape to [B, T, H, D]
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [B, H, T, D]
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Apply RoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Use HF's official flash_attention_forward
        attn_output, attn_weights = flash_attention_forward(
            self,
            query_states,  # [B, H, T, D]
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        # Reshape and output projection
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights
