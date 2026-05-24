"""PCVRHyFormer: A hybrid transformer model for post-click conversion rate prediction."""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, NamedTuple, Tuple, Optional, Union


PAIR_62_66_FIDS = (62, 63, 64, 65, 66)
USER_DENSE_UE_FIDS = (61, 87)
USER_DENSE_RAW_FUSION_FIDS = (61, 87, 89, 90, 91)
USER_ACTIVITY_DENSE_FIDS = tuple(range(900, 916))
DENSE_CROSS_ITEM_SOURCES = ('user_dense', 'item_dense', 'item_ns')


class UserDenseSplit(NamedTuple):
    dense_branch_feats: torch.Tensor
    pair_dense_feats_62_66: torch.Tensor
    dense_head_feats: Tuple[torch.Tensor, ...]


class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    seq_data: dict        # {domain: tensor [B, S, L]}
    seq_lens: dict        # {domain: tensor [B]}
    seq_time_buckets: dict  # {domain: tensor [B, L]}
    user_hour: Optional[torch.Tensor] = None
    user_dayofweek: Optional[torch.Tensor] = None
    seq_hours: Optional[dict] = None       # {domain: tensor [B, L]}
    seq_dayofweeks: Optional[dict] = None  # {domain: tensor [B, L]}


# ═══════════════════════════════════════════════════════════════════════════════
# Rotary Position Embedding (RoPE)
# ═══════════════════════════════════════════════════════════════════════════════


class RotaryEmbedding(nn.Module):
    """Precomputes and caches RoPE cos/sin values.

    Attributes:
        dim: Rotary embedding dimension.
        max_seq_len: Maximum sequence length for cache.
        base: Base frequency for rotary encoding.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inv_freq: (dim // 2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Precompute cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim // 2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0), persistent=False)  # (1, seq_len, dim)
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0), persistent=False)  # (1, seq_len, dim)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes cos/sin values for the given sequence length.

        Returns pre-computed slices from the cache. The cache is built once
        in __init__ with max_seq_len; no runtime expansion is performed so
        that the forward pass remains compatible with torch.compile().
        """
        cos = self.cos_cached[:, :seq_len, :].to(device)
        sin = self.sin_cached[:, :seq_len, :].to(device)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swaps and negates the first and second halves of the last dimension."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_to_tensor(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Applies Rotary Position Embedding to a single tensor.

    Args:
        x: (B, num_heads, L, head_dim)
        cos: (1, L_max, head_dim) or (B, L, head_dim) for batch-specific positions.
        sin: Same shape as cos.

    Returns:
        Rotated tensor of shape (B, num_heads, L, head_dim).
    """
    L = x.shape[2]
    cos_ = cos[:, :L, :].unsqueeze(1)  # (*, 1, L, head_dim)
    sin_ = sin[:, :L, :].unsqueeze(1)
    return x * cos_ + rotate_half(x) * sin_


# ═══════════════════════════════════════════════════════════════════════════════
# HyFormer Basic Components
# ═══════════════════════════════════════════════════════════════════════════════


class SwiGLU(nn.Module):
    """SwiGLU activation: x1 * SiLU(x2)."""

    def __init__(self, d_model: int, hidden_mult: int = 4) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.fc = nn.Linear(d_model, 2 * hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = x1 * F.silu(x2)
        x = self.fc_out(x)
        return x


class HeadCrossNet(nn.Module):
    """Lightweight residual CrossNet over the final head vector."""

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        init_scale: float = 0.01,
    ) -> None:
        super().__init__()
        if num_layers < 0:
            raise ValueError("num_layers must be non-negative")
        self.layers = nn.ModuleList([
            nn.Linear(d_model, d_model)
            for _ in range(num_layers)
        ])
        self.scales = nn.Parameter(torch.full((num_layers,), float(init_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        h = x
        for scale, layer in zip(self.scales, self.layers):
            h = h + scale * x0 * layer(h)
        return h


class TokenAutoIntLayer(nn.Module):
    """Position-neutral self-attention over a compact token set."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.post_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.norm(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        y = self.dropout(y)
        return self.post_norm(residual + y)


class GatedTokenAutoInt(nn.Module):
    """Gated AutoInt residual over already-constructed semantic tokens."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive when GatedTokenAutoInt is enabled")
        if num_heads <= 0:
            raise ValueError(f"AutoInt heads must be positive, got {num_heads}")
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by AutoInt heads={num_heads}")
        self.layers = nn.ModuleList([
            TokenAutoIntLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for layer in self.layers:
            y = layer(y)
        return x + self.gate * (y - x)


class RoPEMultiheadAttention(nn.Module):
    """Multi-head attention with Rotary Position Embedding support.

    Manually projects Q/K/V and reshapes for multi-head, then injects RoPE
    after projection and before dot-product. Uses F.scaled_dot_product_attention
    for efficient computation.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_on_q: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_on_q = rope_on_q
        self.dropout = dropout

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.W_g = nn.Linear(d_model, d_model)

        nn.init.zeros_(self.W_g.weight)
        nn.init.constant_(self.W_g.bias, 1.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        q_rope_cos: Optional[torch.Tensor] = None,
        q_rope_sin: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> tuple:
        """Computes multi-head attention with optional RoPE.

        Args:
            query: (B, Lq, D)
            key: (B, Lk, D)
            value: (B, Lk, D)
            key_padding_mask: (B, Lk), True indicates padding positions.
            attn_mask: (Lq, Lk) or (B*num_heads, Lq, Lk), additive mask.
            rope_cos: (1, L, head_dim), RoPE for KV side (also used for Q
                unless q_rope_* is provided).
            rope_sin: Same shape as rope_cos.
            q_rope_cos: (B, Lq, head_dim) or (1, Lq, head_dim), Q-specific
                RoPE for cross-attention with gathered positions.
            q_rope_sin: Same shape as q_rope_cos.
            need_weights: Compatibility parameter, not used.

        Returns:
            Tuple of (output, None).
        """
        B, Lq, _ = query.shape
        Lk = key.shape[1]

        # 1. Linear projection
        Q = self.W_q(query)  # (B, Lq, D)
        K = self.W_k(key)    # (B, Lk, D)
        V = self.W_v(value)  # (B, Lk, D)

        # 2. Reshape to (B, num_heads, L, head_dim)
        Q = Q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Apply RoPE independently to Q and K
        if rope_cos is not None and rope_sin is not None:
            # K always uses rope_cos/rope_sin (KV-side positional encoding)
            K = apply_rope_to_tensor(K, rope_cos, rope_sin)

            if self.rope_on_q:
                # Q side: prefer dedicated q_rope_cos/sin (top_k positions in LongerEncoder cross-attn)
                q_cos = q_rope_cos if q_rope_cos is not None else rope_cos
                q_sin = q_rope_sin if q_rope_sin is not None else rope_sin
                Q = apply_rope_to_tensor(Q, q_cos, q_sin)

        # 4. Convert key_padding_mask to SDPA format
        sdpa_attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, Lk), True = padding
            # SDPA expects (B, 1, 1, Lk) bool mask, True = attend
            sdpa_attn_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, Lk)
            sdpa_attn_mask = sdpa_attn_mask.expand(B, self.num_heads, Lq, Lk)

        if attn_mask is not None:
            # attn_mask: additive float mask (Lq, Lk), -inf means do not attend
            # Convert to bool: positions that are not -inf are True
            bool_attn = (attn_mask == 0)  # (Lq, Lk)
            bool_attn = bool_attn.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, Lq, Lk)
            if sdpa_attn_mask is not None:
                sdpa_attn_mask = sdpa_attn_mask & bool_attn
            else:
                sdpa_attn_mask = bool_attn

        # 5. Scaled Dot-Product Attention
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=sdpa_attn_mask,
            dropout_p=dropout_p,
        )  # (B, num_heads, Lq, head_dim)

        # Replace NaN from all-padding softmax with 0 (zero vectors preserve original input via residual)
        out = torch.nan_to_num(out, nan=0.0)

        # 6. Reshape back and output projection
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        G = self.W_g(query)
        out = out * torch.sigmoid(G)
        out = self.W_o(out)

        return out, None


class CrossAttention(nn.Module):
    """Cross-attention module.

    Query comes from global tokens (Q tokens), Key/Value comes from sequence
    tokens. Only applies RoPE to KV side (rope_on_q=False).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        ln_mode: str = 'pre'
    ) -> None:
        super().__init__()
        self.ln_mode = ln_mode

        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=False,
        )

        if ln_mode in ['pre', 'post']:
            self.norm_q = nn.LayerNorm(d_model)
            self.norm_kv = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes cross-attention between query tokens and sequence tokens.

        Args:
            query: (B, Nq, D), query tokens.
            key_value: (B, L, D), sequence tokens.
            key_padding_mask: (B, L), True indicates padding positions.
            rope_cos: (1, L, head_dim), KV-side RoPE cosine values.
            rope_sin: (1, L, head_dim), KV-side RoPE sine values.

        Returns:
            Output tensor of shape (B, Nq, D).
        """
        residual = query

        if self.ln_mode == 'pre':
            query = self.norm_q(query)
            key_value = self.norm_kv(key_value)

        out, _ = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )

        out = residual + out

        if self.ln_mode == 'post':
            out = self.norm_q(out)

        return out


class RankMixerBlock(nn.Module):
    """HyFormer Query Boosting block.

    Performs three steps:
    1. Token Mixing: Parameter-free tensor reshaping.
    2. Per-token FFN: Shared-parameter feedforward network.
    3. Residual connection: Q_boost = Q + Q_e.

    Constraint: d_model must be divisible by n_total in 'full' mode.
    """

    def __init__(
        self,
        d_model: int,
        n_total: int,  # T = Nq + Nns
        hidden_mult: int = 4,
        dropout: float = 0.0,
        mode: str = 'full'  # 'full' | 'ffn_only' | 'none'
    ) -> None:
        super().__init__()
        self.T = n_total
        self.D = d_model
        self.mode = mode

        if mode == 'none':
            # Pure identity mapping, no submodules created
            return

        if mode == 'full':
            if d_model % n_total != 0:
                raise ValueError(
                    f"d_model={d_model} must be divisible by T={n_total} for token mixing."
                )
            self.d_sub = d_model // n_total

        # Per-token FFN (shared parameters) — used by both 'full' and 'ffn_only'
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * hidden_mult)
        self.fc2 = nn.Linear(d_model * hidden_mult, d_model)
        self.dropout = nn.Dropout(dropout)
        # Post-LN after residual to stabilize stacked block outputs
        self.post_norm = nn.LayerNorm(d_model)

    def token_mixing(self, Q: torch.Tensor) -> torch.Tensor:
        """Performs parameter-free token mixing via reshape and transpose.

        Steps:
        1. Splits channels into T subspaces: (B, T, D) -> (B, T, T, d_sub).
        2. Swaps token and subspace axes: (B, token, h, d_sub) -> (B, h, token, d_sub).
        3. Flattens back: (B, T, D).

        Args:
            Q: (B, T, D)

        Returns:
            Mixed tensor of shape (B, T, D).
        """
        B, T, D = Q.shape

        # (B, T, D) -> (B, T, T, d_sub)
        Q_split = Q.view(B, T, self.T, self.d_sub)

        # (B, token, h, d_sub) -> (B, h, token, d_sub)
        Q_rewired = Q_split.transpose(1, 2).contiguous()

        # (B, T, T, d_sub) -> (B, T, D)
        Q_hat = Q_rewired.view(B, T, D)
        return Q_hat

    def forward(self, Q: torch.Tensor) -> torch.Tensor:
        """Applies query boosting: token mixing, FFN, and residual connection.

        Args:
            Q: (B, T, D) where T = Nq + Nns.

        Returns:
            Boosted tensor of shape (B, T, D).
        """
        if self.mode == 'none':
            return Q

        # Token Mixing (parameter-free rewire) or identity
        if self.mode == 'full':
            Q_hat = self.token_mixing(Q)
        else:  # 'ffn_only'
            Q_hat = Q

        # Per-token FFN
        x = self.norm(Q_hat)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        Q_e = self.fc2(x)

        # Residual from original Q
        Q_boost = Q + Q_e
        Q_boost = self.post_norm(Q_boost)
        return Q_boost


class MultiSeqQueryGenerator(nn.Module):
    """Multi-sequence query generation module.

    Generates Q tokens independently for each sequence:
    For each sequence i:
        GlobalInfo_i = Concat(F1..FM, MeanPool(Seq_i))
        Q_i = [FFN_{i,1}(GlobalInfo_i), ..., FFN_{i,N}(GlobalInfo_i)]
    """

    def __init__(
        self,
        d_model: int,
        num_ns: int,
        num_queries: int,
        num_sequences: int,
        hidden_mult: int = 4
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.num_sequences = num_sequences
        self.d_model = d_model

        global_info_dim = (num_ns + 1) * d_model

        # LayerNorm on global_info to prevent gradient explosion from large-dim concat
        self.global_info_norm = nn.LayerNorm(global_info_dim)

        # Each sequence has N independent FFNs
        self.query_ffns_per_seq = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(global_info_dim, d_model * hidden_mult),
                    nn.SiLU(),
                    nn.Linear(d_model * hidden_mult, d_model),
                    nn.LayerNorm(d_model),
                )
                for _ in range(num_queries)
            ])
            for _ in range(num_sequences)
        ])

    def forward(
        self,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list
    ) -> list:
        """Generates query tokens for each sequence.

        Args:
            ns_tokens: (B, M, D), shared NS tokens.
            seq_tokens_list: List of (B, L_i, D) tensors, length S.
            seq_padding_masks: List of (B, L_i) masks, length S. True
                indicates padding.

        Returns:
            List of (B, Nq, D) query token tensors, length S.
        """
        B = ns_tokens.shape[0]
        ns_flat = ns_tokens.view(B, -1)  # (B, M*D)

        q_tokens_list = []
        for i in range(self.num_sequences):
            # MeanPool(Seq_i)
            valid_mask = ~seq_padding_masks[i]  # True = valid
            valid_mask_expanded = valid_mask.unsqueeze(-1).float()  # (B, L_i, 1)
            seq_sum = (seq_tokens_list[i] * valid_mask_expanded).sum(dim=1)  # (B, D)
            seq_count = valid_mask_expanded.sum(dim=1).clamp(min=1)  # (B, 1)
            seq_pooled = seq_sum / seq_count  # (B, D)

            # GlobalInfo_i = Concat(NS_flat, seq_pooled_i)
            global_info = torch.cat([ns_flat, seq_pooled], dim=-1)  # (B, (M+1)*D)
            global_info = self.global_info_norm(global_info)

            # Generate N query tokens
            queries = [ffn(global_info) for ffn in self.query_ffns_per_seq[i]]
            q_tokens = torch.stack(queries, dim=1)  # (B, Nq, D)
            q_tokens_list.append(q_tokens)

        return q_tokens_list


# ═══════════════════════════════════════════════════════════════════════════════
# Sequence Encoders
# ═══════════════════════════════════════════════════════════════════════════════


class SwiGLUEncoder(nn.Module):
    """Efficient attention-free sequence encoder.

    Structure: x + Dropout(SwiGLU(LN(x))).
    """

    def __init__(
        self,
        d_model: int,
        hidden_mult: int = 4,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.swiglu = SwiGLU(d_model, hidden_mult)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Applies the SwiGLU encoder with residual connection.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding. Not used by
                this encoder variant.
            **kwargs: Absorbs rope_cos/rope_sin and other unused parameters.

        Returns:
            Tuple of (output tensor of shape (B, L, D), key_padding_mask).
        """
        residual = x
        x = self.norm(x)
        x = self.swiglu(x)
        x = self.dropout(x)
        x = residual + x
        return x, key_padding_mask


class TransformerEncoder(nn.Module):
    """High-capacity sequence encoder with self-attention and RoPE.

    Structure: Standard Transformer Encoder Layer (Pre-LN).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        hidden_mult: int = 4,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
        )

        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Applies one Transformer encoder layer.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding positions.
            rope_cos: (1, L, head_dim), RoPE cosine values.
            rope_sin: (1, L, head_dim), RoPE sine values.

        Returns:
            Tuple of (output tensor of shape (B, L, D), key_padding_mask).
        """
        # Self-Attention (Pre-LN) with RoPE
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        x = residual + x

        # FFN (Pre-LN)
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x, key_padding_mask

class LongerEncoder(nn.Module):
    """Top-K compressed sequence encoder.

    Adapts behavior based on input length:
    - L > top_k (first MultiSeqHyFormerBlock): Cross Attention.
      Q = latest top_k tokens, K/V = all seq tokens -> output (B, top_k, D).
    - L <= top_k (subsequent MultiSeqHyFormerBlocks): Self Attention.
      Q = K = V = top_k tokens -> output (B, top_k, D).

    Causal mask is only applied among top_k tokens (self-attention layers);
    the first cross-attention layer does not use a causal mask since Q and K
    have different lengths.

    Returns (output, new_key_padding_mask) so downstream can update the mask.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        top_k: int = 50,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        causal: bool = False
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.causal = causal

        # Pre-LN for attention
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        # Shared RoPEMHA for both cross and self attention
        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
        )

        # FFN (Pre-LN + residual)
        self.ffn_norm = nn.LayerNorm(d_model)
        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def _gather_top_k(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Selects the latest top_k valid tokens from each sample.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding.

        Returns:
            top_k_tokens: (B, top_k, D)
            new_padding_mask: (B, top_k), True indicates padding.
            position_indices: (B, top_k), original position index for each
                selected token, used for Q-side RoPE.
        """
        B, L, D = x.shape
        device = x.device

        # Valid lengths per sample
        valid_len = (~key_padding_mask).sum(dim=1)  # (B,)

        # Start position for each sample: max(valid_len - top_k, 0)
        actual_k = torch.clamp(valid_len, max=self.top_k)  # (B,)
        start_pos = valid_len - actual_k  # (B,)

        # Build gather indices: (B, top_k)
        offsets = torch.arange(self.top_k, device=device).unsqueeze(0).expand(B, -1)  # (B, top_k)
        indices = start_pos.unsqueeze(1) + offsets  # (B, top_k)

        # For samples with valid_len < top_k, early indices may exceed valid range;
        # clamp to [0, L-1] and handle via mask below
        indices = torch.clamp(indices, min=0, max=L - 1)

        # Gather: (B, top_k, D)
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, D)  # (B, top_k, D)
        top_k_tokens = torch.gather(x, dim=1, index=indices_expanded)

        # New padding mask: first (top_k - actual_k) positions are padding
        new_valid_len = actual_k  # (B,)
        pad_count = self.top_k - new_valid_len  # (B,)
        pos_indices = torch.arange(self.top_k, device=device).unsqueeze(0)  # (1, top_k)
        new_padding_mask = pos_indices < pad_count.unsqueeze(1)  # (B, top_k)

        # Zero out tokens at padding positions
        top_k_tokens = top_k_tokens * (~new_padding_mask).unsqueeze(-1).float()

        # position_indices for Q-side RoPE
        position_indices = indices  # (B, top_k)

        return top_k_tokens, new_padding_mask, position_indices

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the LongerEncoder with adaptive cross/self attention.

        Args:
            x: (B, L, D), sequence tokens.
            key_padding_mask: (B, L), True indicates padding.
            rope_cos: (1, L, head_dim), RoPE cosine values (length must cover
                original sequence length L).
            rope_sin: (1, L, head_dim), RoPE sine values.

        Returns:
            output: (B, top_k, D), compressed sequence.
            new_key_padding_mask: (B, top_k), updated padding mask.
        """
        B, L, D = x.shape

        if L > self.top_k:
            # === Cross Attention mode (first MultiSeqHyFormerBlock) ===
            # 1. Extract latest top_k tokens as query
            q, new_mask, q_pos_indices = self._gather_top_k(x, key_padding_mask)

            # 2. Pre-LN
            q_normed = self.norm_q(q)
            kv_normed = self.norm_kv(x)

            # 3. Build Q-side RoPE cos/sin by gathering from global cos/sin at top_k positions
            q_rope_cos = None
            q_rope_sin = None
            if rope_cos is not None and rope_sin is not None:
                # rope_cos: (1, L_max, head_dim), q_pos_indices: (B, top_k)
                head_dim = rope_cos.shape[2]
                # Expand to batch dimension
                cos_expanded = rope_cos.expand(B, -1, -1)  # (B, L_max, head_dim)
                sin_expanded = rope_sin.expand(B, -1, -1)
                idx = q_pos_indices.unsqueeze(-1).expand(-1, -1, head_dim)  # (B, top_k, head_dim)
                q_rope_cos = torch.gather(cos_expanded, 1, idx)  # (B, top_k, head_dim)
                q_rope_sin = torch.gather(sin_expanded, 1, idx)

            # 4. Cross Attention (no causal mask since Q and K have different lengths)
            attn_out, _ = self.attn(
                query=q_normed,
                key=kv_normed,
                value=kv_normed,
                key_padding_mask=key_padding_mask,  # Original (B, L) mask
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                q_rope_cos=q_rope_cos,
                q_rope_sin=q_rope_sin,
            )
            out = q + attn_out  # Residual based on q
        else:
            # === Self Attention mode (subsequent MultiSeqHyFormerBlocks) ===
            new_mask = key_padding_mask

            # Pre-LN (Q and KV share norm_q)
            x_normed = self.norm_q(x)

            # Causal mask
            attn_mask = None
            if self.causal:
                attn_mask = nn.Transformer.generate_square_subsequent_mask(
                    L, device=x.device
                )

            attn_out, _ = self.attn(
                query=x_normed,
                key=x_normed,
                value=x_normed,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )
            out = x + attn_out

        # FFN (Pre-LN + residual)
        residual = out
        out = self.ffn_norm(out)
        out = self.ffn(out)
        out = residual + out

        return out, new_mask


def create_sequence_encoder(
    encoder_type: str,
    d_model: int,
    num_heads: int = 4,
    hidden_mult: int = 4,
    dropout: float = 0.0,
    top_k: int = 50,
    causal: bool = False
) -> nn.Module:
    """Creates a sequence encoder of the specified type.

    Args:
        encoder_type: One of 'swiglu', 'transformer', or 'longer'.
        d_model: Model dimension.
        num_heads: Number of attention heads (used by transformer/longer).
        hidden_mult: FFN expansion multiplier.
        dropout: Dropout rate.
        top_k: Compression length for LongerEncoder (only used by longer).
        causal: Whether to use causal mask in LongerEncoder (only used by
            longer).

    Returns:
        A sequence encoder module.
    """
    if encoder_type == 'swiglu':
        return SwiGLUEncoder(d_model, hidden_mult, dropout)
    elif encoder_type == 'transformer':
        return TransformerEncoder(d_model, num_heads, hidden_mult, dropout)
    elif encoder_type == 'longer':
        return LongerEncoder(d_model, num_heads, top_k, hidden_mult, dropout, causal)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# ═══════════════════════════════════════════════════════════════════════════════
# HyFormer Blocks
# ═══════════════════════════════════════════════════════════════════════════════


class MultiSeqHyFormerBlock(nn.Module):
    """Multi-sequence HyFormer block.

    Each of the S sequences independently performs Sequence Evolution and
    Query Decoding, then all Q tokens and shared NS tokens are merged for
    joint Query Boosting.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_queries: int,
        num_ns: int,
        num_sequences: int,
        seq_encoder_type: str = 'swiglu',
        hidden_mult: int = 4,
        dropout: float = 0.0,
        top_k: int = 50,
        causal: bool = False,
        rank_mixer_mode: str = 'full',
        use_ns_self_attn: bool = False,
        ns_self_attn_impl: str = 'mha',
        ns_autoint_layers: int = 2,
        ns_autoint_heads: int = 0,
        ns_autoint_gate_init: float = 0.05,
        ns_autoint_activation: str = 'relu',
        use_combined_autoint: bool = False,
        combined_autoint_layers: int = 1,
        combined_autoint_heads: int = 0,
        combined_autoint_dropout: Optional[float] = None,
        combined_autoint_gate_init: float = 0.0,
        combined_autoint_position: str = "pre_mixer",
    ) -> None:
        super().__init__()
        self.num_sequences = num_sequences
        self.num_queries = num_queries
        self.num_ns = num_ns
        if ns_self_attn_impl not in ('mha', 'autoint'):
            raise ValueError(
                "ns_self_attn_impl must be 'mha' or 'autoint', "
                f"got {ns_self_attn_impl!r}")
        if combined_autoint_position not in ("pre_mixer", "post_mixer"):
            raise ValueError(
                "combined_autoint_position must be 'pre_mixer' or 'post_mixer', "
                f"got {combined_autoint_position!r}")
        self.combined_autoint_position = combined_autoint_position

        # Independent sequence encoder per sequence
        self.seq_encoders = nn.ModuleList([
            create_sequence_encoder(
                encoder_type=seq_encoder_type,
                d_model=d_model,
                num_heads=num_heads,
                hidden_mult=hidden_mult,
                dropout=dropout,
                top_k=top_k,
                causal=causal
            )
            for _ in range(num_sequences)
        ])

        # Independent cross-attention per sequence
        self.cross_attns = nn.ModuleList([
            CrossAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                ln_mode='pre'
            )
            for _ in range(num_sequences)
        ])

        # RankMixer: input token count = Nq * S + Nns
        n_total = num_queries * num_sequences + num_ns
        self.mixer = RankMixerBlock(
            d_model=d_model,
            n_total=n_total,
            hidden_mult=hidden_mult,
            dropout=dropout,
            mode=rank_mixer_mode
        )
        self.ns_feature_cross = None
        if use_ns_self_attn:
            if ns_self_attn_impl == 'mha':
                self.ns_feature_cross = NSSelfAttention(d_model, num_heads, dropout)
            elif ns_self_attn_impl == 'autoint':
                heads = ns_autoint_heads if ns_autoint_heads > 0 else num_heads
                self.ns_feature_cross = NSAutoIntStack(
                    d_model=d_model,
                    num_heads=heads,
                    num_layers=ns_autoint_layers,
                    dropout=dropout,
                    activation=ns_autoint_activation,
                    gate_init=ns_autoint_gate_init,
                )
        heads = combined_autoint_heads if combined_autoint_heads > 0 else num_heads
        auto_dropout = dropout if combined_autoint_dropout is None else combined_autoint_dropout
        self.combined_autoint = (
            GatedTokenAutoInt(
                d_model=d_model,
                num_heads=heads,
                num_layers=combined_autoint_layers,
                dropout=auto_dropout,
                gate_init=combined_autoint_gate_init,
            )
            if use_combined_autoint else None
        )

    def forward(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list,
        rope_cos_list: Optional[List[torch.Tensor]] = None,
        rope_sin_list: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[list, torch.Tensor, list, list]:
        """Processes one multi-sequence HyFormer block step.

        Args:
            q_tokens_list: List of (B, Nq, D) tensors, length S.
            ns_tokens: (B, Nns, D)
            seq_tokens_list: List of (B, L_i, D) tensors, length S.
            seq_padding_masks: List of (B, L_i) masks, length S.
            rope_cos_list: List of (1, L_i, head_dim) tensors, length S.
            rope_sin_list: List of (1, L_i, head_dim) tensors, length S.

        Returns:
            A tuple (next_q_list, next_ns, next_seq_list, next_masks), where
            next_q_list is a list of (B, Nq, D) updated query tensors,
            next_ns is (B, Nns, D) updated non-sequence tokens,
            next_seq_list is a list of (B, L_i', D) encoded sequence tensors,
            and next_masks is a list of (B, L_i') updated padding masks.
        """
        S = self.num_sequences
        Nq = self.num_queries

        # 1. Independent Sequence Evolution per sequence
        next_seqs = []
        next_masks = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            result = self.seq_encoders[i](
                seq_tokens_list[i], seq_padding_masks[i],
                rope_cos=rc, rope_sin=rs,
            )
            next_seq_i, mask_i = result
            next_seqs.append(next_seq_i)
            next_masks.append(mask_i)

        # 2. Independent Query Decoding per sequence
        decoded_qs = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            decoded_q_i = self.cross_attns[i](
                q_tokens_list[i], next_seqs[i], next_masks[i],
                rope_cos=rc, rope_sin=rs,
            )
            decoded_qs.append(decoded_q_i)

        # Optional position-neutral feature crossing among current-block NS tokens.
        if self.ns_feature_cross is not None:
            ns_tokens = self.ns_feature_cross(ns_tokens)

        # 3. Token Fusion: concatenate all decoded_q + ns_tokens
        combined = torch.cat(decoded_qs + [ns_tokens], dim=1)  # (B, Nq*S + Nns, D)
        if self.combined_autoint is not None and self.combined_autoint_position == "pre_mixer":
            combined = self.combined_autoint(combined)

        # 4. Query Boosting
        boosted = self.mixer(combined)  # (B, Nq*S + Nns, D)
        if self.combined_autoint is not None and self.combined_autoint_position == "post_mixer":
            boosted = self.combined_autoint(boosted)

        # 5. Split back into per-sequence Q and NS
        next_q_list = []
        offset = 0
        for i in range(S):
            next_q_list.append(boosted[:, offset:offset + Nq, :])
            offset += Nq
        next_ns = boosted[:, offset:, :]

        return next_q_list, next_ns, next_seqs, next_masks


class NSSelfAttention(nn.Module):
    """Self-attention over NS tokens for in-block feature crossing.

    NS tokens have no sequence position, so no RoPE or positional bias is
    applied. The block keeps the operation local to the current HyFormerBlock.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, ns_tokens: torch.Tensor) -> torch.Tensor:
        residual = ns_tokens
        x = self.norm(ns_tokens)
        x, _ = self.attn(x, x, x, need_weights=False)
        return residual + x


class NSAutoIntLayer(nn.Module):
    """One gated AutoInt interacting layer over position-neutral NS tokens."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        activation: str = 'relu',
        gate_init: float = 0.05,
    ) -> None:
        super().__init__()
        if activation not in ('relu', 'silu', 'gelu'):
            raise ValueError(
                "activation must be one of 'relu', 'silu', or 'gelu', "
                f"got {activation!r}")
        self.activation = activation
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.res_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.res_proj.weight)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == 'relu':
            return F.relu(x)
        if self.activation == 'silu':
            return F.silu(x)
        return F.gelu(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        autoint_out = self.dropout(attn_out) + self.res_proj(residual)
        autoint_out = self._activate(autoint_out)
        return residual + self.gate * (autoint_out - residual)


class NSAutoIntStack(nn.Module):
    """Stacked AutoInt interacting layers for NS tokens."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        activation: str = 'relu',
        gate_init: float = 0.05,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1 for NSAutoIntStack")
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1 for NSAutoIntStack, got {num_heads}")
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by NS AutoInt heads={num_heads}")
        self.layers = nn.ModuleList([
            NSAutoIntLayer(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                activation=activation,
                gate_init=gate_init,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# PCVRHyFormer Main Model
# ═══════════════════════════════════════════════════════════════════════════════


class GroupNSTokenizer(nn.Module):
    """NS tokenizer used by ns_tokenizer_type='group'.

    Groups discrete features by fid, applies shared embedding with mean
    pooling per multi-valued feature, then projects each group to a single
    NS token (one token per group).
    """

    def __init__(self, feature_specs: List[Tuple[int, int, int]],
                 groups: List[List[int]], emb_dim: int, d_model: int,
                 emb_skip_threshold: int = 0,
                 pair_feature_specs: Optional[List[Tuple[int, int, int, int, int]]] = None,
                 use_pair_62_66: bool = False) -> None:
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.emb_skip_threshold = emb_skip_threshold
        self.use_pair_62_66 = use_pair_62_66
        self._pair_by_int_offset = self._build_pair_lookup(pair_feature_specs or [])
        self.last_pair_shapes: Dict[int, Tuple[int, ...]] = {}

        # One embedding table per fid (None if skipped by emb_skip_threshold
        # or if vocab_size <= 0 / no vocab info).
        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        # Map from fid index to position in self.embs (or -1 if filtered)
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)
        if self.use_pair_62_66:
            offset_to_fid_idx = {
                offset: i for i, (_, offset, _) in enumerate(feature_specs)
            }
            for int_offset, (fid, _, _, _) in self._pair_by_int_offset.items():
                fid_idx = offset_to_fid_idx[int_offset]
                if self._emb_index[fid_idx] == -1:
                    raise ValueError(
                        f"use_pair_62_66=True requires an embedding for fid {fid}; "
                        "increase emb_skip_threshold or disable the pair feature")

        # Per-group projection: num_fids_in_group * emb_dim -> d_model (with LayerNorm)
        self.group_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(group) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for group in groups
        ])

    @staticmethod
    def _build_pair_lookup(
        pair_feature_specs: List[Tuple[int, int, int, int, int]]
    ) -> Dict[int, Tuple[int, int, int, int]]:
        """Map int-feature offset to (fid, int_len, dense_offset, dense_len)."""
        lookup: Dict[int, Tuple[int, int, int, int]] = {}
        for fid, int_offset, int_len, dense_offset, dense_len in pair_feature_specs:
            if fid not in PAIR_62_66_FIDS:
                raise ValueError(f"pair feature fid={fid} is not in 62-66")
            if int_len != dense_len:
                raise ValueError(
                    f"pair fid={fid} int length ({int_len}) must match dense length ({dense_len})"
                )
            lookup[int_offset] = (fid, int_len, dense_offset, dense_len)
        return lookup

    def _pool_multi_value_embedding(
        self,
        vals: torch.Tensor,
        emb_all: torch.Tensor,
        offset: int,
        dense_feats: Optional[torch.Tensor],
    ) -> torch.Tensor:
        mask = (vals != 0).to(emb_all.dtype).unsqueeze(-1)  # (B, length, 1)
        count = mask.sum(dim=1).clamp(min=1)  # (B, 1)
        mean_pool = (emb_all * mask).sum(dim=1) / count  # (B, emb_dim)

        if not self.use_pair_62_66 or offset not in self._pair_by_int_offset:
            return mean_pool
        if dense_feats is None:
            raise ValueError("use_pair_62_66=True requires pair_dense_feats_62_66")

        fid, _, dense_offset, dense_len = self._pair_by_int_offset[offset]
        dense_vals = dense_feats[:, dense_offset:dense_offset + dense_len].to(emb_all.dtype)
        weights = torch.sqrt(torch.log1p(torch.clamp(dense_vals, min=0.0)))
        weights = weights.unsqueeze(-1) * mask
        weighted_sum = (emb_all * weights).sum(dim=1)
        all_zero_weight = weights.sum(dim=1) <= 0
        self.last_pair_shapes[fid] = tuple(weighted_sum.shape)
        return torch.where(all_zero_weight, mean_pool, weighted_sum)

    def forward(
        self,
        int_feats: torch.Tensor,
        dense_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embeds and projects grouped discrete features into NS tokens.

        Args:
            int_feats: (B, total_int_dim), concatenated integer features.

        Returns:
            Tokens of shape (B, num_groups, D).
        """
        tokens = []
        for group, proj in zip(self.groups, self.group_projs):
            fid_embs = []
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    # Filtered high-cardinality feature: output zero vector
                    fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        # Single-value feature: direct lookup
                        fid_emb = emb_layer(int_feats[:, offset].long())  # (B, emb_dim)
                    else:
                        # Multi-value feature: lookup then mean pooling (ignoring padding=0)
                        vals = int_feats[:, offset:offset + length].long()  # (B, length)
                        emb_all = emb_layer(vals)  # (B, length, emb_dim)
                        fid_emb = self._pool_multi_value_embedding(
                            vals, emb_all, offset, dense_feats)
                fid_embs.append(fid_emb)
            cat_emb = torch.cat(fid_embs, dim=-1)  # (B, num_fids*emb_dim)
            tokens.append(F.silu(proj(cat_emb)).unsqueeze(1))  # (B, 1, D)
        return torch.cat(tokens, dim=1)  # (B, num_groups, D)


class RankMixerNSTokenizer(nn.Module):
    """NS Tokenizer following the RankMixer paper's approach.

    All group embedding vectors are concatenated into a single long vector,
    then equally split into num_ns_tokens segments, each projected to d_model.
    This allows num_ns_tokens to be chosen freely (independent of group count).
    """

    def __init__(
        self,
        feature_specs: List[Tuple[int, int, int]],
        groups: List[List[int]],
        emb_dim: int,
        d_model: int,
        num_ns_tokens: int,
        emb_skip_threshold: int = 0,
        pair_feature_specs: Optional[List[Tuple[int, int, int, int, int]]] = None,
        use_pair_62_66: bool = False,
        randomized_split: bool = False,
        randomized_split_seed: int = 42,
        split_name: str = "ns",
    ) -> None:
        """Initializes RankMixerNSTokenizer.

        Args:
            feature_specs: [(vocab_size, offset, length), ...] per feature.
            groups: List of feature index groups (defines semantic ordering).
            emb_dim: Embedding dimension per feature.
            d_model: Output token dimension.
            num_ns_tokens: Number of NS tokens to produce (T segments).
            emb_skip_threshold: Skip embedding for features with vocab > threshold.
        """
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.num_ns_tokens = num_ns_tokens
        self.emb_skip_threshold = emb_skip_threshold
        self.use_pair_62_66 = use_pair_62_66
        self.randomized_split = bool(randomized_split)
        self.randomized_split_seed = int(randomized_split_seed)
        self.split_name = split_name
        self._pair_by_int_offset = GroupNSTokenizer._build_pair_lookup(
            pair_feature_specs or [])
        self.last_pair_shapes: Dict[int, Tuple[int, ...]] = {}

        # One embedding table per fid (None if skipped by emb_skip_threshold
        # or if vocab_size <= 0 / no vocab info).
        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        # Map from fid index to position in self.embs (or -1 if filtered)
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)
        if self.use_pair_62_66:
            offset_to_fid_idx = {
                offset: i for i, (_, offset, _) in enumerate(feature_specs)
            }
            for int_offset, (fid, _, _, _) in self._pair_by_int_offset.items():
                fid_idx = offset_to_fid_idx[int_offset]
                if self._emb_index[fid_idx] == -1:
                    raise ValueError(
                        f"use_pair_62_66=True requires an embedding for fid {fid}; "
                        "increase emb_skip_threshold or disable the pair feature")

        if num_ns_tokens <= 0:
            raise ValueError(
                f"{split_name}: RankMixerNSTokenizer requires num_ns_tokens > 0, "
                f"got {num_ns_tokens}")

        flat_fid_indices = [fid_idx for group in groups for fid_idx in group]
        if not flat_fid_indices:
            raise ValueError(f"{split_name}: empty fid list for RankMixerNSTokenizer")
        self.flat_fid_indices = flat_fid_indices

        # Compute total embedding dim: sum of all fids across all groups
        total_num_fids = len(flat_fid_indices)
        total_emb_dim = total_num_fids * emb_dim

        if self.randomized_split:
            if num_ns_tokens > total_num_fids:
                raise ValueError(
                    f"{split_name}: randomized_split requires "
                    f"num_ns_tokens <= num_fids, got "
                    f"num_ns_tokens={num_ns_tokens}, num_fids={total_num_fids}")

            gen = torch.Generator()
            gen.manual_seed(self.randomized_split_seed)
            perm = torch.randperm(total_num_fids, generator=gen).tolist()
            shuffled = [flat_fid_indices[i] for i in perm]

            base = total_num_fids // num_ns_tokens
            rem = total_num_fids % num_ns_tokens
            token_fid_groups: List[List[int]] = []
            start = 0
            for token_idx in range(num_ns_tokens):
                cnt = base + (1 if token_idx < rem else 0)
                token_fid_groups.append(shuffled[start:start + cnt])
                start += cnt

            self.token_fid_groups = token_fid_groups
            self.token_projs = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(len(fid_group) * emb_dim, d_model),
                    nn.LayerNorm(d_model),
                )
                for fid_group in self.token_fid_groups
            ])
            self.chunk_dim = None
            self.padded_total_dim = total_emb_dim
            self._pad_size = 0

            logging.info(
                f"RankMixerNSTokenizer[{split_name}] RPS enabled: "
                f"seed={self.randomized_split_seed}, num_fids={total_num_fids}, "
                f"total_emb_dim={total_emb_dim}, num_ns_tokens={num_ns_tokens}, "
                f"group_sizes={[len(g) for g in self.token_fid_groups]}, "
                f"first_groups={self.token_fid_groups[:3]}"
            )
        else:
            self.token_fid_groups = None

            # Pad total_emb_dim to be divisible by num_ns_tokens
            self.chunk_dim = math.ceil(total_emb_dim / num_ns_tokens)
            self.padded_total_dim = self.chunk_dim * num_ns_tokens
            self._pad_size = self.padded_total_dim - total_emb_dim

            # Per-chunk projection: chunk_dim -> d_model with LayerNorm
            self.token_projs = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.chunk_dim, d_model),
                    nn.LayerNorm(d_model),
                )
                for _ in range(num_ns_tokens)
            ])

            logging.info(
                f"RankMixerNSTokenizer[{split_name}]: {total_num_fids} fids, "
                f"total_emb_dim={total_emb_dim}, chunk_dim={self.chunk_dim}, "
                f"num_ns_tokens={num_ns_tokens}, pad={self._pad_size}"
            )

    def _pool_multi_value_embedding(
        self,
        vals: torch.Tensor,
        emb_all: torch.Tensor,
        offset: int,
        dense_feats: Optional[torch.Tensor],
    ) -> torch.Tensor:
        mask = (vals != 0).to(emb_all.dtype).unsqueeze(-1)
        count = mask.sum(dim=1).clamp(min=1)
        mean_pool = (emb_all * mask).sum(dim=1) / count

        if not self.use_pair_62_66 or offset not in self._pair_by_int_offset:
            return mean_pool
        if dense_feats is None:
            raise ValueError("use_pair_62_66=True requires pair_dense_feats_62_66")

        fid, _, dense_offset, dense_len = self._pair_by_int_offset[offset]
        dense_vals = dense_feats[:, dense_offset:dense_offset + dense_len].to(emb_all.dtype)
        weights = torch.sqrt(torch.log1p(torch.clamp(dense_vals, min=0.0)))
        weights = weights.unsqueeze(-1) * mask
        weighted_sum = (emb_all * weights).sum(dim=1)
        all_zero_weight = weights.sum(dim=1) <= 0
        self.last_pair_shapes[fid] = tuple(weighted_sum.shape)
        return torch.where(all_zero_weight, mean_pool, weighted_sum)

    def _embed_one_fid(
        self,
        fid_idx: int,
        int_feats: torch.Tensor,
        dense_feats: Optional[torch.Tensor],
    ) -> torch.Tensor:
        _, offset, length = self.feature_specs[fid_idx]
        emb_real_idx = self._emb_index[fid_idx]
        if emb_real_idx == -1:
            return int_feats.new_zeros(
                (int_feats.shape[0], self.emb_dim),
                dtype=torch.float32,
            )

        emb_layer = self.embs[emb_real_idx]
        if length == 1:
            return emb_layer(int_feats[:, offset].long())

        vals = int_feats[:, offset:offset + length].long()
        emb_all = emb_layer(vals)
        return self._pool_multi_value_embedding(vals, emb_all, offset, dense_feats)

    def forward(
        self,
        int_feats: torch.Tensor,
        dense_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embeds all features, concatenates, splits, and projects.

        Args:
            int_feats: (B, total_int_dim) concatenated integer features.

        Returns:
            (B, num_ns_tokens, d_model) tensor.
        """
        if self.randomized_split:
            tokens = []
            for fid_group, proj in zip(self.token_fid_groups, self.token_projs):
                fid_embs = [
                    self._embed_one_fid(fid_idx, int_feats, dense_feats)
                    for fid_idx in fid_group
                ]
                cat_emb = torch.cat(fid_embs, dim=-1)
                tokens.append(F.silu(proj(cat_emb)).unsqueeze(1))
            return torch.cat(tokens, dim=1)

        # 1. Embed all fids in group order → flat cat
        all_embs = []
        for group in self.groups:
            for fid_idx in group:
                all_embs.append(
                    self._embed_one_fid(fid_idx, int_feats, dense_feats))

        cat_emb = torch.cat(all_embs, dim=-1)  # (B, total_emb_dim)

        # 2. Pad if needed
        if self._pad_size > 0:
            cat_emb = F.pad(cat_emb, (0, self._pad_size))  # (B, padded_total_dim)

        # 3. Split into num_ns_tokens chunks and project each
        chunks = cat_emb.split(self.chunk_dim, dim=-1)  # list of (B, chunk_dim)
        tokens = []
        for chunk, proj in zip(chunks, self.token_projs):
            tokens.append(F.silu(proj(chunk)).unsqueeze(1))  # (B, 1, d_model)

        return torch.cat(tokens, dim=1)  # (B, num_ns_tokens, d_model)


class NSTokenSEGate(nn.Module):
    """Squeeze-and-excitation style gate over NS tokens."""

    def __init__(self, d_model: int, num_tokens: int) -> None:
        super().__init__()
        hidden_dim = max(4, d_model // 4)
        self.excitation = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_tokens),
        )
        out = self.excitation[-1]
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)

    def forward(self, ns_tokens: torch.Tensor) -> torch.Tensor:
        pooled = ns_tokens.mean(dim=1)  # (B, D)
        gate = 2.0 * torch.sigmoid(self.excitation(pooled)).unsqueeze(-1)
        return ns_tokens * gate


class TargetAwareAttentionHead(nn.Module):
    """DIN-style target-aware sequence readout added as a gated residual."""

    def __init__(self, d_model: int, num_sequences: int, hidden_mult: int = 4) -> None:
        super().__init__()
        self.num_sequences = num_sequences
        self.attn_score = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )
        summary_dim = num_sequences * 4 * d_model
        hidden_dim = d_model * hidden_mult
        self.residual_mlp = nn.Sequential(
            nn.LayerNorm(summary_dim),
            nn.Linear(summary_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * d_model),
        )
        out = self.residual_mlp[-1]
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)

    @staticmethod
    def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = (~mask).unsqueeze(-1).to(tokens.dtype)
        denom = valid.sum(dim=1).clamp(min=1.0)
        return (tokens * valid).sum(dim=1) / denom

    @staticmethod
    def _last_valid_token(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, _, D = tokens.shape
        valid_len = (~mask).sum(dim=1)
        gather_idx = torch.clamp(valid_len - 1, min=0)
        gather_idx = gather_idx.view(B, 1, 1).expand(B, 1, D)
        last = torch.gather(tokens, dim=1, index=gather_idx).squeeze(1)
        has_valid = (valid_len > 0).to(tokens.dtype).unsqueeze(-1)
        return last * has_valid

    def _din_context(
        self,
        target_repr: torch.Tensor,
        seq_tokens: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor:
        target_exp = target_repr.unsqueeze(1).expand(-1, seq_tokens.shape[1], -1)
        attn_features = torch.cat(
            [
                target_exp,
                seq_tokens,
                target_exp - seq_tokens,
                target_exp * seq_tokens,
            ],
            dim=-1,
        )
        scores = self.attn_score(attn_features).squeeze(-1)
        valid = ~seq_mask
        scores = scores.masked_fill(~valid, -1e9)
        weights = torch.softmax(scores, dim=1) * valid.to(scores.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return (seq_tokens * weights.unsqueeze(-1)).sum(dim=1)

    def forward(
        self,
        head_repr: torch.Tensor,
        item_ns_tokens: torch.Tensor,
        seq_tokens_list: List[torch.Tensor],
        seq_masks_list: List[torch.Tensor],
    ) -> torch.Tensor:
        target_repr = item_ns_tokens.mean(dim=1)
        summaries = []
        for seq_tokens, seq_mask in zip(seq_tokens_list, seq_masks_list):
            seq_mean = self._masked_mean(seq_tokens, seq_mask)
            seq_last = self._last_valid_token(seq_tokens, seq_mask)
            din_context = self._din_context(target_repr, seq_tokens, seq_mask)
            summaries.append(torch.cat(
                [target_repr, seq_mean, seq_last, din_context], dim=-1))

        residual_raw, residual_gate = self.residual_mlp(
            torch.cat(summaries, dim=-1)).chunk(2, dim=-1)
        return head_repr + residual_raw * torch.sigmoid(residual_gate)


class PCVRHyFormer(nn.Module):
    """PCVRHyFormer model for post-click conversion rate prediction.

    Combines MultiSeqHyFormerBlock and MultiSeqQueryGenerator to process
    multiple input sequences with non-sequence features.
    """

    @staticmethod
    def _feature_specs_to_indices(
        feature_specs: Optional[List[Tuple[int, int, int]]],
        keep_fids: Tuple[int, ...],
    ) -> List[int]:
        if feature_specs is None:
            return []
        keep = set(keep_fids)
        indices: List[int] = []
        for fid, offset, length in feature_specs:
            if fid in keep:
                indices.extend(range(offset, offset + length))
        return indices

    @staticmethod
    def _parse_fids(fids: Union[str, List[int], Tuple[int, ...], None]) -> Tuple[int, ...]:
        if fids is None:
            return ()
        if isinstance(fids, str):
            fids = fids.strip()
            if not fids:
                return ()
            return tuple(int(part.strip()) for part in fids.split(',') if part.strip())
        return tuple(int(fid) for fid in fids)

    @staticmethod
    def _feature_specs_to_indices_required(
        feature_specs: Optional[List[Tuple[int, int, int]]],
        keep_fids: Tuple[int, ...],
        source_name: str,
    ) -> List[int]:
        if not keep_fids:
            raise ValueError(f"{source_name} dense cross fids must be non-empty")
        if len(set(keep_fids)) != len(keep_fids):
            raise ValueError(f"{source_name} dense cross fids contain duplicates: {keep_fids}")
        if feature_specs is None:
            raise ValueError(f"{source_name} dense feature specs are required")

        by_fid = {fid: (offset, length) for fid, offset, length in feature_specs}
        missing = [fid for fid in keep_fids if fid not in by_fid]
        if missing:
            raise KeyError(
                f"{source_name} dense cross fids missing from schema: {missing}")

        indices: List[int] = []
        for fid in keep_fids:
            offset, length = by_fid[fid]
            indices.extend(range(offset, offset + length))
        return indices

    @staticmethod
    def _feature_specs_to_index_groups(
        feature_specs: Optional[List[Tuple[int, int, int]]],
        keep_fids: Tuple[int, ...],
    ) -> List[Tuple[int, List[int]]]:
        if feature_specs is None:
            return []
        keep = set(keep_fids)
        groups: List[Tuple[int, List[int]]] = []
        for fid, offset, length in feature_specs:
            if fid in keep:
                groups.append((fid, list(range(offset, offset + length))))
        return groups

    @staticmethod
    def _feature_specs_excluding_indices(
        feature_specs: Optional[List[Tuple[int, int, int]]],
        exclude_fids: Tuple[int, ...],
    ) -> List[int]:
        if feature_specs is None:
            return []
        exclude = set(exclude_fids)
        indices: List[int] = []
        for fid, offset, length in feature_specs:
            if fid not in exclude:
                indices.extend(range(offset, offset + length))
        return indices

    @staticmethod
    def _complement_indices(total_dim: int, excluded_indices: List[int]) -> List[int]:
        excluded = set(excluded_indices)
        return [i for i in range(total_dim) if i not in excluded]

    @staticmethod
    def _make_index_tensor(indices: List[int]) -> torch.Tensor:
        return torch.tensor(indices, dtype=torch.long)

    def __init__(
        self,
        # Data schema
        user_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: "dict[str, List[int]]",  # {domain: [vocab_size_per_fid, ...]}
        # NS grouping config (grouped by fid index)
        user_ns_groups: List[List[int]],
        item_ns_groups: List[List[int]],
        # Model hyperparameters
        d_model: int = 64,
        emb_dim: int = 64,
        num_queries: int = 1,
        num_hyformer_blocks: int = 2,
        num_heads: int = 4,
        seq_encoder_type: str = 'transformer',
        hidden_mult: int = 4,
        dropout_rate: float = 0.01,
        seq_top_k: int = 50,
        seq_causal: bool = False,
        action_num: int = 1,
        num_time_buckets: int = 65,
        rank_mixer_mode: str = 'full',
        use_rope: bool = False,
        rope_base: float = 10000.0,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        # NS tokenizer variant
        ns_tokenizer_type: str = 'rankmixer',
        user_ns_tokens: int = 0,
        item_ns_tokens: int = 0,
        use_abs_time_emb: bool = False,
        user_abs_time_missing_as_padding: bool = False,
        use_pair_62_66: bool = False,
        add_user_time_to_dense_tok: bool = False,
        head_cross_layers: int = 0,
        head_cross_init_scale: float = 0.01,
        user_pair_feature_specs: Optional[List[Tuple[int, int, int, int, int]]] = None,
        user_dense_feature_specs: Optional[List[Tuple[int, int, int]]] = None,
        item_dense_feature_specs: Optional[List[Tuple[int, int, int]]] = None,
        user_dense_ue_fids: Tuple[int, ...] = USER_DENSE_UE_FIDS,
        use_dense_cross_token: bool = False,
        dense_cross_user_fids: Union[str, List[int], Tuple[int, ...]] = USER_DENSE_UE_FIDS,
        dense_cross_item_source: str = 'user_dense',
        dense_cross_item_fids: Union[str, List[int], Tuple[int, ...]] = (89, 90, 91),
        dense_cross_dim: int = 0,
        use_target_attention: bool = False,
        use_ns_se_gate: bool = False,
        use_ns_self_attn: bool = False,
        ns_self_attn_impl: str = 'mha',
        ns_autoint_layers: int = 2,
        ns_autoint_heads: int = 0,
        ns_autoint_gate_init: float = 0.05,
        ns_autoint_activation: str = 'relu',
        use_combined_autoint: bool = False,
        combined_autoint_layers: int = 1,
        combined_autoint_heads: int = 0,
        combined_autoint_dropout: float = -1.0,
        combined_autoint_gate_init: float = 0.0,
        combined_autoint_position: str = 'pre_mixer',
        randomized_split: bool = False,
        randomized_split_seed: int = 42,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.emb_dim = emb_dim
        self.action_num = action_num
        self.num_queries = num_queries
        self.seq_domains = sorted(seq_vocab_sizes.keys())  # deterministic order
        self.num_sequences = len(self.seq_domains)
        self.num_time_buckets = num_time_buckets
        self.rank_mixer_mode = rank_mixer_mode
        self.use_rope = use_rope
        self.emb_skip_threshold = emb_skip_threshold
        self.seq_id_threshold = seq_id_threshold
        self.ns_tokenizer_type = ns_tokenizer_type
        self.randomized_split = bool(randomized_split)
        self.randomized_split_seed = int(randomized_split_seed)
        if self.randomized_split and ns_tokenizer_type != 'rankmixer':
            raise ValueError(
                "randomized_split=True is only supported with "
                "ns_tokenizer_type='rankmixer'")
        self.use_abs_time_emb = use_abs_time_emb
        self.user_abs_time_missing_as_padding = bool(user_abs_time_missing_as_padding)
        self.use_pair_62_66 = use_pair_62_66
        self.add_user_time_to_dense_tok = add_user_time_to_dense_tok
        self.use_target_attention = use_target_attention
        self.use_ns_se_gate = use_ns_se_gate
        self.use_ns_self_attn = use_ns_self_attn
        self.ns_self_attn_impl = ns_self_attn_impl
        self.ns_autoint_layers = int(ns_autoint_layers)
        self.ns_autoint_heads = int(ns_autoint_heads)
        self.ns_autoint_gate_init = float(ns_autoint_gate_init)
        self.ns_autoint_activation = ns_autoint_activation
        self.use_combined_autoint = use_combined_autoint
        self.combined_autoint_layers = int(combined_autoint_layers)
        self.combined_autoint_heads = int(combined_autoint_heads)
        self.combined_autoint_dropout = float(combined_autoint_dropout)
        self.combined_autoint_gate_init = float(combined_autoint_gate_init)
        self.combined_autoint_position = combined_autoint_position
        if self.ns_self_attn_impl not in ('mha', 'autoint'):
            raise ValueError(
                "ns_self_attn_impl must be 'mha' or 'autoint', "
                f"got {self.ns_self_attn_impl!r}")
        resolved_ns_autoint_heads = (
            self.ns_autoint_heads if self.ns_autoint_heads > 0 else num_heads
        )
        if self.ns_self_attn_impl == 'autoint':
            if self.ns_autoint_layers < 1:
                raise ValueError("ns_autoint_layers must be >= 1")
            if resolved_ns_autoint_heads < 1:
                raise ValueError("ns_autoint_heads must resolve to >= 1")
            if d_model % resolved_ns_autoint_heads != 0:
                raise ValueError(
                    f"d_model={d_model} must be divisible by NS AutoInt "
                    f"heads={resolved_ns_autoint_heads}")
        resolved_combined_autoint_heads = (
            self.combined_autoint_heads if self.combined_autoint_heads > 0 else num_heads
        )
        resolved_combined_autoint_dropout = (
            dropout_rate if self.combined_autoint_dropout < 0 else self.combined_autoint_dropout
        )
        combined_autoint_dropout_arg = (
            None if self.combined_autoint_dropout < 0 else self.combined_autoint_dropout
        )
        if self.use_combined_autoint:
            if self.combined_autoint_layers < 1:
                raise ValueError("combined_autoint_layers must be >= 1 when enabled")
            if resolved_combined_autoint_heads < 1:
                raise ValueError(
                    "combined_autoint_heads must resolve to >= 1 when enabled")
            if d_model % resolved_combined_autoint_heads != 0:
                raise ValueError(
                    f"d_model={d_model} must be divisible by combined AutoInt "
                    f"heads={resolved_combined_autoint_heads}")
            if self.combined_autoint_position not in ("pre_mixer", "post_mixer"):
                raise ValueError(
                    "combined_autoint_position must be 'pre_mixer' or 'post_mixer', "
                    f"got {self.combined_autoint_position!r}")
        self.head_cross_layers = int(head_cross_layers)
        self.head_cross_init_scale = float(head_cross_init_scale)
        if self.head_cross_layers < 0:
            raise ValueError("head_cross_layers must be non-negative")
        self.use_dense_cross_token = bool(use_dense_cross_token)
        self.dense_cross_user_fids = self._parse_fids(dense_cross_user_fids)
        self.dense_cross_item_fids = self._parse_fids(dense_cross_item_fids)
        self.dense_cross_item_source = dense_cross_item_source
        if self.dense_cross_item_source not in DENSE_CROSS_ITEM_SOURCES:
            raise ValueError(
                "dense_cross_item_source must be one of "
                f"{DENSE_CROSS_ITEM_SOURCES}, got {self.dense_cross_item_source!r}")
        self.dense_cross_dim = int(dense_cross_dim)
        if self.dense_cross_dim < 0:
            raise ValueError("dense_cross_dim must be >= 0")
        resolved_dense_cross_dim = (
            d_model if self.dense_cross_dim == 0 else self.dense_cross_dim
        )
        self.user_dense_ue_fids = tuple(user_dense_ue_fids)
        self.user_dense_fusion_fids = USER_DENSE_RAW_FUSION_FIDS
        self._has_user_dense_feature_specs = user_dense_feature_specs is not None
        if use_pair_62_66 and user_dense_feature_specs is None:
            raise ValueError(
                "use_pair_62_66=True requires user_dense_feature_specs to split "
                "pair weights and per-fid dense heads")
        if use_pair_62_66 and not user_pair_feature_specs:
            raise ValueError(
                "use_pair_62_66=True requires pair specs for fids 62-66")

        raw_pair_specs = user_pair_feature_specs or []
        user_pair_dense_indices: List[int] = []
        compact_pair_specs: List[Tuple[int, int, int, int, int]] = []
        pair_dense_offset = 0
        for fid, int_offset, int_len, dense_offset, dense_len in raw_pair_specs:
            if fid not in PAIR_62_66_FIDS:
                raise ValueError(f"pair feature fid={fid} is not in 62-66")
            user_pair_dense_indices.extend(range(dense_offset, dense_offset + dense_len))
            compact_pair_specs.append((fid, int_offset, int_len, pair_dense_offset, dense_len))
            pair_dense_offset += dense_len

        dense_head_specs = (
            self._feature_specs_to_index_groups(
                user_dense_feature_specs, self.user_dense_fusion_fids)
            if use_pair_62_66 else []
        )
        activity_dense_indices = (
            self._feature_specs_to_indices(
                user_dense_feature_specs, USER_ACTIVITY_DENSE_FIDS)
            if use_pair_62_66 else []
        )
        if use_pair_62_66:
            dense_head_fids = tuple(fid for fid, _ in dense_head_specs)
            missing_dense_fids = [
                fid for fid in self.user_dense_fusion_fids
                if fid not in dense_head_fids
            ]
            if missing_dense_fids:
                raise KeyError(
                    "use_pair_62_66=True requires dense fids "
                    f"{missing_dense_fids} for user_dense_tok fusion")
            present_activity_fids = {
                fid for fid, _, _ in (user_dense_feature_specs or [])
                if fid in USER_ACTIVITY_DENSE_FIDS
            }
            if present_activity_fids:
                expected_activity_fids = set(USER_ACTIVITY_DENSE_FIDS)
                if (
                    present_activity_fids != expected_activity_fids
                    or len(activity_dense_indices) != len(USER_ACTIVITY_DENSE_FIDS)
                ):
                    raise ValueError(
                        "activity dense fids must be all-or-none with one "
                        "dimension each")
                dense_head_specs = dense_head_specs + [
                    (USER_ACTIVITY_DENSE_FIDS[0], activity_dense_indices)
                ]

        dense_branch_indices = [] if use_pair_62_66 else list(range(user_dense_dim))
        dense_cross_user_indices: List[int] = []
        dense_cross_item_indices: List[int] = []
        dense_cross_user_input_dim = 0
        dense_cross_item_input_dim = 0
        if self.use_dense_cross_token:
            dense_cross_user_indices = self._feature_specs_to_indices_required(
                user_dense_feature_specs,
                self.dense_cross_user_fids,
                'user_dense',
            )
            dense_cross_user_input_dim = len(dense_cross_user_indices)
            if self.dense_cross_item_source == 'user_dense':
                dense_cross_item_indices = self._feature_specs_to_indices_required(
                    user_dense_feature_specs,
                    self.dense_cross_item_fids,
                    'user_dense item-side',
                )
                dense_cross_item_input_dim = len(dense_cross_item_indices)
            elif self.dense_cross_item_source == 'item_dense':
                dense_cross_item_indices = self._feature_specs_to_indices_required(
                    item_dense_feature_specs,
                    self.dense_cross_item_fids,
                    'item_dense',
                )
                dense_cross_item_input_dim = len(dense_cross_item_indices)
            else:
                dense_cross_item_input_dim = d_model

        self.register_buffer(
            '_user_pair_dense_index',
            self._make_index_tensor(user_pair_dense_indices if use_pair_62_66 else []),
            persistent=False,
        )
        self.register_buffer(
            '_user_dense_branch_index',
            self._make_index_tensor(dense_branch_indices),
            persistent=False,
        )
        self.register_buffer(
            '_dense_cross_user_index',
            self._make_index_tensor(dense_cross_user_indices),
            persistent=False,
        )
        self.register_buffer(
            '_dense_cross_item_index',
            self._make_index_tensor(dense_cross_item_indices),
            persistent=False,
        )
        self.user_dense_branch_dim = len(dense_branch_indices)
        self.dense_cross_user_input_dim = dense_cross_user_input_dim
        self.dense_cross_item_input_dim = dense_cross_item_input_dim
        self.resolved_dense_cross_dim = resolved_dense_cross_dim
        self.user_dense_head_fids = [fid for fid, _ in dense_head_specs]
        self.user_dense_head_dims = [len(indices) for _, indices in dense_head_specs]
        self._user_dense_head_index_names: List[str] = []
        for i, (_, indices) in enumerate(dense_head_specs):
            name = f'_user_dense_head_index_{i}'
            self.register_buffer(
                name,
                self._make_index_tensor(indices),
                persistent=False,
            )
            self._user_dense_head_index_names.append(name)

        # ================== NS Tokens Construction ==================

        if ns_tokenizer_type == 'group':
            # Original: one NS token per group
            self.user_ns_tokenizer = GroupNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
                pair_feature_specs=compact_pair_specs,
                use_pair_62_66=use_pair_62_66,
            )
            num_user_ns = len(user_ns_groups)

            self.item_ns_tokenizer = GroupNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_item_ns = len(item_ns_groups)
        elif ns_tokenizer_type == 'rankmixer':
            # RankMixer paper style: all embeddings cat → split → project
            # 0 means auto: fall back to group count
            if user_ns_tokens <= 0:
                user_ns_tokens = len(user_ns_groups)
            if item_ns_tokens <= 0:
                item_ns_tokens = len(item_ns_groups)
            self.user_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=user_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
                pair_feature_specs=compact_pair_specs,
                use_pair_62_66=use_pair_62_66,
                randomized_split=self.randomized_split,
                randomized_split_seed=self.randomized_split_seed,
                split_name="user",
            )
            num_user_ns = user_ns_tokens

            self.item_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=item_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
                randomized_split=self.randomized_split,
                randomized_split_seed=self.randomized_split_seed + 100003,
                split_name="item",
            )
            num_item_ns = item_ns_tokens
        else:
            raise ValueError(f"Unknown ns_tokenizer_type: {ns_tokenizer_type}")

        # User dense feature projection: baseline concat path when pair pooling
        # is off; per-fid heads fused into one dense token when pair pooling is on.
        self.has_user_dense = (
            len(self.user_dense_head_dims) > 0
            if use_pair_62_66 else self.user_dense_branch_dim > 0
        )
        if self.has_user_dense and use_pair_62_66:
            self.user_dense_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(head_dim, d_model),
                    nn.LayerNorm(d_model),
                )
                for head_dim in self.user_dense_head_dims
            ])
            self.user_dense_fusion = nn.Sequential(
                nn.Linear(len(self.user_dense_head_dims) * d_model, d_model),
                nn.LayerNorm(d_model),
            )
        elif self.has_user_dense:
            self.user_dense_proj = nn.Sequential(
                nn.Linear(self.user_dense_branch_dim, d_model),
                nn.LayerNorm(d_model),
            )

        # Item dense feature projection (if available)
        self.has_item_dense = item_dense_dim > 0
        if self.has_item_dense:
            self.item_dense_proj = nn.Sequential(
                nn.Linear(item_dense_dim, d_model),
                nn.LayerNorm(d_model),
            )

        if self.use_dense_cross_token:
            if self.dense_cross_item_source == 'item_ns' and num_item_ns <= 0:
                raise ValueError(
                    "dense_cross_item_source='item_ns' requires at least one item NS token")
            self.dense_cross_user_proj = nn.Sequential(
                nn.Linear(self.dense_cross_user_input_dim, self.resolved_dense_cross_dim),
                nn.LayerNorm(self.resolved_dense_cross_dim),
            )
            self.dense_cross_item_proj = nn.Sequential(
                nn.Linear(self.dense_cross_item_input_dim, self.resolved_dense_cross_dim),
                nn.LayerNorm(self.resolved_dense_cross_dim),
            )
            self.dense_cross_proj = nn.Sequential(
                nn.Linear(self.resolved_dense_cross_dim, d_model),
                nn.LayerNorm(d_model),
            )

        # Total NS token count
        self.num_user_ns = num_user_ns
        self.num_item_ns = num_item_ns
        self.num_ns = (num_user_ns + (1 if self.has_user_dense else 0) + num_item_ns
                       + (1 if self.has_item_dense else 0)
                       + (1 if self.use_dense_cross_token else 0))
        self.item_ns_start = num_user_ns + (1 if self.has_user_dense else 0)
        self.item_ns_end = self.item_ns_start + num_item_ns
        if use_target_attention and num_item_ns <= 0:
            raise ValueError("use_target_attention=True requires at least one item NS token")

        # ================== Check d_model % T == 0 constraint (full mode only) ==================
        T = num_queries * self.num_sequences + self.num_ns
        if rank_mixer_mode == 'full' and d_model % T != 0:
            valid_T_values = [t for t in range(1, d_model + 1) if d_model % t == 0]
            raise ValueError(
                f"d_model={d_model} must be divisible by T=num_queries*num_sequences+num_ns="
                f"{num_queries}*{self.num_sequences}+{self.num_ns}={T}. "
                f"Valid T values for d_model={d_model}: {valid_T_values}"
            )

        # ================== Seq Tokens Embedding ==================
        # seq_id_threshold decides which features inside the seq tokenizer are
        # treated as id features (they receive extra dropout). It is fully
        # independent of emb_skip_threshold (which skips Embedding creation).
        self.seq_id_emb_dropout = nn.Dropout(dropout_rate * 2)

        def _make_seq_embs(vocab_sizes):
            """Create embedding list, returning None for features skipped via
            emb_skip_threshold or with no vocab info (vs<=0)."""
            embs_raw = []
            for vs in vocab_sizes:
                skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
                if skip:
                    embs_raw.append(None)
                else:
                    embs_raw.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
            module_list = nn.ModuleList([e for e in embs_raw if e is not None])
            # Map from position index to real index in module_list (-1 if skipped)
            index_map = []
            real_idx = 0
            for e in embs_raw:
                if e is not None:
                    index_map.append(real_idx)
                    real_idx += 1
                else:
                    index_map.append(-1)
            is_id = [int(vs) > seq_id_threshold for vs in vocab_sizes]
            return module_list, index_map, is_id

        # ================== Dynamic Sequence Embeddings ==================
        self._seq_embs = nn.ModuleDict()
        self._seq_emb_index = {}    # domain -> index_map
        self._seq_is_id = {}        # domain -> is_id list
        self._seq_vocab_sizes = {}  # domain -> vocab_sizes list
        self._seq_proj = nn.ModuleDict()

        for domain in self.seq_domains:
            vs = seq_vocab_sizes[domain]
            embs, idx_map, is_id = _make_seq_embs(vs)
            self._seq_embs[domain] = embs
            self._seq_emb_index[domain] = idx_map
            self._seq_is_id[domain] = is_id
            self._seq_vocab_sizes[domain] = vs
            self._seq_proj[domain] = nn.Sequential(
                nn.Linear(len(vs) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )

        # ================== Time Interval Bucket Embedding (optional) ==================
        if num_time_buckets > 0:
            self.time_embedding = nn.Embedding(num_time_buckets, d_model, padding_idx=0)

        # ================== Absolute Time Embeddings (optional) ==================
        if use_abs_time_emb:
            if self.user_abs_time_missing_as_padding:
                self.user_hour_emb = nn.Embedding(25, d_model, padding_idx=0)
                self.user_dow_emb = nn.Embedding(8, d_model, padding_idx=0)
            else:
                self.user_hour_emb = nn.Embedding(24, d_model)
                self.user_dow_emb = nn.Embedding(7, d_model)
            self.seq_hour_emb = nn.Embedding(25, d_model, padding_idx=0)
            self.seq_dow_emb = nn.Embedding(8, d_model, padding_idx=0)

        # ================== HyFormer Components ==================
        # MultiSeqQueryGenerator
        self.query_generator = MultiSeqQueryGenerator(
            d_model=d_model,
            num_ns=self.num_ns,
            num_queries=num_queries,
            num_sequences=self.num_sequences,
            hidden_mult=hidden_mult,
        )

        # MultiSeqHyFormerBlock stack
        self.blocks = nn.ModuleList([
            MultiSeqHyFormerBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_queries=num_queries,
                num_ns=self.num_ns,
                num_sequences=self.num_sequences,
                seq_encoder_type=seq_encoder_type,
                hidden_mult=hidden_mult,
                dropout=dropout_rate,
                top_k=seq_top_k,
                causal=seq_causal,
                rank_mixer_mode=rank_mixer_mode,
                use_ns_self_attn=use_ns_self_attn,
                ns_self_attn_impl=ns_self_attn_impl,
                ns_autoint_layers=ns_autoint_layers,
                ns_autoint_heads=ns_autoint_heads,
                ns_autoint_gate_init=ns_autoint_gate_init,
                ns_autoint_activation=ns_autoint_activation,
                use_combined_autoint=use_combined_autoint,
                combined_autoint_layers=combined_autoint_layers,
                combined_autoint_heads=combined_autoint_heads,
                combined_autoint_dropout=combined_autoint_dropout_arg,
                combined_autoint_gate_init=combined_autoint_gate_init,
                combined_autoint_position=combined_autoint_position,
            )
            for _ in range(num_hyformer_blocks)
        ])
        logging.info(
            f"ns_self_attn={use_ns_self_attn}, "
            f"impl={ns_self_attn_impl}, "
            f"autoint_layers={ns_autoint_layers}, "
            f"autoint_heads={resolved_ns_autoint_heads}, "
            f"autoint_gate_init={ns_autoint_gate_init}, "
            f"autoint_activation={ns_autoint_activation}"
        )
        logging.info(
            f"combined_autoint={use_combined_autoint}, "
            f"layers={combined_autoint_layers}, "
            f"heads={resolved_combined_autoint_heads}, "
            f"dropout={resolved_combined_autoint_dropout}, "
            f"gate_init={combined_autoint_gate_init}, "
            f"position={combined_autoint_position}"
        )

        # ================== RoPE ==================
        if use_rope:
            head_dim = d_model // num_heads
            self.rotary_emb = RotaryEmbedding(dim=head_dim, base=rope_base)
        else:
            self.rotary_emb = None

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(num_queries * self.num_sequences * d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.head_cross_net = (
            HeadCrossNet(d_model, self.head_cross_layers, self.head_cross_init_scale)
            if self.head_cross_layers > 0 else None
        )

        # Dropout
        self.emb_dropout = nn.Dropout(dropout_rate)

        # Optional ablation modules. Both are identity-initialized via zeroed
        # output projections inside their constructors.
        if use_ns_se_gate:
            self.ns_se_gate = NSTokenSEGate(
                d_model=d_model,
                num_tokens=self.num_ns,
            )
        if use_target_attention:
            self.target_attention_head = TargetAwareAttentionHead(
                d_model=d_model,
                num_sequences=self.num_sequences,
                hidden_mult=hidden_mult,
            )

        # Classifier
        self.clsfier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, action_num)
        )

        # Initialize parameters
        self._init_params()

        # Log emb_skip_threshold filtering stats
        if emb_skip_threshold > 0:
            def _count_filtered(vocab_sizes, emb_index):
                filtered = sum(1 for idx in emb_index if idx == -1)
                return filtered, len(vocab_sizes)
            for domain in self.seq_domains:
                f, t = _count_filtered(self._seq_vocab_sizes[domain], self._seq_emb_index[domain])
                if f > 0:
                    logging.info(f"emb_skip_threshold={emb_skip_threshold}: {domain} skipped {f}/{t} features")
            for name, tokenizer in [
                ("user_ns", self.user_ns_tokenizer),
                ("item_ns", self.item_ns_tokenizer),
            ]:
                f = sum(1 for idx in tokenizer._emb_index if idx == -1)
                t = len(tokenizer._emb_index)
                if f > 0:
                    logging.info(f"emb_skip_threshold={emb_skip_threshold}: {name} skipped {f}/{t} features")

    def _init_params(self) -> None:
        """Applies Xavier initialization to all embedding weights."""
        for domain in self.seq_domains:
            for emb in self._seq_embs[domain]:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        for tokenizer in [self.user_ns_tokenizer, self.item_ns_tokenizer]:
            for emb in tokenizer.embs:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        if self.num_time_buckets > 0:
            nn.init.xavier_normal_(self.time_embedding.weight.data)
            self.time_embedding.weight.data[0, :] = 0

        if self.use_abs_time_emb:
            for emb in [self.user_hour_emb, self.user_dow_emb]:
                nn.init.xavier_normal_(emb.weight.data)
                if self.user_abs_time_missing_as_padding:
                    emb.weight.data[0, :] = 0
            for emb in [self.seq_hour_emb, self.seq_dow_emb]:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

    def reinit_high_cardinality_params(
        self, cardinality_threshold: int = 10000
    ) -> "set[int]":
        """Reinitializes only high-cardinality embeddings.

        Preserves low-cardinality and time feature embeddings.

        Args:
            cardinality_threshold: Only embeddings with vocab_size exceeding
                this value are reinitialized.

        Returns:
            A set of data_ptr() values for reinitialized parameters.
        """
        reinit_count = 0
        skip_count = 0
        reinit_ptrs = set()

        for emb_list, vocab_sizes, emb_index in [
            (self._seq_embs[d], self._seq_vocab_sizes[d], self._seq_emb_index[d])
            for d in self.seq_domains
        ]:
            for i, vs in enumerate(vocab_sizes):
                real_idx = emb_index[i]
                if real_idx == -1:
                    # Skipped by emb_skip_threshold, no embedding to reinit
                    continue
                emb = emb_list[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        for tokenizer, specs in [
            (self.user_ns_tokenizer, self.user_ns_tokenizer.feature_specs),
            (self.item_ns_tokenizer, self.item_ns_tokenizer.feature_specs),
        ]:
            for i, (vs, offset, length) in enumerate(specs):
                real_idx = tokenizer._emb_index[i]
                if real_idx == -1:
                    continue
                emb = tokenizer.embs[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        # time_embedding and absolute time embeddings are always preserved
        if self.num_time_buckets > 0:
            skip_count += 1
        if self.use_abs_time_emb:
            skip_count += 4

        logging.info(f"Re-initialized {reinit_count} high-cardinality Embeddings "
                     f"(vocab>{cardinality_threshold}), kept {skip_count}")
        return reinit_ptrs

    def get_sparse_params(self) -> List[nn.Parameter]:
        """Returns all embedding table parameters (optimized with Adagrad)."""
        sparse_params = set()
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                sparse_params.add(module.weight.data_ptr())
        return [p for p in self.parameters() if p.data_ptr() in sparse_params]

    def get_dense_params(self) -> List[nn.Parameter]:
        """Returns all non-embedding parameters (optimized with AdamW)."""
        sparse_ptrs = {p.data_ptr() for p in self.get_sparse_params()}
        return [p for p in self.parameters() if p.data_ptr() not in sparse_ptrs]

    def _embed_seq_domain(
        self,
        seq: torch.Tensor,
        sideinfo_embs: nn.ModuleList,
        proj: nn.Module,
        is_id: List[bool],
        emb_index: List[int],
        time_bucket_ids: torch.Tensor,
        hour_ids: Optional[torch.Tensor] = None,
        dayofweek_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embeds a sequence domain by concatenating sideinfo embeddings and projecting to d_model."""
        B, S, L = seq.shape
        emb_list = []
        for i in range(S):
            real_idx = emb_index[i] if i < len(emb_index) else -1
            if real_idx == -1:
                # Feature skipped by emb_skip_threshold: output zero vector
                emb_list.append(seq.new_zeros(B, L, self.emb_dim, dtype=torch.float))
            else:
                emb = sideinfo_embs[real_idx]
                e = emb(seq[:, i, :])  # (B, L, emb_dim)
                if is_id[i] and self.training:
                    e = self.seq_id_emb_dropout(e)
                emb_list.append(e)
        cat_emb = torch.cat(emb_list, dim=-1)  # (B, L, S*emb_dim)
        token_emb = F.gelu(proj(cat_emb))  # (B, L, D)

        # Add time bucket embedding (all-zero ids produce zero vectors via padding_idx=0)
        if self.num_time_buckets > 0:
            token_emb = token_emb + self.time_embedding(time_bucket_ids)

        if self.use_abs_time_emb:
            if hour_ids is None or dayofweek_ids is None:
                raise ValueError("use_abs_time_emb=True requires sequence hour/dayofweek tensors")
            hour_idx = hour_ids.long()
            dow_idx = dayofweek_ids.long()
            self._validate_seq_abs_time_indices(hour_idx, dow_idx)
            valid_mask = (hour_idx != 0) & (dow_idx != 0)
            seq_time_emb = self.seq_hour_emb(hour_idx) + self.seq_dow_emb(dow_idx)
            seq_time_emb = seq_time_emb * valid_mask.unsqueeze(-1).to(seq_time_emb.dtype)
            token_emb = token_emb + seq_time_emb

        return token_emb

    def _validate_abs_time_ids(
        self,
        hour_ids: torch.Tensor,
        dayofweek_ids: torch.Tensor,
    ) -> None:
        """Fail fast if absolute time ids are outside their embedding ranges."""
        max_hour = 24 if self.user_abs_time_missing_as_padding else 23
        max_dow = 7 if self.user_abs_time_missing_as_padding else 6
        if bool(((hour_ids < 0) | (hour_ids > max_hour)).any()):
            raise ValueError(f"hour ids must be in [0, {max_hour}]")
        if bool(((dayofweek_ids < 0) | (dayofweek_ids > max_dow)).any()):
            raise ValueError(f"dayofweek ids must be in [0, {max_dow}]")
        if self.user_abs_time_missing_as_padding:
            if bool(((hour_ids == 0) != (dayofweek_ids == 0)).any()):
                raise ValueError("user hour/dayofweek padding indices must match")

    @staticmethod
    def _validate_seq_abs_time_indices(
        hour_idx: torch.Tensor,
        dayofweek_idx: torch.Tensor,
    ) -> None:
        """Validate shifted sequence time ids: 0=padding, valid ranges are 1..24 / 1..7."""
        if bool(((hour_idx < 0) | (hour_idx > 24)).any()):
            raise ValueError("sequence hour indices must be in [0, 24]")
        if bool(((dayofweek_idx < 0) | (dayofweek_idx > 7)).any()):
            raise ValueError("sequence dayofweek indices must be in [0, 7]")
        if bool(((hour_idx == 0) != (dayofweek_idx == 0)).any()):
            raise ValueError("sequence hour/dayofweek padding indices must match")

    def _user_abs_time_embedding(self, inputs: ModelInput) -> torch.Tensor:
        if inputs.user_hour is None or inputs.user_dayofweek is None:
            raise ValueError("use_abs_time_emb=True requires user_hour/user_dayofweek")
        self._validate_abs_time_ids(inputs.user_hour, inputs.user_dayofweek)
        return (
            self.user_hour_emb(inputs.user_hour.long())
            + self.user_dow_emb(inputs.user_dayofweek.long())
        )

    def _select_dense_columns(
        self,
        feats: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        if index.numel() == 0:
            return feats.new_zeros(feats.shape[0], 0)
        return feats.index_select(1, index)

    def _select_user_dense_columns(
        self,
        feats: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        return self._select_dense_columns(feats, index)

    def _split_user_dense_feats(self, user_dense_feats: torch.Tensor) -> UserDenseSplit:
        if not self.use_pair_62_66:
            empty = user_dense_feats.new_zeros(user_dense_feats.shape[0], 0)
            return UserDenseSplit(
                dense_branch_feats=user_dense_feats,
                pair_dense_feats_62_66=empty,
                dense_head_feats=(),
            )

        pair_dense_feats_62_66 = self._select_user_dense_columns(
            user_dense_feats, self._user_pair_dense_index)
        dense_head_feats = tuple(
            self._select_user_dense_columns(user_dense_feats, getattr(self, name))
            for name in self._user_dense_head_index_names
        )
        empty = user_dense_feats.new_zeros(user_dense_feats.shape[0], 0)
        return UserDenseSplit(
            dense_branch_feats=empty,
            pair_dense_feats_62_66=pair_dense_feats_62_66,
            dense_head_feats=dense_head_feats,
        )

    def _make_user_dense_token(
        self,
        user_dense_split: UserDenseSplit,
        user_time_emb: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not self.has_user_dense:
            return None

        if self.use_pair_62_66:
            head_tokens = [
                F.silu(head(feats.float()))
                for head, feats in zip(self.user_dense_heads, user_dense_split.dense_head_feats)
            ]
            dense_tok = F.silu(
                self.user_dense_fusion(torch.cat(head_tokens, dim=-1))
            ).unsqueeze(1)
        else:
            dense_tok = F.silu(
                self.user_dense_proj(user_dense_split.dense_branch_feats)
            ).unsqueeze(1)

        if self.add_user_time_to_dense_tok and user_time_emb is not None:
            dense_tok = dense_tok + user_time_emb.unsqueeze(1)
        return dense_tok

    def _make_dense_cross_token(
        self,
        inputs: ModelInput,
        item_ns: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if not self.use_dense_cross_token:
            return None

        user_vec = self._select_dense_columns(
            inputs.user_dense_feats, self._dense_cross_user_index)
        if self.dense_cross_item_source == 'user_dense':
            item_vec = self._select_dense_columns(
                inputs.user_dense_feats, self._dense_cross_item_index)
        elif self.dense_cross_item_source == 'item_dense':
            item_vec = self._select_dense_columns(
                inputs.item_dense_feats, self._dense_cross_item_index)
        else:
            item_vec = item_ns.mean(dim=1)

        user_z = self.dense_cross_user_proj(user_vec.float())
        item_z = self.dense_cross_item_proj(item_vec.float())
        cross_z = user_z * item_z
        return F.silu(self.dense_cross_proj(cross_z)).unsqueeze(1)

    def _make_padding_mask(
        self, seq_len: torch.Tensor, max_len: int
    ) -> torch.Tensor:
        """Generates a padding mask from sequence lengths."""
        device = seq_len.device
        idx = torch.arange(max_len, device=device).unsqueeze(0)  # (1, max_len)
        return idx >= seq_len.unsqueeze(1)  # (B, max_len)

    def _build_ns_tokens(self, inputs: ModelInput) -> torch.Tensor:
        user_dense_split = self._split_user_dense_feats(inputs.user_dense_feats)
        user_ns = self.user_ns_tokenizer(
            inputs.user_int_feats,
            user_dense_split.pair_dense_feats_62_66 if self.use_pair_62_66 else None,
        )
        item_ns = self.item_ns_tokenizer(inputs.item_int_feats)

        user_time_emb = None
        if self.use_abs_time_emb:
            user_time_emb = self._user_abs_time_embedding(inputs)
            user_ns = user_ns + user_time_emb.unsqueeze(1)

        ns_parts = [user_ns]
        user_dense_tok = self._make_user_dense_token(user_dense_split, user_time_emb)
        if user_dense_tok is not None:
            ns_parts.append(user_dense_tok)
        ns_parts.append(item_ns)
        if self.has_item_dense:
            item_dense_tok = F.silu(
                self.item_dense_proj(inputs.item_dense_feats)).unsqueeze(1)
            ns_parts.append(item_dense_tok)
        dense_cross_tok = self._make_dense_cross_token(inputs, item_ns)
        if dense_cross_tok is not None:
            ns_parts.append(dense_cross_tok)

        ns_tokens = torch.cat(ns_parts, dim=1)
        if self.use_ns_se_gate:
            ns_tokens = self.ns_se_gate(ns_tokens)
        return ns_tokens

    def _build_sequence_tokens(
        self,
        inputs: ModelInput,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        seq_tokens_list = []
        seq_masks_list = []
        for domain in self.seq_domains:
            tokens = self._embed_seq_domain(
                inputs.seq_data[domain],
                self._seq_embs[domain],
                self._seq_proj[domain],
                self._seq_is_id[domain],
                self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain],
                inputs.seq_hours[domain] if inputs.seq_hours is not None else None,
                inputs.seq_dayofweeks[domain] if inputs.seq_dayofweeks is not None else None,
            )
            seq_tokens_list.append(tokens)
            mask = self._make_padding_mask(
                inputs.seq_lens[domain],
                inputs.seq_data[domain].shape[2],
            )
            seq_masks_list.append(mask)
        return seq_tokens_list, seq_masks_list

    def _run_multi_seq_blocks(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_masks_list: list,
        apply_dropout: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, list, list]:
        """Runs the multi-sequence block stack with dropout and output projection."""
        if apply_dropout:
            q_tokens_list = [self.emb_dropout(q) for q in q_tokens_list]
            ns_tokens = self.emb_dropout(ns_tokens)
            seq_tokens_list = [self.emb_dropout(s) for s in seq_tokens_list]

        curr_qs = q_tokens_list
        curr_ns = ns_tokens
        curr_seqs = seq_tokens_list
        curr_masks = seq_masks_list

        for block in self.blocks:
            # Precompute RoPE cos/sin for each sequence
            rope_cos_list = None
            rope_sin_list = None
            if self.rotary_emb is not None:
                rope_cos_list = []
                rope_sin_list = []
                device = curr_seqs[0].device
                for seq_i in curr_seqs:
                    seq_len = seq_i.shape[1]
                    cos, sin = self.rotary_emb(seq_len, device)
                    rope_cos_list.append(cos)
                    rope_sin_list.append(sin)

            curr_qs, curr_ns, curr_seqs, curr_masks = block(
                q_tokens_list=curr_qs,
                ns_tokens=curr_ns,
                seq_tokens_list=curr_seqs,
                seq_padding_masks=curr_masks,
                rope_cos_list=rope_cos_list,
                rope_sin_list=rope_sin_list,
            )

        # Output: concatenate all sequences' Q tokens then project via MLP
        B = curr_qs[0].shape[0]
        all_q = torch.cat(curr_qs, dim=1)  # (B, Nq*S, D)
        output = all_q.view(B, -1)  # (B, Nq*S*D)
        output = self.output_proj(output)  # (B, D)
        if self.head_cross_net is not None:
            output = self.head_cross_net(output)

        return output, curr_ns, curr_seqs, curr_masks

    def _apply_target_attention(
        self,
        output: torch.Tensor,
        ns_tokens: torch.Tensor,
        seq_tokens_list: List[torch.Tensor],
        seq_masks_list: List[torch.Tensor],
    ) -> torch.Tensor:
        if not self.use_target_attention:
            return output
        item_ns_tokens = ns_tokens[:, self.item_ns_start:self.item_ns_end, :]
        return self.target_attention_head(
            output,
            item_ns_tokens,
            seq_tokens_list,
            seq_masks_list,
        )

    def _forward_impl(
        self,
        inputs: ModelInput,
        apply_dropout: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ns_tokens = self._build_ns_tokens(inputs)
        seq_tokens_list, seq_masks_list = self._build_sequence_tokens(inputs)
        q_tokens_list = self.query_generator(ns_tokens, seq_tokens_list, seq_masks_list)
        output, curr_ns, curr_seqs, curr_masks = self._run_multi_seq_blocks(
            q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
            apply_dropout=apply_dropout,
        )
        output = self._apply_target_attention(output, curr_ns, curr_seqs, curr_masks)
        logits = self.clsfier(output)
        return logits, output

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        """Runs the forward pass of the PCVRHyFormer model."""
        logits, _ = self._forward_impl(inputs, apply_dropout=self.training)
        return logits

    def predict(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs inference without dropout, returning both logits and embeddings."""
        return self._forward_impl(inputs, apply_dropout=False)
