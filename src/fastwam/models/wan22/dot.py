from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .wan_video_dit import flash_attention, modulate, rope_apply
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)

class DoT(nn.Module):
    def __init__(
        self,
        mixtures: Dict[str, nn.Module],
        mot_checkpoint_mixed_attn: bool = False,
    ):
        super().__init__()

        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())

        self.video_expert = self.mixtures["video"]
        self.action_expert = self.mixtures["action"]
        self.video_num_layers = len(self.video_expert.blocks)
        self.action_num_layers = len(self.action_expert.blocks)
        self.num_heads = self.video_expert.num_heads
        self.attn_head_dim = self.video_expert.attn_head_dim
        self.video_attn_dim = self.video_expert.num_heads * self.video_expert.attn_head_dim
        self.action_attn_dim = self.action_expert.num_heads * self.action_expert.attn_head_dim

        # Remap the video KV channel space into the action-head attention space.
        self.kv_k_proj = nn.Linear(self.video_attn_dim, self.action_attn_dim, bias=False)
        self.kv_v_proj = nn.Linear(self.video_attn_dim, self.action_attn_dim, bias=False)

        # One layer-mixing matrix per action head. The same matrix is used for K and V.
        self.layer_mix = nn.Parameter(
            torch.full(
                (
                    self.action_expert.num_heads,
                    self.action_num_layers,
                    self.video_num_layers,
                ),
                1.0 / self.video_num_layers,
            )
        )
        self.mot_checkpoint_mixed_attn = bool(mot_checkpoint_mixed_attn)
        self.compile_training_layers = False
        self.num_layers = self.video_num_layers

    @staticmethod
    def _split_modulation(
        block: nn.Module,
        t_mod: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Expand one Video-DiT block's six modulation tensors."""
        modulation = block.modulation.to(
            device=t_mod.device,
            dtype=t_mod.dtype,
        ) + t_mod
        chunk_dim = 2 if t_mod.ndim == 4 else 1
        chunks = modulation.chunk(6, dim=chunk_dim)
        if t_mod.ndim == 4:
            chunks = tuple(chunk.squeeze(2) for chunk in chunks)
        return chunks

    @staticmethod
    def _normalize_context_mask(
        context_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if context_mask is None:
            return None
        if context_mask.ndim == 2:
            return context_mask[:, None, None, :]
        if context_mask.ndim == 3:
            return context_mask.unsqueeze(1)
        return context_mask

    def _run_video_backbone(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context: Optional[torch.Tensor],
        video_context_mask: Optional[torch.Tensor],
        video_attention_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Run every video block and retain the rotated K/V from every layer."""
        x = video_tokens
        video_cache_k: list[torch.Tensor] = []
        video_cache_v: list[torch.Tensor] = []

        video_freqs = video_freqs.to(device=x.device)
        video_t_mod = video_t_mod.to(device=x.device, dtype=x.dtype)
        if video_attention_mask is not None:
            video_attention_mask = video_attention_mask.to(device=x.device)
        context_mask = self._normalize_context_mask(video_context_mask)
        if context_mask is not None:
            context_mask = context_mask.to(device=x.device)
        if video_context is not None:
            video_context = video_context.to(device=x.device)

        for block in self.video_expert.blocks:
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = self._split_modulation(block, video_t_mod)

            attn_input = modulate(block.norm1(x), shift_msa, scale_msa)
            q = block.self_attn.norm_q(block.self_attn.q(attn_input))
            k = block.self_attn.norm_k(block.self_attn.k(attn_input))
            v = block.self_attn.v(attn_input)
            q = rope_apply(q, video_freqs, block.num_heads)
            k = rope_apply(k, video_freqs, block.num_heads)

            video_cache_k.append(k)
            video_cache_v.append(v)

            attn_out = flash_attention(
                q=q,
                k=k,
                v=v,
                num_heads=block.num_heads,
                ctx_mask=video_attention_mask,
            )
            x = block.gate(x, gate_msa, block.self_attn.o(attn_out))

            if video_context is not None:
                x = x + block.cross_attn(
                    block.norm3(x),
                    video_context,
                    ctx_mask=context_mask,
                )

            mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
            x = block.gate(x, gate_mlp, block.ffn(mlp_input))

        return x, video_cache_k, video_cache_v

    def prefill_video_cache_tensor(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context: Optional[torch.Tensor],
        video_context_mask: Optional[torch.Tensor],
        video_attention_mask: Optional[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Run the video backbone once and collect the K/V from every block."""
        _, video_cache_k, video_cache_v = self._run_video_backbone(
            video_tokens=video_tokens,
            video_freqs=video_freqs,
            video_t_mod=video_t_mod,
            video_context=video_context,
            video_context_mask=video_context_mask,
            video_attention_mask=video_attention_mask,
        )
        return video_cache_k, video_cache_v

    def prefill_video_state_tensor(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context: Optional[torch.Tensor],
        video_context_mask: Optional[torch.Tensor],
        video_attention_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Run the video hub once and return its output together with every KV cache.

        The cache-only method above remains the inference-compatible API.  The
        state-returning variant is used by DriftingVLA training because the
        video loss and all action siblings must share the same differentiable
        video computation.
        """
        return self._run_video_backbone(
            video_tokens=video_tokens,
            video_freqs=video_freqs,
            video_t_mod=video_t_mod,
            video_context=video_context,
            video_context_mask=video_context_mask,
            video_attention_mask=video_attention_mask,
        )


    @staticmethod
    def _undo_rope(
        x: torch.Tensor,
        freqs: torch.Tensor,
        num_heads: int,
    ) -> torch.Tensor:
        """Undo the RoPE rotation previously applied to a [B, S, H*D] tensor."""
        batch_size, seq_len, _ = x.shape
        x_heads = x.reshape(batch_size, seq_len, num_heads, -1)
        x_complex = torch.view_as_complex(
            x_heads.to(torch.float64).reshape(
                batch_size, seq_len, num_heads, -1, 2
            )
        )
        x_complex = x_complex * freqs.to(device=x.device).conj()
        return torch.view_as_real(x_complex).flatten(2).to(x.dtype)

    def kv_fusion(
        self,
        video_cache_k: list[torch.Tensor],
        video_cache_v: list[torch.Tensor],
        video_freqs: torch.Tensor,
        video_action_freqs: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Fuse all video-layer KV into the single action-head KV space."""
        canonical_k = torch.stack(
            [
                self._undo_rope(
                    layer_k,
                    video_freqs,
                    self.video_expert.num_heads,
                )
                for layer_k in video_cache_k
            ],
            dim=0,
        )
        video_v = torch.stack(video_cache_v, dim=0)

        video_layers, batch_size, seq_len, _ = canonical_k.shape
        action_heads = self.action_expert.num_heads
        action_head_dim = self.action_expert.attn_head_dim

        mixed_k = self.kv_k_proj(canonical_k).reshape(
            video_layers,
            batch_size,
            seq_len,
            action_heads,
            action_head_dim,
        )
        mixed_v = self.kv_v_proj(video_v).reshape(
            video_layers,
            batch_size,
            seq_len,
            action_heads,
            action_head_dim,
        )

        # The current DoT has one action layer, so aggregate directly to it.
        layer_mix = self.layer_mix[:, 0, :].to(
            device=mixed_k.device,
            dtype=mixed_k.dtype,
        )
        fused_k = torch.einsum("hl,lbshd->bshd", layer_mix, mixed_k)
        fused_v = torch.einsum("hl,lbshd->bshd", layer_mix, mixed_v)

        fused_k = fused_k.reshape(batch_size, seq_len, self.action_attn_dim)
        fused_v = fused_v.reshape(batch_size, seq_len, self.action_attn_dim)

        action_block = self.action_expert.blocks[0]
        fused_k = action_block.self_attn.norm_k(fused_k)
        if video_action_freqs is None:
            video_action_freqs = self.action_expert.get_freqs(seq_len)
        fused_k = rope_apply(
            fused_k,
            video_action_freqs.to(device=fused_k.device),
            action_block.num_heads,
        )

        return {"k": fused_k, "v": fused_v}

    def _run_action_block(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        video_kv: dict[str, torch.Tensor],
        action_attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run the single action block against fused video and action KV."""
        block = self.action_expert.blocks[0]
        x = action_tokens
        attn_input = block.norm1(x)

        q = block.self_attn.norm_q(block.self_attn.q(attn_input))
        action_k = block.self_attn.norm_k(block.self_attn.k(attn_input))
        action_v = block.self_attn.v(attn_input)

        action_freqs = action_freqs.to(device=x.device)
        q = rope_apply(q, action_freqs, block.num_heads)
        action_k = rope_apply(action_k, action_freqs, block.num_heads)

        k = torch.cat([video_kv["k"], action_k], dim=1)
        v = torch.cat([video_kv["v"], action_v], dim=1)
        if action_attention_mask is not None:
            action_attention_mask = action_attention_mask.to(device=x.device)
        attn_out = flash_attention(
            q=q,
            k=k,
            v=v,
            num_heads=block.num_heads,
            ctx_mask=action_attention_mask,
        )
        x = x + block.self_attn.o(attn_out)
        x = x + block.ffn(block.norm2(x))
        return x

    def forward_action_with_video_cache_tensor(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        video_cache_k: Optional[list[torch.Tensor]] = None,
        video_cache_v: Optional[list[torch.Tensor]] = None,
        video_freqs: Optional[torch.Tensor] = None,
        action_attention_mask: Optional[torch.Tensor] = None,
        video_action_freqs: Optional[torch.Tensor] = None,
        fused_video_k: Optional[torch.Tensor] = None,
        fused_video_v: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the one-layer action head using a cached video backbone."""
        if fused_video_k is None or fused_video_v is None:
            if video_cache_k is None or video_cache_v is None or video_freqs is None:
                raise ValueError(
                    "Raw video KV, video_freqs, or both fused video tensors are required."
                )
            video_kv = self.kv_fusion(
                video_cache_k=video_cache_k,
                video_cache_v=video_cache_v,
                video_freqs=video_freqs,
                video_action_freqs=video_action_freqs,
            )
        else:
            video_kv = {"k": fused_video_k, "v": fused_video_v}
        return self._run_action_block(
            action_tokens=action_tokens,
            action_freqs=action_freqs,
            video_kv=video_kv,
            action_attention_mask=action_attention_mask,
        )

    @staticmethod
    def _split_attention_mask(
        attention_mask: Optional[torch.Tensor],
        video_seq_len: int,
        action_seq_len: int,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if attention_mask is None:
            return None, None
        if attention_mask.ndim == 2:
            video_attention_mask = attention_mask[:video_seq_len, :video_seq_len]
            action_attention_mask = attention_mask[
                video_seq_len : video_seq_len + action_seq_len,
                : video_seq_len + action_seq_len,
            ]
            return video_attention_mask, action_attention_mask
        if attention_mask.ndim == 4:
            video_attention_mask = attention_mask[
                ..., :video_seq_len, :video_seq_len
            ]
            action_attention_mask = attention_mask[
                ...,
                video_seq_len : video_seq_len + action_seq_len,
                : video_seq_len + action_seq_len,
            ]
            return video_attention_mask, action_attention_mask
        if attention_mask.ndim == 3:
            video_attention_mask = attention_mask[
                ..., :video_seq_len, :video_seq_len
            ].unsqueeze(1)
            action_attention_mask = attention_mask[
                ...,
                video_seq_len : video_seq_len + action_seq_len,
                : video_seq_len + action_seq_len,
            ].unsqueeze(1)
            return video_attention_mask, action_attention_mask
        return None, attention_mask

    def forward_joint_core(
        self,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        action_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context: Optional[torch.Tensor],
        video_context_mask: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the Video hub and dock the single-layer Action head onto it."""
        video_seq_len = video_tokens.shape[1]
        action_seq_len = action_tokens.shape[1]
        video_attention_mask, action_attention_mask = self._split_attention_mask(
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
            action_seq_len=action_seq_len,
        )

        video_tokens, video_cache_k, video_cache_v = self._run_video_backbone(
            video_tokens=video_tokens,
            video_freqs=video_freqs,
            video_t_mod=video_t_mod,
            video_context=video_context,
            video_context_mask=video_context_mask,
            video_attention_mask=video_attention_mask,
        )
        action_tokens = self.forward_action_with_video_cache_tensor(
            action_tokens=action_tokens,
            action_freqs=action_freqs,
            video_cache_k=video_cache_k,
            video_cache_v=video_cache_v,
            video_freqs=video_freqs,
            action_attention_mask=action_attention_mask,
        )
        return video_tokens, action_tokens

    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compatibility entry point for the former MoT training call."""
        video_payload = context_all.get("video") or {}
        video_tokens, action_tokens = self.forward_joint_core(
            video_tokens=embeds_all["video"],
            action_tokens=embeds_all["action"],
            video_freqs=freqs_all["video"],
            action_freqs=freqs_all["action"],
            video_t_mod=t_mod_all["video"],
            video_context=video_payload.get("context"),
            video_context_mask=video_payload.get("mask"),
            attention_mask=attention_mask,
        )
        return {"video": video_tokens, "action": action_tokens}
