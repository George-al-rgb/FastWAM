"""FastDriftingWAM with a native one-step action expert."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from .dot import DoT
from .drifting_loss import PDTDObjective
from .fastwam import FastWAM


class FastDriftingWAM(FastWAM):
    def __init__(
        self,
        *args,
        drifting_config: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.configure_drifting(drifting_config or {})

    def configure_drifting(self, drifting_config: Mapping[str, Any]) -> None:
        """Set the non-trainable PDTD hyperparameters from Hydra config."""
        self.drifting_num_siblings = int(drifting_config.get("num_siblings", 8))
        if self.drifting_num_siblings < 2:
            raise ValueError("`drifting.num_siblings` must be at least 2.")

        bandwidths = drifting_config.get("bandwidths", (0.02, 0.05, 0.2))
        metric = drifting_config.get("metric")
        metric_path = drifting_config.get("metric_path")
        if metric_path:
            payload = torch.load(
                Path(metric_path).expanduser(),
                map_location="cpu",
            )
            metric = payload.get("metric") if isinstance(payload, Mapping) else payload
        self.drifting_objective = PDTDObjective(
            bandwidths=bandwidths,
            eps_distance=float(drifting_config.get("eps_distance", 1.0e-6)),
            eps_affinity=float(drifting_config.get("eps_affinity", 1.0e-6)),
            eps_force=float(drifting_config.get("eps_force", 1.0e-6)),
            beta=float(drifting_config.get("beta", 0.1)),
            metric=metric,
        )

    def _forward_action_siblings(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context: Optional[torch.Tensor],
        video_context_mask: Optional[torch.Tensor],
        video_attention_mask: Optional[torch.Tensor],
        action_noise: torch.Tensor,
        action_attention_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode video once, then evaluate all direct action siblings in parallel."""
        batch_size, num_siblings, action_horizon, action_dim = action_noise.shape

        video_tokens, video_cache_k, video_cache_v = self.dot.prefill_video_state_tensor(
            video_tokens=video_tokens,
            video_freqs=video_freqs,
            video_t_mod=video_t_mod,
            video_context=video_context,
            video_context_mask=video_context_mask,
            video_attention_mask=video_attention_mask,
        )
        fused_video_kv = self.dot.kv_fusion(
            video_cache_k=video_cache_k,
            video_cache_v=video_cache_v,
            video_freqs=video_freqs,
        )

        # prefix is repeated only after its single differentiable computation.
        flat_noise = action_noise.reshape(
            batch_size * num_siblings,
            action_horizon,
            action_dim,
        )
        action_tokens, action_freqs = self.action_expert.prepare(flat_noise)
        fused_video_k = fused_video_kv["k"].repeat_interleave(num_siblings, dim=0)
        fused_video_v = fused_video_kv["v"].repeat_interleave(num_siblings, dim=0)
        if action_attention_mask is not None:
            action_attention_mask = action_attention_mask.to(device=action_tokens.device)

        action_tokens = self.dot.forward_action_with_video_cache_tensor(
            action_tokens=action_tokens,
            action_freqs=action_freqs,
            action_attention_mask=action_attention_mask,
            fused_video_k=fused_video_k,
            fused_video_v=fused_video_v,
        )
        generated_actions = self.action_expert.post(action_tokens).reshape(
            batch_size,
            num_siblings,
            action_horizon,
            action_dim,
        )
        return video_tokens, generated_actions

    def training_loss(self, sample, tiled: bool = False):
        """Keep Wan video flow matching and replace only the action objective.

        The video branch is sampled exactly as in ``FastWAM.training_loss``.
        The action branch follows DriftingVLA Eq. (15), (17), (21), and (33):
        Gaussian action chunks are generated directly, and PDTD supplies the
        detached regression target.
        """
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_target = inputs["action_target"]
        action_is_pad = inputs["action_is_pad"]
        action_dim_is_pad = inputs["action_dim_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        # Preserve the original Wan video flow-matching training path.
        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents_video = self.train_video_scheduler.add_noise(
            input_latents,
            noise_video,
            timestep_video,
        )
        target_video = self.train_video_scheduler.training_target(
            input_latents,
            noise_video,
            timestep_video,
        )
        if inputs["first_frame_latents"] is not None:
            latents_video[:, :, 0:1] = inputs["first_frame_latents"]

        (
            video_tokens,
            t_video,
            video_t_mod,
            video_context,
            video_context_mask,
            video_freqs,
            f_video,
            h_video,
            w_video,
            tokens_per_frame,
        ) = self.video_expert.prepare(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action.shape[1],
            video_tokens_per_frame=tokens_per_frame,
            device=video_tokens.device,
        )
        video_attention_mask, action_attention_mask = DoT._split_attention_mask(
            attention_mask=attention_mask,
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action.shape[1],
        )

        action_noise = torch.randn(
            (
                batch_size,
                self.drifting_num_siblings,
                action.shape[1],
                action.shape[2],
            ),
            device=action.device,
            dtype=action.dtype,
        )
        video_tokens, generated_actions = self._forward_action_siblings(
            video_tokens=video_tokens,
            video_freqs=video_freqs,
            video_t_mod=video_t_mod,
            video_context=video_context,
            video_context_mask=video_context_mask,
            video_attention_mask=video_attention_mask,
            action_noise=action_noise,
            action_attention_mask=action_attention_mask,
        )
        pred_video = self.video_expert.post(
            video_tokens,
            t_video,
            f_video,
            h_video,
            w_video,
        )

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device,
            dtype=loss_video_per_sample.dtype,
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        valid_action = None
        if action_is_pad is not None:
            valid_action = ~action_is_pad
        if action_dim_is_pad is not None:
            valid_dimensions = ~action_dim_is_pad
            if valid_action is None:
                valid_action = valid_dimensions[:, None, :].expand(
                    -1,
                    action.shape[1],
                    -1,
                )
            else:
                valid_action = valid_action[:, :, None] & valid_dimensions[:, None, :]
        loss_action = self.drifting_objective(
            generated=generated_actions,
            demonstrated=action_target,
            valid_mask=valid_action,
        ).to(dtype=loss_video.dtype)

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        return loss_total, loss_dict

    @staticmethod
    def _validate_single_image(input_image: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                "`input_image` must have shape [1,3,H,W] or [3,H,W], "
                f"got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                "`input_image` must be resized before infer, expected multiples of 16 "
                f"but got HxW=({height},{width})"
            )
        return input_image, height, width

    def _prepare_inference_proprio(self, proprio: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if proprio is None:
            return None
        if self.proprio_dim is None:
            raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        elif proprio.ndim != 2 or proprio.shape[0] != 1:
            raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
        if proprio.shape[1] != self.proprio_dim:
            raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
        return proprio.to(device=self.device, dtype=self.torch_dtype)

    def _resolve_inference_context(
        self,
        prompt: Optional[str],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    "`context/context_mask` must be [B,L,D]/[B,L], "
                    f"got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)

        proprio = self._prepare_inference_proprio(proprio)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )
        return context, context_mask

    @torch.no_grad()
    def _predict_action_once(
        self,
        first_frame_latents: torch.Tensor,
        action_noise: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        (
            video_tokens,
            _t_video,
            video_t_mod,
            video_context,
            video_context_mask,
            video_freqs,
            _f_video,
            _h_video,
            _w_video,
            tokens_per_frame,
        ) = self.video_expert.prepare(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        video_seq_len = int(video_tokens.shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=action_noise.shape[1],
            video_tokens_per_frame=tokens_per_frame,
            device=video_tokens.device,
        )
        video_attention_mask, action_attention_mask = DoT._split_attention_mask(
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
            action_seq_len=action_noise.shape[1],
        )
        _video_tokens, video_cache_k, video_cache_v = self.dot.prefill_video_state_tensor(
            video_tokens=video_tokens,
            video_freqs=video_freqs,
            video_t_mod=video_t_mod,
            video_context=video_context,
            video_context_mask=video_context_mask,
            video_attention_mask=video_attention_mask,
        )
        fused_video_kv = self.dot.kv_fusion(
            video_cache_k=video_cache_k,
            video_cache_v=video_cache_v,
            video_freqs=video_freqs,
        )
        action_tokens, action_freqs = self.action_expert.prepare(action_noise)
        action_tokens = self.dot.forward_action_with_video_cache_tensor(
            action_tokens=action_tokens,
            action_freqs=action_freqs,
            action_attention_mask=action_attention_mask,
            fused_video_k=fused_video_kv["k"],
            fused_video_v=fused_video_kv["v"],
        )
        return self.action_expert.post(action_tokens)

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        compile_action_infer: bool = False,
    ) -> dict[str, Any]:
        del negative_prompt, text_cfg_scale, num_inference_steps, sigma_shift, compile_action_infer
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError("`infer_action` requires `video_attention_mask_mode='first_frame_causal'`.")
        if action_horizon is None or int(action_horizon) <= 0:
            raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")

        input_image, _height, _width = self._validate_single_image(input_image)
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_noise = torch.randn(
            (1, int(action_horizon), self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(
            input_image=input_image,
            tiled=tiled,
        )
        context, context_mask = self._resolve_inference_context(
            prompt=prompt,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
        )
        action = self._predict_action_once(
            first_frame_latents=first_frame_latents,
            action_noise=action_noise,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=bool(
                getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)
            ),
        )
        return {"action": action[0].detach().to(device="cpu", dtype=torch.float32)}

    @torch.no_grad()
    def _predict_video_step(
        self,
        latents_video: torch.Tensor,
        timestep_video: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        action_condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        (
            video_tokens,
            t_video,
            video_t_mod,
            video_context,
            video_context_mask,
            video_freqs,
            f_video,
            h_video,
            w_video,
            tokens_per_frame,
        ) = self.video_expert.prepare(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action_condition,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        video_attention_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_tokens.shape[1],
            video_tokens_per_frame=tokens_per_frame,
            device=video_tokens.device,
        )
        video_tokens, _video_cache_k, _video_cache_v = self.dot.prefill_video_state_tensor(
            video_tokens=video_tokens,
            video_freqs=video_freqs,
            video_t_mod=video_t_mod,
            video_context=video_context,
            video_context_mask=video_context_mask,
            video_attention_mask=video_attention_mask,
        )
        return self.video_expert.post(video_tokens, t_video, f_video, h_video, w_video)

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
        compile_action_infer: bool = False,
    ) -> dict[str, Any]:
        del test_action_with_infer_action
        self.eval()

        action_out = self.infer_action(
            prompt=prompt,
            input_image=input_image.clone(),
            action_horizon=int(action_horizon),
            proprio=proprio.clone() if proprio is not None else None,
            context=context.clone() if context is not None else None,
            context_mask=context_mask.clone() if context_mask is not None else None,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=1,
            sigma_shift=None,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            compile_action_infer=compile_action_infer,
        )["action"]

        input_image, height, width = self._validate_single_image(input_image)
        if num_video_frames % 4 != 1:
            raise ValueError(f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}")
        if num_video_frames <= 1:
            raise ValueError(f"`num_video_frames` must be > 1, got {num_video_frames}")

        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != int(action_horizon):
                raise ValueError(
                    "`action` must have shape [1, T, a_dim] or [T, a_dim], "
                    f"got {tuple(action.shape)} with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)

        context, context_mask = self._resolve_inference_context(
            prompt=prompt,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
        )
        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(
            input_image=input_image,
            tiled=tiled,
        )
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        infer_timesteps, infer_deltas = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=int(num_inference_steps),
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        for step_t, step_delta in zip(infer_timesteps, infer_deltas):
            timestep_video = step_t.unsqueeze(0).to(
                dtype=latents_video.dtype,
                device=self.device,
            )
            pred_video = self._predict_video_step(
                latents_video=latents_video,
                timestep_video=timestep_video,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                action_condition=action if getattr(self.video_expert, "action_conditioned", False) else None,
            )
            latents_video = self.infer_video_scheduler.step(
                pred_video,
                step_delta,
                latents_video,
            )
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": action_out,
        }

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        del action_cfg_scale
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )
