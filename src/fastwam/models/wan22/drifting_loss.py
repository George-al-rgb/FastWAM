from __future__ import annotations

from typing import Iterable, Optional

import torch


class PDTDObjective:
    def __init__(
        self,
        bandwidths: Iterable[float] = (0.02, 0.05, 0.2),
        eps_distance: float = 1.0e-6,
        eps_affinity: float = 1.0e-6,
        eps_force: float = 1.0e-6,
        beta: float = 0.1,
        metric: Optional[torch.Tensor] = None,
    ):
        self.bandwidths = tuple(float(value) for value in bandwidths)
        self.eps_distance = float(eps_distance)
        self.eps_affinity = float(eps_affinity)
        self.eps_force = float(eps_force)
        self.beta = float(beta)
        if metric is None:
            self.metric = None
        else:
            metric = torch.as_tensor(metric, dtype=torch.float32).detach()
            self.metric = 0.5 * (metric + metric.transpose(-1, -2))

    @staticmethod
    def _coordinate_valid_mask(
        valid_mask: Optional[torch.Tensor],
        batch_size: int,
        horizon: int,
        action_dim: int,
        device: torch.device,
    ) -> torch.Tensor:
        if valid_mask is None:
            return torch.ones(
                (batch_size, horizon, action_dim),
                dtype=torch.bool,
                device=device,
            )
        if valid_mask.ndim == 2:
            valid_mask = valid_mask.unsqueeze(-1).expand(-1, -1, action_dim)
        return valid_mask.to(device=device, dtype=torch.bool)

    def __call__(
        self,
        generated: torch.Tensor,
        demonstrated: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_siblings, horizon, action_dim = generated.shape

        coordinate_valid = self._coordinate_valid_mask(
            valid_mask=valid_mask,
            batch_size=batch_size,
            horizon=horizon,
            action_dim=action_dim,
            device=generated.device,
        )
        valid = coordinate_valid.to(dtype=torch.float32)

        with torch.no_grad():
            siblings = generated.float().detach()
            target_action = demonstrated.float().detach()
            references = torch.cat(
                [siblings, target_action.unsqueeze(1)],
                dim=1,
            )

            pairwise_delta = siblings[:, :, None, :, :] - references[:, None, :, :, :]
            valid_coordinates = valid[:, None, None, :, :].bool()
            masked_delta = torch.where(
                valid_coordinates,
                pairwise_delta,
                torch.zeros_like(pairwise_delta),
            )
            valid_length = valid.sum(dim=1).clamp_min(1.0)
            pairwise_distance = torch.sqrt(
                masked_delta.square().sum(dim=3)
                / valid_length[:, None, None, :]
            )
            relation_has_valid_coordinate = valid.any(dim=1)[:, None, None, :]
            relation_weight = relation_has_valid_coordinate.expand(
                -1,
                num_siblings,
                num_siblings + 1,
                -1,
            ).to(dtype=pairwise_distance.dtype)
            mu_d = (
                (pairwise_distance * relation_weight).sum(dim=(0, 1, 2))
                / relation_weight.sum(dim=(0, 1, 2)).clamp_min(1.0)
            )
            distance_scale = mu_d.clamp_min(self.eps_distance)
            normalized_distance = pairwise_distance / distance_scale

            metric = self.metric
            if metric is None:
                metric = torch.eye(
                    action_dim,
                    device=pairwise_delta.device,
                    dtype=pairwise_delta.dtype,
                )
            else:
                metric = metric.to(
                    device=pairwise_delta.device,
                    dtype=pairwise_delta.dtype,
                )
            joint_distance_square = torch.einsum(
                "bgrhd,de,bgrhe->bgrh",
                masked_delta / distance_scale[None, None, None, None, :],
                metric,
                masked_delta / distance_scale[None, None, None, None, :],
            )
            valid_steps = valid.any(dim=-1).to(dtype=pairwise_delta.dtype)
            joint_distance_square = (
                joint_distance_square * valid_steps[:, None, None, :]
            ).sum(dim=3) / (
                valid_steps.sum(dim=1)[:, None, None].clamp_min(1.0)
                * float(action_dim)
            )
            normalized_distance = torch.sqrt(
                (
                    (1.0 - self.beta) * normalized_distance.square()
                    + self.beta * joint_distance_square.unsqueeze(-1)
                ).clamp_min(0.0)
            )

            self_reference = torch.zeros(
                (1, num_siblings, num_siblings + 1, 1),
                dtype=torch.bool,
                device=generated.device,
            )
            sibling_indices = torch.arange(num_siblings, device=generated.device)
            self_reference[:, sibling_indices, sibling_indices, :] = True
            normalized_distance = normalized_distance.masked_fill(
                self_reference,
                float("inf"),
            )

            eta = distance_scale
            normalized_siblings = siblings / eta
            normalized_references = references / eta

            total_force = torch.zeros_like(siblings)
            for bandwidth in self.bandwidths:
                logits = -normalized_distance / bandwidth
                probability_forward = torch.softmax(logits, dim=2)
                probability_reverse = torch.softmax(logits, dim=1)

                affinity = torch.sqrt(
                    (probability_forward * probability_reverse).clamp_min(
                        self.eps_affinity
                    )
                )
                affinity = affinity.masked_fill(self_reference, 0.0)

                generated_affinity = affinity[:, :, :num_siblings, :]
                demonstrated_affinity = affinity[:, :, num_siblings:, :]
                negative_mass = generated_affinity.sum(dim=2, keepdim=True)
                positive_mass = demonstrated_affinity.sum(dim=2, keepdim=True)
                alpha_generated = -generated_affinity * positive_mass
                alpha_demonstrated = demonstrated_affinity * negative_mass
                alpha = torch.cat([alpha_generated, alpha_demonstrated], dim=2)

                displacement = (
                    normalized_references[:, None, :, :, :]
                    - normalized_siblings[:, :, None, :, :]
                )
                force = (alpha.unsqueeze(-2) * displacement).sum(dim=2)
                force = force * valid[:, None, :, :]

                rms_numerator = (force.square() * valid[:, None, :, :]).sum(dim=(0, 1, 2))
                rms_denominator = (
                    valid.sum(dim=(0, 1)) * num_siblings
                ).clamp_min(1.0)
                force_rms = (rms_numerator / rms_denominator).sqrt().clamp_min(
                    self.eps_force
                )
                total_force = total_force + force / force_rms

            normalized_siblings = siblings / eta
            drifting_target = (normalized_siblings + total_force).detach()

        prediction = generated.float() / eta
        squared_error = (prediction - drifting_target).square()
        squared_error = squared_error * valid[:, None, :, :]
        per_condition_denominator = (
            valid.sum(dim=(1, 2)) * num_siblings
        ).clamp_min(1.0)
        per_condition_loss = squared_error.sum(dim=(1, 2, 3)) / per_condition_denominator
        return per_condition_loss.mean()
