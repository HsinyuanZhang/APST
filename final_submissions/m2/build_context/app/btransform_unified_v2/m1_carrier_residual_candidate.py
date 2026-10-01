"""Frozen-B, carrier-conditioned residual candidate for the M1 protocol.

This module deliberately leaves :class:`CrossSessionM1Decoder` untouched.  It
uses a completed ``B_ACTIVITY_ONLY`` decoder as an immutable prediction
baseline and learns only a carrier-conditioned residual from source data.
Carrier features must be constructed by the caller; this module neither reads
nor pools a bank carrier tensor, so the caller can preserve its required
unit/time alignment.
"""
from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor, nn

from .cross_session_m1_model import B_ACTIVITY_ONLY, CrossSessionM1Decoder

_OUTPUT_DIM = 16


def _require_prediction(prediction: Tensor) -> None:
    if prediction.ndim != 2 or prediction.shape[1] != _OUTPUT_DIM:
        raise ValueError(
            "base_prediction must have shape [batch, 16]; "
            f"received {tuple(prediction.shape)}"
        )
    if not prediction.is_floating_point():
        raise TypeError("base_prediction must be a floating-point tensor")


def _normalise_visibility(
    carrier_visible: Tensor | None,
    *,
    batch: int,
    device: torch.device,
    training: bool,
    dropout_p: float,
    generator: torch.Generator | None,
) -> Tensor:
    """Return one boolean carrier decision for every batch row.

    An explicit all-false tensor is a hard fallback: callers receive the
    baseline prediction without evaluating the residual head.
    """
    if carrier_visible is None:
        if not training:
            return torch.ones(batch, dtype=torch.bool, device=device)
        return torch.rand(batch, device=device, generator=generator) >= dropout_p

    visible = carrier_visible.to(device=device)
    if visible.dtype != torch.bool:
        raise TypeError("carrier_visible must be a bool tensor")
    if visible.ndim == 2 and visible.shape[1] == 1:
        visible = visible[:, 0]
    if visible.ndim != 1 or visible.shape[0] != batch:
        raise ValueError(
            "carrier_visible must have shape [batch] or [batch, 1]; "
            f"received {tuple(carrier_visible.shape)} for batch={batch}"
        )
    return visible


class CarrierResidualHead(nn.Module):
    """A source-normalized carrier residual head conditioned on B predictions.

    ``install_source_normalization`` is intentionally the only way to replace
    the initial identity normalization.  The caller must compute its inputs
    from source-training carrier features; this class never estimates moments
    from a forward batch and therefore cannot implicitly fit target features.
    """

    def __init__(self, feature_dim: int, *, hidden_dim: int = 64) -> None:
        super().__init__()
        if feature_dim < 1:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.register_buffer("carrier_mean", torch.zeros(self.feature_dim))
        self.register_buffer("carrier_scale", torch.ones(self.feature_dim))
        self.register_buffer("source_normalization_installed", torch.tensor(False))

        self.mlp = nn.Sequential(
            nn.Linear(_OUTPUT_DIM + self.feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, _OUTPUT_DIM),
        )
        # This makes the residual exactly zero at initialization irrespective
        # of base predictions or carrier inputs.
        final = self.mlp[-1]
        if not isinstance(final, nn.Linear):  # Defensive against future edits.
            raise RuntimeError("carrier residual final layer must be Linear")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def install_source_normalization(self, mean: Tensor, scale: Tensor) -> None:
        """Install precomputed source-training feature moments.

        ``scale`` is required to be finite and strictly positive.  Both values
        are copied into module buffers, detached from the caller's graph.
        """
        self._check_statistic("mean", mean)
        self._check_statistic("scale", scale)
        mean = mean.detach().to(device=self.carrier_mean.device, dtype=self.carrier_mean.dtype)
        scale = scale.detach().to(device=self.carrier_scale.device, dtype=self.carrier_scale.dtype)
        if not bool(torch.isfinite(mean).all()):
            raise ValueError("source carrier mean must contain only finite values")
        if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
            raise ValueError("source carrier scale must be finite and strictly positive")
        with torch.no_grad():
            self.carrier_mean.copy_(mean)
            self.carrier_scale.copy_(scale)
            self.source_normalization_installed.fill_(True)

    def _check_statistic(self, name: str, value: Tensor) -> None:
        if not isinstance(value, Tensor):
            raise TypeError(f"source carrier {name} must be a Tensor")
        if value.ndim != 1 or value.shape[0] != self.feature_dim:
            raise ValueError(
                f"source carrier {name} must have shape [{self.feature_dim}]; "
                f"received {tuple(value.shape)}"
            )
        if not value.is_floating_point():
            raise TypeError(f"source carrier {name} must be floating point")

    def _check_inputs(self, base_prediction: Tensor, carrier_features: Tensor) -> None:
        _require_prediction(base_prediction)
        if carrier_features.ndim != 2 or carrier_features.shape != (
            base_prediction.shape[0], self.feature_dim,
        ):
            raise ValueError(
                "carrier_features must have shape [batch, feature_dim] matching "
                f"base_prediction; expected [{base_prediction.shape[0]}, {self.feature_dim}], "
                f"received {tuple(carrier_features.shape)}"
            )
        if not carrier_features.is_floating_point():
            raise TypeError("carrier_features must be a floating-point tensor")
        if carrier_features.device != base_prediction.device:
            raise ValueError("carrier_features and base_prediction must be on the same device")
        if base_prediction.device != self.carrier_mean.device:
            raise ValueError(
                "CarrierResidualHead and its inputs must be on the same device; "
                "move the wrapper with .to(device) before calling forward"
            )

    def forward(
        self,
        base_prediction: Tensor,
        carrier_features: Tensor,
        *,
        carrier_visible: Tensor | None = None,
    ) -> Tensor:
        """Return a residual, with explicit false rows fixed at exact zero."""
        _require_prediction(base_prediction)
        visible = _normalise_visibility(
            carrier_visible,
            batch=base_prediction.shape[0],
            device=base_prediction.device,
            training=False,
            dropout_p=0.0,
            generator=None,
        )
        if not bool(visible.any()):
            return torch.zeros_like(base_prediction)
        self._check_inputs(base_prediction, carrier_features)
        dtype = self.mlp[0].weight.dtype
        prediction = base_prediction.detach().to(dtype=dtype)
        features = carrier_features.to(dtype=dtype)
        normalized = (features - self.carrier_mean.to(dtype=dtype)) / self.carrier_scale.to(dtype=dtype)
        residual = self.mlp(torch.cat((prediction, normalized), dim=-1)).to(dtype=base_prediction.dtype)
        return torch.where(visible.unsqueeze(-1), residual, torch.zeros_like(base_prediction))


class FrozenBCarrierResidual(nn.Module):
    """Freeze an activity-only M1 B decoder and add a gated carrier residual.

    ``carrier_features`` is an external, already time-aligned feature matrix
    with shape ``[batch, feature_dim]``.  ``carrier_visible`` is an optional
    bool tensor with shape ``[batch]`` or ``[batch, 1]``; a false value removes
    the entire carrier channel for that row.  In evaluation mode the default
    is all visible; in training mode the default drops whole carrier channels
    independently with probability ``carrier_dropout_p``.
    """

    def __init__(
        self,
        base: CrossSessionM1Decoder,
        *,
        feature_dim: int,
        hidden_dim: int = 64,
        carrier_dropout_p: float = 0.5,
    ) -> None:
        super().__init__()
        if not isinstance(base, CrossSessionM1Decoder):
            raise TypeError("base must be a CrossSessionM1Decoder")
        if base.arm != B_ACTIVITY_ONLY:
            raise ValueError(
                "FrozenBCarrierResidual requires a B_ACTIVITY_ONLY base; "
                f"received arm={base.arm!r}"
            )
        if not 0.0 <= carrier_dropout_p < 1.0:
            raise ValueError("carrier_dropout_p must satisfy 0 <= p < 1")

        self.base = base
        self.carrier_dropout_p = float(carrier_dropout_p)
        self.residual_head = CarrierResidualHead(feature_dim, hidden_dim=hidden_dim)
        self._freeze_base()

    def _freeze_base(self) -> None:
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()

    def train(self, mode: bool = True) -> FrozenBCarrierResidual:
        """Set head mode while keeping the frozen B decoder in evaluation mode."""
        super().train(mode)
        self.base.eval()
        return self

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Yield exactly the parameters an optimizer may update."""
        return self.residual_head.parameters()

    def _base_prediction(
        self,
        x: Tensor,
        bank: object,
        *,
        input_valid_mask: Tensor | None,
    ) -> Tensor:
        self.base.eval()
        with torch.no_grad():
            prediction = self.base(x, bank, input_valid_mask=input_valid_mask)
        _require_prediction(prediction)
        return prediction.detach()

    def _resolve_visibility(
        self,
        carrier_visible: Tensor | None,
        *,
        batch: int,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> Tensor:
        return _normalise_visibility(
            carrier_visible,
            batch=batch,
            device=device,
            training=self.training,
            dropout_p=self.carrier_dropout_p,
            generator=generator,
        )

    def _combine(
        self,
        base_prediction: Tensor,
        carrier_features: Tensor | None,
        *,
        carrier_visible: Tensor | None,
        visibility_generator: torch.Generator | None,
    ) -> Tensor:
        _require_prediction(base_prediction)
        visible = self._resolve_visibility(
            carrier_visible,
            batch=base_prediction.shape[0],
            device=base_prediction.device,
            generator=visibility_generator,
        )
        # Preserve the original B tensor exactly for a deliberate all-false
        # fallback.  This also permits a caller to omit features on that path.
        if not bool(visible.any()):
            return base_prediction
        if carrier_features is None:
            raise ValueError("carrier_features are required when any carrier_visible value is true")
        residual = self.residual_head(
            base_prediction, carrier_features, carrier_visible=visible
        )
        combined = base_prediction + residual
        # False rows select the original tensor rather than an arithmetic
        # equivalent, giving a hard, per-row B fallback.
        return torch.where(visible.unsqueeze(-1), combined, base_prediction)

    def forward(
        self,
        x: Tensor,
        bank: object,
        carrier_features: Tensor | None,
        *,
        input_valid_mask: Tensor | None = None,
        carrier_visible: Tensor | None = None,
        visibility_generator: torch.Generator | None = None,
    ) -> Tensor:
        """Run frozen B through its public forward, then apply the residual."""
        base_prediction = self._base_prediction(x, bank, input_valid_mask=input_valid_mask)
        return self._combine(
            base_prediction,
            carrier_features,
            carrier_visible=carrier_visible,
            visibility_generator=visibility_generator,
        )

    def forward_cached(
        self,
        base_prediction: Tensor,
        carrier_features: Tensor | None,
        *,
        carrier_visible: Tensor | None = None,
        visibility_generator: torch.Generator | None = None,
    ) -> Tensor:
        """Train or evaluate the head from a detached cached B prediction."""
        return self._combine(
            base_prediction.detach(),
            carrier_features,
            carrier_visible=carrier_visible,
            visibility_generator=visibility_generator,
        )


__all__ = ["CarrierResidualHead", "FrozenBCarrierResidual"]
