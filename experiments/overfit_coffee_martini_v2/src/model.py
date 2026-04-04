from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Inverse of sigmoid: logit(x) = log(x / (1 - x))."""
    return torch.log(x / (1.0 - x))


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions in wxyz convention.

    Args:
        q1: [..., 4] quaternion (w, x, y, z)
        q2: [..., 4] quaternion (w, x, y, z)

    Returns:
        [..., 4] product quaternion (w, x, y, z)
    """
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)


# ---------------------------------------------------------------------------
# CanonicalGaussianHead
# ---------------------------------------------------------------------------

class CanonicalGaussianHead(nn.Module):
    """MLP decoder: [P, 2048] -> K Gaussians per patch.

    For each of P tracked points, produces K Gaussians anchored to the
    3D point cloud positions.  K=128 by default, SH degree 0.

    Architecture:
        LayerNorm -> MLP (2048 -> 1024 -> 512) -> 5 output heads
        (xyz offset, log_scale, rotation quaternion, logit_opacity, sh)
    """

    def __init__(self, token_dim: int = 2048, K: int = 128):
        super().__init__()
        self.K = K
        # SH degree 0: 1 coefficient per colour channel
        self.sh_dim = 3

        self.norm = nn.LayerNorm(token_dim)

        self.mlp = nn.Sequential(
            nn.Linear(token_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
        )

        hidden = 512

        # xyz offset relative to anchor point, one per Gaussian
        self.head_xyz = nn.Linear(hidden, K * 3)
        # log scale (3 axes)
        self.head_scale = nn.Linear(hidden, K * 3)
        # rotation quaternion (wxyz)
        self.head_rot = nn.Linear(hidden, K * 4)
        # opacity logit
        self.head_opacity = nn.Linear(hidden, K * 1)
        # SH DC coefficients (degree 0)
        self.head_sh = nn.Linear(hidden, K * self.sh_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        # Small init for xyz offsets so Gaussians start near anchor
        nn.init.normal_(self.head_xyz.weight, std=1e-3)
        nn.init.zeros_(self.head_xyz.bias)

        # Init log_scale so softplus gives a sensible starting size
        nn.init.normal_(self.head_scale.weight, std=1e-3)
        nn.init.constant_(self.head_scale.bias, -4.0)

        # Identity quaternion: w=1, xyz=0
        nn.init.zeros_(self.head_rot.weight)
        bias = torch.zeros(self.K * 4)
        bias[0::4] = 1.0  # w component
        self.head_rot.bias = nn.Parameter(bias)

        # Low initial opacity
        nn.init.normal_(self.head_opacity.weight, std=1e-3)
        nn.init.constant_(self.head_opacity.bias, -2.0)

        nn.init.normal_(self.head_sh.weight, std=1e-3)
        nn.init.zeros_(self.head_sh.bias)

    def forward(self, tokens: torch.Tensor, anchor_xyz: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            tokens:     [B, P, 2048]  time-pooled SpaTrackerV2 tokens
            anchor_xyz: [B, P, 3]     3D positions of tracked points

        Returns dict with keys:
            xyz:           [B, P, K, 3]
            log_scale:     [B, P, K, 3]
            rot:           [B, P, K, 4]  (wxyz, unit quaternion)
            logit_opacity: [B, P, K, 1]
            sh:            [B, P, K, 3]
        """
        B, P, _ = tokens.shape
        K = self.K

        x = self.norm(tokens)        # [B, P, 2048]
        x = self.mlp(x)              # [B, P, 512]

        xyz_offset = self.head_xyz(x).reshape(B, P, K, 3)
        # Anchor each Gaussian near its tracked point
        xyz = anchor_xyz.unsqueeze(2) + xyz_offset

        log_scale = self.head_scale(x).reshape(B, P, K, 3)

        rot_raw = self.head_rot(x).reshape(B, P, K, 4)
        rot = F.normalize(rot_raw, dim=-1)

        logit_opacity = self.head_opacity(x).reshape(B, P, K, 1)

        sh = self.head_sh(x).reshape(B, P, K, self.sh_dim)

        return {
            "xyz": xyz,
            "log_scale": log_scale,
            "rot": rot,
            "logit_opacity": logit_opacity,
            "sh": sh,
        }


# ---------------------------------------------------------------------------
# DeformationHead
# ---------------------------------------------------------------------------

class DeformationHead(nn.Module):
    """TransformerEncoder + output heads producing per-frame residuals.

    Architecture:
        LayerNorm -> proj (2048 -> 256) -> 2-layer TransformerEncoder
        (d=256, 8 heads) -> 5 delta heads

    Deltas are tanh-bounded: dxyz*0.1, dscale*0.5, dopacity*0.3.
    Rotation delta is a unit quaternion (applied via quaternion_multiply).
    """

    def __init__(self, token_dim: int = 2048, K: int = 128, d_model: int = 256):
        super().__init__()
        self.K = K
        self.sh_dim = 3

        self.norm = nn.LayerNorm(token_dim)
        self.proj = nn.Linear(token_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Delta heads
        self.head_dxyz = nn.Linear(d_model, K * 3)
        self.head_dscale = nn.Linear(d_model, K * 3)
        self.head_drot = nn.Linear(d_model, K * 4)
        self.head_dopacity = nn.Linear(d_model, K * 1)
        self.head_dsh = nn.Linear(d_model, K * self.sh_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for head in [self.head_dxyz, self.head_dscale, self.head_drot,
                     self.head_dopacity, self.head_dsh]:
            nn.init.normal_(head.weight, std=1e-3)
            nn.init.zeros_(head.bias)

        # Identity rotation delta: w=1, xyz=0
        bias = torch.zeros(self.K * 4)
        bias[0::4] = 1.0
        self.head_drot.bias = nn.Parameter(bias)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            tokens: [B, T, P, 2048]  per-frame SpaTrackerV2 tokens

        Returns dict with keys (one set of deltas per frame):
            dxyz:     [B, T, P, K, 3]
            dscale:   [B, T, P, K, 3]
            drot:     [B, T, P, K, 4]  (unit quaternion)
            dopacity: [B, T, P, K, 1]
            dsh:      [B, T, P, K, 3]
        """
        B, T, P, _ = tokens.shape
        K = self.K

        # Flatten time into batch dimension for token processing
        x = tokens.reshape(B * T, P, -1)          # [B*T, P, 2048]
        x = self.norm(x)
        x = self.proj(x)                           # [B*T, P, 256]
        x = self.transformer(x)                    # [B*T, P, 256]

        dxyz = torch.tanh(self.head_dxyz(x)) * 0.1
        dxyz = dxyz.reshape(B, T, P, K, 3)

        dscale = torch.tanh(self.head_dscale(x)) * 0.5
        dscale = dscale.reshape(B, T, P, K, 3)

        drot_raw = self.head_drot(x).reshape(B * T, P, K, 4)
        drot = F.normalize(drot_raw, dim=-1)
        drot = drot.reshape(B, T, P, K, 4)

        dopacity = torch.tanh(self.head_dopacity(x)) * 0.3
        dopacity = dopacity.reshape(B, T, P, K, 1)

        dsh = self.head_dsh(x).reshape(B, T, P, K, self.sh_dim)

        return {
            "dxyz": dxyz,
            "dscale": dscale,
            "drot": drot,
            "dopacity": dopacity,
            "dsh": dsh,
        }


# ---------------------------------------------------------------------------
# compose_gaussians
# ---------------------------------------------------------------------------

def compose_gaussians(
    canonical: dict[str, torch.Tensor],
    deltas: Optional[dict[str, torch.Tensor]] = None,
    scale_factor: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Combine canonical Gaussians with optional per-frame deltas.

    Static (no deltas):
        xyz           = canonical xyz
        scale         = softplus(log_scale) * scale_factor
        rot           = canonical rot (already unit quaternion)
        opacity       = sigmoid(logit_opacity)
        sh            = canonical sh

    Dynamic (with deltas):
        xyz           = canonical xyz + dxyz
        scale         = clamp(softplus(log_scale + dscale) * scale_factor, 1e-6, 5.0)
        rot           = quaternion_multiply(drot, canonical rot)
        opacity       = sigmoid(logit_opacity + dopacity)
        sh            = canonical sh + dsh

    Args:
        canonical:    output of CanonicalGaussianHead.forward()
        deltas:       output of DeformationHead.forward() for one frame,
                      shapes [B, P, K, ...] (time dimension already selected)
        scale_factor: global scale multiplier (e.g. for resolution adaptation)

    Returns dict with keys: xyz, scale, rot, opacity, sh
    """
    xyz = canonical["xyz"]
    log_scale = canonical["log_scale"]
    rot = canonical["rot"]
    logit_opacity = canonical["logit_opacity"]
    sh = canonical["sh"]

    if deltas is None:
        scale = F.softplus(log_scale) * scale_factor
        opacity = torch.sigmoid(logit_opacity)
        return {
            "xyz": xyz,
            "scale": scale,
            "rot": rot,
            "opacity": opacity,
            "sh": sh,
        }
    else:
        xyz = xyz + deltas["dxyz"]
        scale = torch.clamp(
            F.softplus(log_scale + deltas["dscale"]) * scale_factor,
            min=1e-6,
            max=5.0,
        )
        rot = quaternion_multiply(deltas["drot"], rot)
        rot = F.normalize(rot, dim=-1)
        opacity = torch.sigmoid(logit_opacity + deltas["dopacity"])
        sh = sh + deltas["dsh"]
        return {
            "xyz": xyz,
            "scale": scale,
            "rot": rot,
            "opacity": opacity,
            "sh": sh,
        }


# ---------------------------------------------------------------------------
# build_model / compose_forward
# ---------------------------------------------------------------------------

def build_model(config: dict) -> dict[str, nn.Module]:
    """Instantiate model modules from config.

    Returns:
        {"canonical_head": CanonicalGaussianHead, "deformation_head": DeformationHead}
    """
    K = config.get("K", 128)
    token_dim = config.get("token_dim", 2048)
    d_model = config.get("d_model", 256)

    canonical_head = CanonicalGaussianHead(token_dim=token_dim, K=K)
    deformation_head = DeformationHead(token_dim=token_dim, K=K, d_model=d_model)

    return {
        "canonical_head": canonical_head,
        "deformation_head": deformation_head,
    }


def compose_forward(
    modules: dict[str, nn.Module],
    batch: dict,
    step: int,
    phase: str,
    device: torch.device,
) -> dict:
    """Experiment-specific forward pass.

    Args:
        modules:  dict returned by build_model()
        batch:    dict from CachedSceneDataset with keys:
                    tokens      [B, T, P, 2048]
                    points_map  [B, T, H, W, 3]  (or anchor_xyz [B, P, 3])
                    frames      [B, T, C, H, W]
                    cameras     list of camera dicts
        step:     global training step
        phase:    "canonical" | "deformation" | "joint"
        device:   torch device

    Returns dict with keys expected by the shared engine's loss / render pipeline:
        gaussians_per_frame: list[T] of dicts, each with keys
                             xyz, scale, rot, opacity, sh — shapes [B, P*K, ...]
    """
    canonical_head: CanonicalGaussianHead = modules["canonical_head"]
    deformation_head: DeformationHead = modules["deformation_head"]

    tokens: torch.Tensor = batch["tokens"].to(device)       # [B, T, P, 2048]
    # anchor_xyz: use first-frame point positions, shape [B, P, 3]
    if "anchor_xyz" in batch:
        anchor_xyz: torch.Tensor = batch["anchor_xyz"].to(device)
    else:
        # Fall back to first frame of points_map, spatially averaged
        points_map: torch.Tensor = batch["points_map"].to(device)  # [B, T, H, W, 3]
        B_pm, T_pm, H_pm, W_pm, _ = points_map.shape
        anchor_xyz = points_map[:, 0].reshape(B_pm, H_pm * W_pm, 3)

    B, T, P, _ = tokens.shape

    # Time-pool tokens for canonical head
    pooled_tokens = tokens.mean(dim=1)   # [B, P, 2048]

    # Compute scale_factor based on spatial resolution
    if "points_map" in batch:
        H = batch["points_map"].shape[2]
        W = batch["points_map"].shape[3]
    else:
        H, W = 384, 512  # default resolution
    scale_factor = 1.0 / math.sqrt(H * W)

    # Canonical head (always computed)
    canonical = canonical_head(pooled_tokens, anchor_xyz)   # [B, P, K, ...]

    gaussians_per_frame = []

    if phase == "canonical":
        # Static: same Gaussians for every frame
        gauss = compose_gaussians(canonical, deltas=None, scale_factor=scale_factor)
        # Flatten P*K
        gauss_flat = {k: v.reshape(B, P * canonical_head.K, *v.shape[4:])
                      for k, v in gauss.items()}
        for _ in range(T):
            gaussians_per_frame.append(gauss_flat)

    else:
        # Deformation head produces per-frame deltas
        all_deltas = deformation_head(tokens)   # [B, T, P, K, ...]

        for t in range(T):
            deltas_t = {k: v[:, t] for k, v in all_deltas.items()}
            gauss = compose_gaussians(canonical, deltas=deltas_t, scale_factor=scale_factor)
            gauss_flat = {k: v.reshape(B, P * canonical_head.K, *v.shape[4:])
                          for k, v in gauss.items()}
            gaussians_per_frame.append(gauss_flat)

    return {"gaussians_per_frame": gaussians_per_frame}
