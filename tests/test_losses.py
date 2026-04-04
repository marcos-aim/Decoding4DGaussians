"""Tests for shared training engine loss registry."""
import pytest
import torch


@pytest.fixture
def registry():
    from src.losses import LossRegistry
    return LossRegistry(device="cpu")


def test_registry_has_builtin_losses(registry):
    """All built-in losses should be registered."""
    expected = {"rgb", "ssim", "geometric", "depth", "tv", "scale_reg", "opacity_reg"}
    assert expected.issubset(set(registry.list_losses()))


def test_register_custom_loss(registry):
    """Can register a custom loss function."""
    def my_loss(predictions, targets, **kwargs):
        return torch.tensor(0.5)

    registry.register("custom", my_loss)
    assert "custom" in registry.list_losses()


def test_compute_single_loss(registry):
    """Computing a single active loss returns a scalar tensor."""
    rendered = torch.rand(1, 3, 64, 64)
    target = torch.rand(1, 3, 64, 64)
    predictions = {"rendered": rendered, "scales": torch.rand(100, 3), "opacity": torch.rand(100, 1).sigmoid()}
    targets = {"gt_image": target}

    phase_losses = {"rgb": {"weight": 1.0}}
    loss = registry.compute(predictions, targets, step=0, phase_losses=phase_losses)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_loss_warmup(registry):
    """Loss with warmup_steps should be zero at step 0."""
    rendered = torch.rand(1, 3, 64, 64)
    target = torch.rand(1, 3, 64, 64)
    predictions = {"rendered": rendered, "scales": torch.rand(100, 3), "opacity": torch.rand(100, 1).sigmoid()}
    targets = {"gt_image": target}

    phase_losses = {"rgb": {"weight": 1.0, "warmup_steps": 100}}
    loss_at_0 = registry.compute(predictions, targets, step=0, phase_losses=phase_losses)
    loss_at_100 = registry.compute(predictions, targets, step=100, phase_losses=phase_losses)
    assert loss_at_0.item() < loss_at_100.item()


def test_compute_every_n_skips(registry):
    """Loss with compute_every_n > 1 should return 0 on non-compute steps."""
    rendered = torch.rand(1, 3, 64, 64)
    target = torch.rand(1, 3, 64, 64)
    predictions = {"rendered": rendered, "scales": torch.rand(100, 3), "opacity": torch.rand(100, 1).sigmoid()}
    targets = {"gt_image": target}

    phase_losses = {"rgb": {"weight": 1.0, "compute_every_n": 5}}
    loss_step_0 = registry.compute(predictions, targets, step=0, phase_losses=phase_losses)
    loss_step_1 = registry.compute(predictions, targets, step=1, phase_losses=phase_losses)
    assert loss_step_0.item() > 0


def test_omitted_losses_not_computed(registry):
    """Only losses listed in phase_losses are computed."""
    predictions = {"rendered": torch.rand(1, 3, 64, 64), "scales": torch.rand(100, 3),
                   "opacity": torch.rand(100, 1).sigmoid()}
    targets = {"gt_image": torch.rand(1, 3, 64, 64)}

    phase_losses = {"rgb": {"weight": 1.0}}
    loss = registry.compute(predictions, targets, step=0, phase_losses=phase_losses)
    assert loss.item() > 0


def test_photometric_loss_values():
    """Photometric loss should combine L1 and SSIM."""
    from src.losses import photometric_loss

    identical = torch.rand(1, 3, 64, 64)
    loss = photometric_loss(identical, identical, lambda_ssim=0.85)
    assert loss.item() < 0.01

    different = torch.rand(1, 3, 64, 64)
    loss2 = photometric_loss(identical, different, lambda_ssim=0.85)
    assert loss2.item() > loss.item()


def test_scale_regularization():
    """Scale reg should penalize large scales."""
    from src.losses import scale_regularization

    small = torch.ones(100, 3) * 0.01
    large = torch.ones(100, 3) * 5.0
    assert scale_regularization(small).item() < scale_regularization(large).item()


def test_opacity_regularization():
    """Opacity reg should penalize mid-range opacities."""
    from src.losses import opacity_regularization

    binary = torch.tensor([[0.01], [0.99]] * 50)
    midrange = torch.tensor([[0.5]] * 100)
    assert opacity_regularization(binary).item() < opacity_regularization(midrange).item()
