"""Tests for async-friendly metric accumulation."""
import pytest
import torch


def test_accumulator_tracks_running_mean():
    from src.metrics import MetricAccumulator

    acc = MetricAccumulator()
    acc.update("loss", torch.tensor(1.0))
    acc.update("loss", torch.tensor(3.0))
    result = acc.flush()
    assert abs(result["loss"] - 2.0) < 1e-6


def test_accumulator_tracks_multiple_metrics():
    from src.metrics import MetricAccumulator

    acc = MetricAccumulator()
    acc.update("loss", torch.tensor(1.0))
    acc.update("psnr", torch.tensor(25.0))
    result = acc.flush()
    assert "loss" in result
    assert "psnr" in result


def test_flush_resets_accumulator():
    from src.metrics import MetricAccumulator

    acc = MetricAccumulator()
    acc.update("loss", torch.tensor(1.0))
    acc.flush()
    acc.update("loss", torch.tensor(5.0))
    result = acc.flush()
    assert abs(result["loss"] - 5.0) < 1e-6


def test_flush_empty_returns_empty():
    from src.metrics import MetricAccumulator

    acc = MetricAccumulator()
    result = acc.flush()
    assert result == {}


def test_compute_psnr_identical():
    from src.metrics import compute_psnr_tensor

    img = torch.rand(3, 64, 64)
    psnr = compute_psnr_tensor(img, img)
    assert psnr.item() > 50.0


def test_compute_psnr_different():
    from src.metrics import compute_psnr_tensor

    a = torch.zeros(3, 64, 64)
    b = torch.ones(3, 64, 64)
    psnr = compute_psnr_tensor(a, b)
    assert abs(psnr.item() - 0.0) < 0.01
