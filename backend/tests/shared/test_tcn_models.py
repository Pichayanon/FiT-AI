from __future__ import annotations

import torch

from shared.tcn_models import PhaseTCN, PhaseTemporalBlock, SimpleTCN, TemporalBlock


def test_temporal_block_preserves_time_length_and_changes_channels() -> None:
    block = TemporalBlock(in_ch=2, out_ch=4, k=3, dilation=2, dropout=0.0)
    x = torch.randn(1, 2, 12)

    y = block(x)

    assert y.shape == (1, 4, 12)


def test_simple_tcn_forward_with_and_without_attention() -> None:
    x = torch.randn(2, 8, 3)

    attention_model = SimpleTCN(
        in_dim=3,
        num_classes=2,
        channels=(4, 4),
        dropout=0.0,
        use_attention=True,
    )
    pooled_model = SimpleTCN(
        in_dim=3,
        num_classes=3,
        channels=(4, 4),
        dropout=0.0,
        use_attention=False,
    )

    attention_out = attention_model(x)
    pooled_out = pooled_model(x)

    assert attention_out.shape == (2, 2)
    assert pooled_out.shape == (2, 3)


def test_phase_temporal_block_and_phase_tcn_forward_shapes() -> None:
    block = PhaseTemporalBlock(in_ch=3, out_ch=5, d=2)
    phase_model = PhaseTCN(in_dim=3, num_classes=2)
    x = torch.randn(1, 3, 10)
    phase_x = torch.randn(2, 10, 3)

    block_out = block(x)
    phase_out = phase_model(phase_x)

    assert block_out.shape == (1, 5, 10)
    assert phase_out.shape == (2, 10, 2)
