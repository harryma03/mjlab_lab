"""Minimal regression check for Dynawheel swing-clearance shaping."""

import torch

from np3o.tasks.locomotion.cmdp.rewards import _clearance_shortfall


def test_clearance_shortfall() -> None:
    peak_height = torch.tensor([0.120, 0.136, 0.166, 0.196])
    actual = _clearance_shortfall(peak_height, 0.116, 0.05, 0.005)
    expected = torch.tensor([0.0, 0.36, 0.0, 0.0])
    torch.testing.assert_close(actual, expected, atol=1.0e-6, rtol=1.0e-6)


if __name__ == "__main__":
    test_clearance_shortfall()
    print("dynawheel clearance self-check passed")
