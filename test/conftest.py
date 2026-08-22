"""Shared fixtures for gas-mapping characterization tests."""

from types import SimpleNamespace

import pytest


@pytest.fixture
def map_message_factory():
    """Create the map-shaped object consumed by the pure model classes."""
    def make(data=None, width=3, height=3, resolution=1.0,
             origin_x=0.0, origin_y=0.0):
        if data is None:
            data = [0] * (width * height)
        position = SimpleNamespace(x=origin_x, y=origin_y)
        origin = SimpleNamespace(position=position)
        info = SimpleNamespace(
            width=width, height=height, resolution=resolution,
            origin=origin)
        return SimpleNamespace(info=info, data=list(data))

    return make
