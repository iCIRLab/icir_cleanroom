"""Backward-compatible imports for the renamed gas environment adapter."""

from .environment_node import GasEnvironmentNode, main


EmptyGasMapNode = GasEnvironmentNode


__all__ = ['EmptyGasMapNode', 'GasEnvironmentNode', 'main']
