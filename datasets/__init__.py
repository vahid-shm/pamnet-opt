"""Datasets module for the PAMNet-QM9 project.

Currently only the QM9 dataset is provided.  Additional datasets from
the original repository have been removed to simplify the codebase and
reduce dependency complexity.
"""

from .qm9_dataset import QM9  # noqa: F401

__all__ = [
    "QM9",
]
