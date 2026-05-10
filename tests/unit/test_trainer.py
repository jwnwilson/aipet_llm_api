"""Unit tests for domain/train/trainer.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestWeightedTrainerDataLoader:
    """Regression tests for _WeightedTrainer.get_train_dataloader."""

    def _make_trainer(self, sample_weights: list[float] | None = None):
        """Build a _WeightedTrainer with minimal mocked collaborators."""
        from domain.train.trainer import _WeightedTrainer

        args = MagicMock()
        args.per_device_train_batch_size = 1
        # Deliberately do NOT set args.no_cuda — that attribute was removed in
        # transformers 5.x and must not be referenced by our code.
        del args.no_cuda

        dataset = MagicMock()
        dataset.__len__ = lambda self: 4

        trainer = _WeightedTrainer.__new__(_WeightedTrainer)
        trainer.args = args
        trainer.train_dataset = dataset
        trainer.data_collator = MagicMock()
        trainer._sample_weights = sample_weights
        return trainer

    def test_no_cuda_attribute_not_accessed(self):
        """get_train_dataloader must not touch args.no_cuda (removed in transformers 5)."""
        trainer = self._make_trainer(sample_weights=[1.0, 1.0, 1.0, 1.0])
        with patch("torch.cuda.is_available", return_value=False):
            dl = trainer.get_train_dataloader()
        assert dl is not None

    def test_num_workers_zero_when_cuda_unavailable(self):
        """Non-CUDA devices (MPS, CPU) must get num_workers=0 to avoid sharing errors."""
        trainer = self._make_trainer(sample_weights=[1.0, 1.0, 1.0, 1.0])
        with patch("torch.cuda.is_available", return_value=False):
            dl = trainer.get_train_dataloader()
        assert dl.num_workers == 0

    def test_num_workers_two_when_cuda_available(self):
        """CUDA devices should use num_workers=2 for parallel data loading."""
        trainer = self._make_trainer(sample_weights=[1.0, 1.0, 1.0, 1.0])
        with patch("torch.cuda.is_available", return_value=True):
            dl = trainer.get_train_dataloader()
        assert dl.num_workers == 2

    def test_falls_back_to_super_when_no_weights(self):
        """When sample_weights is None the default Trainer DataLoader is used."""
        trainer = self._make_trainer(sample_weights=None)
        with patch.object(type(trainer).__mro__[1], "get_train_dataloader", return_value="super_dl"):
            result = trainer.get_train_dataloader()
        assert result == "super_dl"
