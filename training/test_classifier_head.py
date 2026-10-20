"""Tests to ensure the fine-tuning run starts with a fresh task-specific head."""

import torch
from torch import nn

from training.train_classifier import reset_task_head


class _FakeSequenceClassifier:
    def __init__(self):
        self.pooler = nn.Sequential(nn.Linear(4, 4))
        self.classifier = nn.Linear(4, 3)
        with torch.no_grad():
            for parameter in self.pooler.parameters():
                parameter.fill_(1.0)
            for parameter in self.classifier.parameters():
                parameter.fill_(1.0)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.constant_(module.weight, 0.123)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)


def test_risk_classification_head_is_explicitly_reinitialized():
    model = _FakeSequenceClassifier()

    reset_names = reset_task_head(model)

    assert "pooler" in reset_names
    assert "classifier" in reset_names
    assert torch.allclose(
        model.classifier.weight,
        torch.full_like(model.classifier.weight, 0.123),
    )
    assert torch.allclose(
        model.pooler[0].weight,
        torch.full_like(model.pooler[0].weight, 0.123),
    )
