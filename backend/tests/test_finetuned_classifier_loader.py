"""Regression tests for optional local and Hugging Face Hub risk checkpoints."""

import sys
import types
from types import SimpleNamespace

from backend.analyzer import ModelManager


RISK_ID2LABEL = {0: "LOW", 1: "MODERATE", 2: "HIGH"}


def _fake_transformers(monkeypatch, id2label=RISK_ID2LABEL):
    tokenizer_calls = []
    model_calls = []

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            tokenizer_calls.append((name, kwargs))
            return cls()

    class FakeModel:
        config = SimpleNamespace(id2label=id2label)

        @classmethod
        def from_pretrained(cls, name, **kwargs):
            model_calls.append((name, kwargs))
            return cls()

        def to(self, _device):
            return self

        def eval(self):
            return self

    fake_module = types.ModuleType("transformers")
    fake_module.AutoTokenizer = FakeTokenizer
    fake_module.AutoModelForSequenceClassification = FakeModel
    fake_module.AutoModelForSeq2SeqLM = object
    monkeypatch.setitem(sys.modules, "transformers", fake_module)
    return tokenizer_calls, model_calls


def test_no_model_path_preserves_original_nli_and_question_generator_defaults(monkeypatch):
    monkeypatch.delenv("PITCHPROBE_CLASSIFIER_PATH", raising=False)
    monkeypatch.delenv("PITCHPROBE_USE_FLAN_T5", raising=False)

    manager = ModelManager()

    assert manager.classifier_mode == "nli"
    assert manager.classifier_name == "cross-encoder/nli-deberta-v3-base"
    assert manager.use_flan_t5 is False
    assert manager.generator_name == "google/flan-t5-small"


def test_hub_checkpoint_loads_with_revision_and_optional_secret_token(monkeypatch):
    monkeypatch.setenv(
        "PITCHPROBE_CLASSIFIER_PATH",
        "test-owner/pitchprobe-deberta-risk",
    )
    monkeypatch.setenv("PITCHPROBE_CLASSIFIER_REVISION", "0123456789abcdef")
    monkeypatch.setenv("HF_TOKEN", "test-token-value")
    tokenizer_calls, model_calls = _fake_transformers(monkeypatch)

    manager = ModelManager()
    manager.try_load_models(background=False)

    assert manager.classifier_loaded is True
    assert manager.classifier_mode == "fine_tuned"
    assert manager.classifier_name == "test-owner/pitchprobe-deberta-risk"
    assert tokenizer_calls == [
        (
            "test-owner/pitchprobe-deberta-risk",
            {
                "local_files_only": False,
                "revision": "0123456789abcdef",
                "token": "test-token-value",
            },
        )
    ]
    assert model_calls == tokenizer_calls


def test_local_checkpoint_remains_local_only(monkeypatch, tmp_path):
    monkeypatch.setenv("PITCHPROBE_CLASSIFIER_PATH", str(tmp_path))
    monkeypatch.setenv("PITCHPROBE_CLASSIFIER_REVISION", "ignored-for-local-path")
    monkeypatch.setenv("HF_TOKEN", "unused-for-local-path")
    tokenizer_calls, model_calls = _fake_transformers(monkeypatch)

    manager = ModelManager()
    manager.try_load_models(background=False)

    assert manager.classifier_loaded is True
    assert manager.classifier_mode == "fine_tuned"
    assert tokenizer_calls == [
        (str(tmp_path), {"local_files_only": True})
    ]
    assert model_calls == tokenizer_calls


def test_incompatible_pairwise_labels_are_not_accepted_as_risk_labels(monkeypatch):
    monkeypatch.setenv(
        "PITCHPROBE_CLASSIFIER_PATH",
        "test-owner/scientific-pair-classifier",
    )
    monkeypatch.delenv("PITCHPROBE_CLASSIFIER_REVISION", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    tokenizer_calls, model_calls = _fake_transformers(
        monkeypatch,
        {0: "same", 1: "exaggerates", 2: "downplays"},
    )

    # First call is the incompatible configured checkpoint. The second load is
    # the original NLI fallback; return a valid NLI-shaped fake for that call.
    class FallbackModel:
        config = SimpleNamespace(
            id2label={0: "contradiction", 1: "entailment", 2: "neutral"}
        )

        @classmethod
        def from_pretrained(cls, name, **kwargs):
            model_calls.append((name, kwargs))
            return cls()

        def to(self, _device):
            return self

        def eval(self):
            return self

    fake_module = types.ModuleType("transformers")

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            tokenizer_calls.append((name, kwargs))
            return cls()

    # First model call returns pairwise labels; subsequent NLI fallback is a
    # different model class with standard NLI labels.
    bad = _fake_transformers(
        monkeypatch,
        {0: "same", 1: "exaggerates", 2: "downplays"},
    )

    calls = {"count": 0}

    class SequenceClassifier:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                return bad_model()
            return FallbackModel.from_pretrained(name, **kwargs)

    class BadModel:
        config = SimpleNamespace(
            id2label={0: "same", 1: "exaggerates", 2: "downplays"}
        )

        def to(self, _device):
            return self

        def eval(self):
            return self

    def bad_model():
        return BadModel()

    fake_module.AutoTokenizer = Tokenizer
    fake_module.AutoModelForSequenceClassification = SequenceClassifier
    fake_module.AutoModelForSeq2SeqLM = object
    monkeypatch.setitem(sys.modules, "transformers", fake_module)

    manager = ModelManager()
    manager.try_load_models(background=False)

    assert manager.classifier_loaded is True
    assert manager.classifier_mode == "nli"
    assert manager.classifier_name == "cross-encoder/nli-deberta-v3-base"
