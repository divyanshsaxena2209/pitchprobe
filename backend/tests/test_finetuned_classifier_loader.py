"""Regression tests for optional local and Hugging Face Hub risk checkpoints."""

import sys
import types
from types import SimpleNamespace

from backend.analyzer import ModelManager


RISK_ID2LABEL = {0: "LOW", 1: "MODERATE", 2: "HIGH"}


def _install_transformers(monkeypatch, sequence_classifier, tokenizer_calls=None):
    tokenizer_calls = tokenizer_calls if tokenizer_calls is not None else []

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            tokenizer_calls.append((name, kwargs))
            return cls()

    fake_module = types.ModuleType("transformers")
    fake_module.AutoTokenizer = FakeTokenizer
    fake_module.AutoModelForSequenceClassification = sequence_classifier
    fake_module.AutoModelForSeq2SeqLM = object
    monkeypatch.setitem(sys.modules, "transformers", fake_module)
    return tokenizer_calls


def _valid_model_class(id2label=RISK_ID2LABEL, calls=None):
    class FakeModel:
        config = SimpleNamespace(id2label=id2label)

        @classmethod
        def from_pretrained(cls, name, **kwargs):
            if calls is not None:
                calls.append((name, kwargs))
            return cls()

        def to(self, _device):
            return self

        def eval(self):
            return self

    return FakeModel


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
    model_calls = []
    tokenizer_calls = _install_transformers(
        monkeypatch,
        _valid_model_class(calls=model_calls),
    )

    manager = ModelManager()
    manager.try_load_models(background=False)

    expected_kwargs = {
        "local_files_only": False,
        "revision": "0123456789abcdef",
        "token": "test-token-value",
    }
    assert manager.classifier_loaded is True
    assert manager.classifier_mode == "fine_tuned"
    assert manager.classifier_name == "test-owner/pitchprobe-deberta-risk"
    assert tokenizer_calls == [
        ("test-owner/pitchprobe-deberta-risk", expected_kwargs)
    ]
    assert model_calls == tokenizer_calls


def test_local_checkpoint_remains_local_only(monkeypatch, tmp_path):
    monkeypatch.setenv("PITCHPROBE_CLASSIFIER_PATH", str(tmp_path))
    monkeypatch.setenv("PITCHPROBE_CLASSIFIER_REVISION", "ignored-for-local-path")
    monkeypatch.setenv("HF_TOKEN", "unused-for-local-path")
    model_calls = []
    tokenizer_calls = _install_transformers(
        monkeypatch,
        _valid_model_class(calls=model_calls),
    )

    manager = ModelManager()
    manager.try_load_models(background=False)

    expected_call = (str(tmp_path), {"local_files_only": True})
    assert manager.classifier_loaded is True
    assert manager.classifier_mode == "fine_tuned"
    assert tokenizer_calls == [expected_call]
    assert model_calls == [expected_call]


def test_incompatible_pairwise_labels_are_rejected_and_original_nli_fallback_is_kept(
    monkeypatch,
):
    monkeypatch.setenv(
        "PITCHPROBE_CLASSIFIER_PATH",
        "test-owner/scientific-pair-classifier",
    )
    monkeypatch.delenv("PITCHPROBE_CLASSIFIER_REVISION", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    tokenizer_calls = []
    model_calls = []

    class PairwiseModel:
        config = SimpleNamespace(
            id2label={0: "same", 1: "exaggerates", 2: "downplays"}
        )

        def to(self, _device):
            return self

        def eval(self):
            return self

    class SequenceClassifier:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            model_calls.append((name, kwargs))
            if len(model_calls) == 1:
                return PairwiseModel()
            return _valid_model_class(
                id2label={0: "contradiction", 1: "entailment", 2: "neutral"}
            ).from_pretrained(name, **kwargs)

    _install_transformers(monkeypatch, SequenceClassifier, tokenizer_calls)

    manager = ModelManager()
    manager.try_load_models(background=False)

    assert manager.classifier_loaded is True
    assert manager.classifier_mode == "nli"
    assert manager.classifier_name == "cross-encoder/nli-deberta-v3-base"
    assert model_calls[0][0] == "test-owner/scientific-pair-classifier"
    assert model_calls[1][0] == "cross-encoder/nli-deberta-v3-base"
    assert len(tokenizer_calls) == 2
