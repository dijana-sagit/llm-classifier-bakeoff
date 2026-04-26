"""Zero-shot LLM tests with a fake provider.

The fake satisfies the ``LLMProvider`` Protocol via structural typing —
no inheritance, no patching of registries. Demonstrates that the Adapter
abstraction is real: any object with the right shape works.
"""
from __future__ import annotations

from dataclasses import dataclass

from data.banking77 import LabelledExample
from methods.zero_shot import UNPARSED_LABEL, ZeroShotLLM, parse_label
from providers.base import LLMProvider, ProviderResponse


@dataclass(slots=True)
class _FakeProvider:
    """Hand-rolled stand-in for an LLMProvider."""

    name: str = "fake"
    next_text: str = "card_arrival"
    calls: int = 0

    def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 64,
        temperature: float = 0.0,
        cache_system: bool = False,
    ) -> ProviderResponse:
        self.calls += 1
        return ProviderResponse(
            text=self.next_text,
            input_tokens=100,
            output_tokens=5,
            cost_usd=0.0001,
            latency_ms=12.0,
            model=model,
        )


def test_fake_provider_satisfies_protocol():
    assert isinstance(_FakeProvider(), LLMProvider)


def test_parse_label_exact_match():
    assert parse_label("card_arrival", ["card_arrival", "lost_card"]) == "card_arrival"


def test_parse_label_strips_prefix_and_punctuation():
    assert parse_label("Label: card_arrival.", ["card_arrival"]) == "card_arrival"
    assert parse_label('"card_arrival"', ["card_arrival"]) == "card_arrival"


def test_parse_label_handles_spaces_to_underscores():
    assert parse_label("card arrival", ["card_arrival"]) == "card_arrival"


def test_parse_label_substring_fallback():
    response = "I think this is card_arrival based on the wording"
    assert parse_label(response, ["card_arrival", "lost_card"]) == "card_arrival"


def test_parse_label_returns_sentinel_on_no_match():
    assert parse_label("totally unrelated text", ["x", "y"]) == UNPARSED_LABEL


def test_zero_shot_predict_uses_provider_response(monkeypatch):
    fake = _FakeProvider(next_text="card_arrival")

    # Bypass the disk-cache / singleton-registry path; route directly to the fake.
    from methods import zero_shot as zs

    def _direct_complete(
        *,
        provider_name,
        model,
        system,
        user,
        max_tokens,
        temperature,
        cache_system,
    ):
        return fake.complete(
            model=model,
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
            cache_system=cache_system,
        )

    monkeypatch.setattr(zs, "cached_complete", _direct_complete)

    method = ZeroShotLLM(provider=fake, model="fake-1")
    method.fit([LabelledExample("hello", "card_arrival"), LabelledExample("hi", "lost_card")])
    pred = method.predict("when will my card arrive?")

    assert pred.predicted_label == "card_arrival"
    assert pred.cost_usd == 0.0001
    assert pred.raw_response == "card_arrival"
    assert fake.calls == 1


def test_zero_shot_records_unparsed_when_response_is_garbage(monkeypatch):
    fake = _FakeProvider(next_text="???")
    from methods import zero_shot as zs

    monkeypatch.setattr(
        zs,
        "cached_complete",
        lambda **kw: fake.complete(
            model=kw["model"],
            system=kw["system"],
            user=kw["user"],
            max_tokens=kw["max_tokens"],
            temperature=kw["temperature"],
            cache_system=kw["cache_system"],
        ),
    )

    method = ZeroShotLLM(provider=fake, model="fake-1")
    method.fit([LabelledExample("hello", "card_arrival")])
    pred = method.predict("anything")
    assert pred.predicted_label == UNPARSED_LABEL


def test_zero_shot_raises_if_predict_called_before_fit():
    method = ZeroShotLLM(provider=_FakeProvider(), model="fake-1")
    try:
        method.predict("anything")
    except RuntimeError as exc:
        assert "fit" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when predict precedes fit.")
