import pytest
from llm.plugins import pm


@pytest.fixture
def registered():
    class Capture:
        def __init__(self):
            self.pairs = []

        def __call__(self, model, async_model=None, aliases=None):
            self.pairs.append((model, async_model, aliases))

    return Capture()


def test_plugin_is_installed(monkeypatch):
    # LLM skips entry-point loading under pytest (sys._called_from_test),
    # so load manually the same way load_plugins() does:
    monkeypatch.setenv("SAIL_API_KEY", "test-key")
    pm.load_setuptools_entrypoints("llm")
    try:
        names = [mod.__name__ for mod in pm.get_plugins()]
        assert "llm_sailresearch" in names
    finally:
        for name, mod in pm.list_name_plugin():
            if name == "llm_sailresearch":
                pm.unregister(mod)


def test_models_register_with_key(monkeypatch, registered):
    monkeypatch.setenv("SAIL_API_KEY", "test-key")
    from llm_sailresearch import SailChat, MODELS, register_models

    register_models(registered)
    assert [(m.model_id, m.model_name) for m, _, _ in registered.pairs] == [
        (f"sailresearch/{model_id}", model_id) for model_id in MODELS
    ]
    for model, async_model, _ in registered.pairs:
        assert isinstance(model, SailChat)
        # Namespaced IDs, raw API IDs in model_name
        assert model.model_id.startswith("sailresearch/")
        assert model.model_name == model.model_id.removeprefix("sailresearch/")
        # Each sync model registered in an (sync, async) pair
        assert async_model.model_id == model.model_id
        assert async_model.model_name == model.model_name


def test_no_registration_without_key(monkeypatch, registered):
    monkeypatch.delenv("SAIL_API_KEY", raising=False)
    from llm_sailresearch import register_models

    register_models(registered)
    assert registered.pairs == []


def test_options_accept_sail_values():
    from llm_sailresearch import SailOptions

    options = SailOptions(
        reasoning_effort="xhigh",
        max_completion_tokens=8192,
        temperature=0.5,
        top_p=0.9,
        image_detail="low",
        prompt_cache_key="my-app-prefix",
    )
    assert options.reasoning_effort == "xhigh"
    assert options.max_completion_tokens == 8192
    assert options.temperature == 0.5
    assert options.top_p == 0.9
    assert options.image_detail == "low"
    assert options.prompt_cache_key == "my-app-prefix"


def test_options_reject_packaged_openai_only_helpers():
    from pydantic import ValidationError

    from llm_sailresearch import SailOptions

    # reasoning_effort validates against the Sail enum
    with pytest.raises(ValidationError):
        SailOptions(reasoning_effort="bogus")
    # temperature/top_p ranges match the API
    with pytest.raises(ValidationError):
        SailOptions(temperature=3)
    with pytest.raises(ValidationError):
        SailOptions(max_completion_tokens=0)


def test_vision_model_attachment_types():
    from llm_sailresearch import SailChat

    kimi = SailChat("moonshotai/Kimi-K3")
    assert "image/jpeg" in kimi.attachment_types
    assert "application/pdf" not in kimi.attachment_types
    glm = SailChat("zai-org/GLM-5.3")
    assert not glm.attachment_types


def test_model_str():
    from llm_sailresearch import SailChat

    assert str(SailChat("zai-org/GLM-5.3")) == "Sail: sailresearch/zai-org/GLM-5.3"
