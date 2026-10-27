import llm
from pydantic import Field

from llm.default_plugins.openai_models import (
    AsyncChat,
    Chat,
    ImageDetailEnum,
    ReasoningEffortEnum,
)

API_BASE = "https://api.sailresearch.com/v1"

MODELS = (
    "moonshotai/Kimi-K3",
    "zai-org/GLM-5.3",
    "zai-org/GLM-5.3-Flash",
    "deepseek-ai/DeepSeek-V4.1-Flash-1017",
    "deepseek-ai/DeepSeek-V4.1-Flash",
    "deepseek-ai/DeepSeek-V4-Pro-0813",
    "deepseek-ai/DeepSeek-V4-Flash-0731",
    "moonshotai/Kimi-K2.6",
    "google/gemma-4-31B-it",
    "nvidia/Gemma-4-31B-IT-NVFP4",
    "google/gemma-4-12B-it",
    "openai/gpt-oss-120b",
    "Qwen/Qwen3.6-35B-A3B",
)

# Per-model multimodal support documented at docs.sailresearch.com/models.md.
VISION_MODELS = frozenset(
    {
        "moonshotai/Kimi-K3",
        "moonshotai/Kimi-K2.6",
        "google/gemma-4-31B-it",
        "nvidia/Gemma-4-31B-IT-NVFP4",
    }
)

attachment_types = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/gif",
    }
)


class SailOptions(llm.Options):
    # Only options the Sail Chat Completions API accepts. Sail rejects
    # several OpenAI defaults (max_tokens, frequency_penalty,
    # presence_penalty, stop, seed, logit_bias), so nothing here
    # inherits from the OpenAI SharedOptions.
    temperature: float | None = Field(
        description=(
            "What sampling temperature to use, between 0 and 2. Higher values like "
            "0.8 will make the output more random, while lower values like 0.2 will "
            "make it more focused and deterministic."
        ),
        ge=0,
        le=2,
        default=None,
    )
    top_p: float | None = Field(
        description=(
            "An alternative to sampling with temperature, called nucleus sampling, "
            "where the model considers the results of the tokens with top_p "
            "probability mass. So 0.1 means only the tokens comprising the top "
            "10% probability mass are considered. Recommended to use top_p or "
            "temperature but not both."
        ),
        ge=0,
        le=1,
        default=None,
    )
    max_completion_tokens: int | None = Field(
        description=(
            "Maximum number of completion tokens to generate. Reasoning models "
            "spend some of this budget thinking before answering - if you get an "
            "empty response, raise this value."
        ),
        ge=1,
        default=None,
    )
    reasoning_effort: ReasoningEffortEnum | None = Field(
        description=(
            "Constraints effort on reasoning for reasoning models. One of none, "
            "minimal, low, medium, high, xhigh or max (the same tier as xhigh)."
        ),
        default=None,
    )
    json_object: bool | None = Field(
        description="Output a valid JSON object {...}. Prompt must mention JSON.",
        default=None,
    )
    image_detail: ImageDetailEnum | None = Field(
        description=(
            "Controls the detail level for image attachments on multimodal "
            "models. Supported values are auto, low and high."
        ),
        default=None,
    )
    prompt_cache_key: str | None = Field(
        description=(
            "Optional routing hint so requests sharing a large prompt prefix "
            "are preferentially routed for cache hit rates."
        ),
        default=None,
    )


class _SailShared:
    def __init__(self, model_id):
        # Prefix the LLM-facing model_id with our namespace so these
        # models cannot collide with other plugins (e.g. OpenRouter
        # serves the same upstream models). The raw provider ID is kept
        # in model_name, which is what gets sent to the API.
        super().__init__(
            model_id=f"sailresearch/{model_id}",
            model_name=model_id,
            api_base=API_BASE,
            supports_schema=True,
            supports_tools=True,
        )
        if model_id in VISION_MODELS:
            # Images only - Sail also takes PDFs but the OpenAI Chat
            # message building path used here cannot produce file parts
            self.attachment_types = set(attachment_types)

    def __str__(self):
        return "Sail: {}".format(self.model_id)


class SailChat(_SailShared, Chat):
    needs_key = "sail"
    key_env_var = "SAIL_API_KEY"
    Options = SailOptions


class SailAsyncChat(_SailShared, AsyncChat):
    needs_key = "sail"
    key_env_var = "SAIL_API_KEY"
    Options = SailOptions


@llm.hookimpl
def register_models(register):
    # Only do this if the key is set
    key = llm.get_key("", "sail", SailChat.key_env_var)
    if not key:
        return
    for model_id in MODELS:
        register(SailChat(model_id), SailAsyncChat(model_id))
