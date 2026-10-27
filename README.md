# llm-sailresearch
[![Tests](https://github.com/james2doyle/llm-sailresearch/actions/workflows/test.yml/badge.svg)](https://github.com/james2doyle/llm-sailresearch/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/james2doyle/llm-sailresearch/blob/main/LICENSE)

Access [Sail Research](https://sailresearch.com/) hosted models via API

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).

```bash
llm install 'https://github.com/james2doyle/llm-sailresearch/archive/refs/heads/main.zip'
```

Or directly with pip:

```bash
pip install git+https://github.com/james2doyle/llm-sailresearch
```

## Usage

Obtain a [Sail API key](https://app.sailresearch.com) and save it like this:

```bash
llm keys set sail
# <Paste key here>
```

Or export it as an environment variable:

```bash
export SAIL_API_KEY=...
```

Run `llm models` to get the list of models. Model IDs are namespaced, e.g.
`zai-org/GLM-5.3`, `moonshotai/Kimi-K3` or `deepseek-ai/DeepSeek-V4-Pro-0813`.

Run prompts like this:

```bash
llm -m zai-org/GLM-5.3 'five great names for a pet ocelot'
```

Reasoning effort accepts `none`, `minimal`, `low`, `medium`, `high`, `xhigh` or `max` (`max` is the same tier as `xhigh`):

```bash
llm -m zai-org/GLM-5.3 'Explain diffusion models' -o reasoning_effort xhigh
```

The `moonshotai/Kimi-K3`, `moonshotai/Kimi-K2.6`, `google/gemma-4-31B-it` and
`nvidia/Gemma-4-31B-IT-NVFP4` models are multimodal and accept image
attachments (JPEG, PNG, WebP or GIF):

```bash
llm -m moonshotai/Kimi-K3 -a cat.jpg 'What is in this image?'
```

## Options

Sail does not support every standard OpenAI option, so this plugin only exposes
the ones the API accepts: `temperature`, `top_p`, `max_completion_tokens`,
`reasoning_effort`, `json_object`, `image_detail` and `prompt_cache_key`.

Notes:

- Reasoning models think using the `max_completion_tokens` budget. If you get
  an empty response, try a higher value.
- `zai-org/GLM-5.3` does not accept `reasoning_effort none` - use `low` for its
  lowest native reasoning tier.
- Thinking output (`reasoning_content`) is not displayed by LLM, but it is
  stored in the logs via `response_json`.
- PDF attachments are not supported by this plugin, only images.
- Model capacity is shared across all clients: under heavy load a model can
  hang instead of erroring (Sail has no public status page for this). If a
  specific model stalls, try another - e.g. if `google/gemma-4-12B-it` is
  slow, `google/gemma-4-31B-it` typically works.

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:

```bash
cd llm-sailresearch
python3 -m venv venv
source venv/bin/activate
```

Now install the dependencies and test dependencies:

```bash
llm install -e '.[test]'
```

To run the tests:

```bash
pytest
```
