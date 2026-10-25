import json

import httpx
import pytest
import respx
from pydantic import BaseModel

from agentic_tte.llm import (
    LLMOutputError,
    Message,
    OllamaClient,
    OpenAICompatClient,
    ResponseCache,
    ScriptedClient,
)
from agentic_tte.llm.base import extract_json, inline_refs, strip_think


class Inner(BaseModel):
    x: int


class Outer(BaseModel):
    inner: Inner
    items: list[Inner]


MSG = [Message(role="user", content="hi")]


def test_strip_think_and_extract_json():
    assert strip_think("<think>reasoning</think>\n{\"x\": 1}") == '{"x": 1}'
    assert strip_think("lost open tag</think> {\"x\": 1}") == '{"x": 1}'
    assert extract_json('Sure!\n```json\n{"x": 2}\n```') == '{"x": 2}'
    assert extract_json('prefix {"x": 3} suffix') == '{"x": 3}'
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_inline_refs_removes_defs():
    schema = inline_refs(Outer.model_json_schema())
    blob = json.dumps(schema)
    assert "$ref" not in blob and "$defs" not in blob
    assert schema["properties"]["inner"]["properties"]["x"]["type"] == "integer"


def test_structured_retries_with_validation_error():
    llm = ScriptedClient(['{"x": "not-an-int"}', '{"x": 5}'], max_retries=1)
    assert llm.structured(MSG, Inner) == Inner(x=5)
    retry_messages = llm.requests[1][0]
    assert retry_messages[-1].role == "user"
    assert "did not match" in retry_messages[-1].content
    assert llm.requests[0][1]["properties"]["x"]["type"] == "integer"


def test_structured_gives_up():
    llm = ScriptedClient(["nope", "still nope"], max_retries=1)
    with pytest.raises(LLMOutputError) as info:
        llm.structured(MSG, Inner)
    assert info.value.last_text == "still nope"


def test_cache_hit_skips_backend(tmp_path):
    cache = ResponseCache(tmp_path / "c.sqlite")
    llm = ScriptedClient(['{"x": 1}'], cache=cache)
    assert llm.structured(MSG, Inner).x == 1
    llm2 = ScriptedClient([], cache=cache)  # would fail if called
    assert llm2.structured(MSG, Inner).x == 1
    assert llm2.usage.cache_hits == 1 and llm2.usage.calls == 0


@respx.mock
def test_ollama_request_shape():
    respx.get("http://ollama:11434/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "qwen3:32b", "digest": "abc"}]})
    )
    chat = respx.post("http://ollama:11434/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "qwen3:32b",
                "message": {"role": "assistant", "content": "<think>hmm</think>{\"x\": 7}"},
                "prompt_eval_count": 11,
                "eval_count": 3,
            },
        )
    )
    llm = OllamaClient("qwen3:32b", host="http://ollama:11434", seed=42)
    out = llm.structured(MSG, Inner)
    assert out.x == 7
    body = json.loads(chat.calls[0].request.content)
    assert body["stream"] is False
    assert body["format"]["properties"]["x"]["type"] == "integer"
    assert body["options"] == {"temperature": 0.0, "seed": 42, "num_ctx": 16384,
                               "num_predict": 8192}  # fmt: skip
    assert "think" not in body  # only sent when configured
    assert llm.model_digest() == "abc"
    assert llm.usage.prompt_tokens == 11 and llm.usage.completion_tokens == 3


@respx.mock
def test_timeout_is_an_item_failure_and_think_level_is_sent():
    """gpt-oss:20b timed out on whole drafts in the exit run; a timeout must surface as an
    LLMOutputError (handled per item), not crash the caller."""
    respx.get("http://ollama:11434/api/tags").mock(return_value=httpx.Response(200, json={}))
    chat = respx.post("http://ollama:11434/api/chat").mock(side_effect=httpx.ReadTimeout("slow"))
    llm = OllamaClient("gpt-oss:20b", host="http://ollama:11434", think="low")
    with pytest.raises(LLMOutputError, match="ReadTimeout"):
        llm.structured(MSG, Inner)
    assert llm.usage.failures == 1
    assert json.loads(chat.calls[0].request.content)["think"] == "low"
    assert chat.call_count == 1  # not retried


def test_settings_accept_think_levels(monkeypatch):
    from agentic_tte.config import Settings

    for raw, parsed in (("false", False), ("low", "low"), ("none", None)):
        monkeypatch.setenv("ATTE_THINK", raw)
        assert Settings().think == parsed


@respx.mock
def test_ollama_http_error():
    respx.get("http://ollama:11434/api/tags").mock(return_value=httpx.Response(200, json={}))
    respx.post("http://ollama:11434/api/chat").mock(
        return_value=httpx.Response(404, json={"error": "model not found"})
    )
    with pytest.raises(RuntimeError, match="404"):
        OllamaClient("missing", host="http://ollama:11434").complete(MSG)


@respx.mock
def test_openai_compat_request_shape():
    route = respx.post("http://vllm:8000/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "m",
                "choices": [{"message": {"content": '{"x": 9}'}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        )
    )
    llm = OpenAICompatClient("m", base_url="http://vllm:8000/v1")
    assert llm.structured(MSG, Inner).x == 9
    body = json.loads(route.calls[0].request.content)
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"]["properties"]["x"]
