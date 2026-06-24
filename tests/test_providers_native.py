"""Unit tests for native provider implementations (OpenAI, Anthropic, Gemini).

These tests mock the underlying SDKs so they run fully offline without API keys.
They verify that each provider:

1. Uses its *own* SDK (not the OpenAI compatibility shim for non-OpenAI providers)
2. Correctly converts SDK intermediate messages / tools into native formats
3. Normalises responses (text, tool_calls, usage, finish_reason) into the
   common shape expected by :func:`ai_sdk.generate_text`
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from ai_sdk.providers.anthropic import (
    AnthropicModel,
    _extract_system_and_messages,
    _map_stop_reason,
)
from ai_sdk.providers.anthropic import (
    _normalise_tools as _anthropic_normalise_tools,
)
from ai_sdk.providers.anthropic import (
    _parse_response as _anthropic_parse_response,
)
from ai_sdk.providers.anthropic import (
    _prepare_request_kwargs as _anthropic_prepare_kwargs,
)
from ai_sdk.providers.gemini import (
    GeminiModel,
    _extract_system_and_contents,
    _map_finish_reason,
)
from ai_sdk.providers.gemini import (
    _build_config as _gemini_build_config,
)
from ai_sdk.providers.gemini import (
    _normalise_tools as _gemini_normalise_tools,
)
from ai_sdk.providers.gemini import (
    _parse_response as _gemini_parse_response,
)
from ai_sdk.providers.openai import (
    OpenAIModel,
    _build_chat_messages,
    _normalise_openai_tools,
)
from ai_sdk.tool import Tool, tool

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


class AddParams(BaseModel):
    a: float = Field(description="First number")
    b: float = Field(description="Second number")


@tool(name="add", description="Add two numbers.", parameters=AddParams)
def add_tool(a: float, b: float) -> float:
    return a + b


class AckSchema(BaseModel):
    ack: str


# ===========================================================================
# Tool helpers
# ===========================================================================


class TestToolProviderFormats:
    """Verify each provider-specific tool serialisation helper."""

    def test_to_openai_dict_shape(self):
        d = add_tool.to_openai_dict()
        assert d["type"] == "function"
        assert d["function"]["name"] == "add"
        assert d["function"]["description"] == "Add two numbers."
        assert "properties" in d["function"]["parameters"]

    def test_to_anthropic_dict_shape(self):
        d = add_tool.to_anthropic_dict()
        assert d["name"] == "add"
        assert d["description"] == "Add two numbers."
        assert "input_schema" in d
        assert "properties" in d["input_schema"]
        # Anthropic must NOT use OpenAI's nested function wrapper.
        assert "function" not in d
        assert "type" not in d

    def test_to_gemini_dict_shape(self):
        d = add_tool.to_gemini_dict()
        assert d["name"] == "add"
        assert d["description"] == "Add two numbers."
        assert "parameters" in d
        assert "input_schema" not in d


# ===========================================================================
# OpenAI provider
# ===========================================================================


class TestOpenAIProviderHelpers:
    def test_build_chat_messages_prompt_only(self):
        msgs = _build_chat_messages(prompt="Hi", system=None, messages=None)
        assert msgs == [{"role": "user", "content": "Hi"}]

    def test_build_chat_messages_with_system(self):
        msgs = _build_chat_messages(prompt="Hi", system="Be helpful", messages=None)
        assert msgs[0] == {"role": "system", "content": "Be helpful"}
        assert msgs[1] == {"role": "user", "content": "Hi"}

    def test_build_chat_messages_tool_result_flattening(self):
        messages = [
            {
                "role": "tool",
                "content": [
                    {
                        "toolCallId": "call-1",
                        "toolName": "add",
                        "result": "10",
                        "type": "tool-result",
                    }
                ],
            }
        ]
        msgs = _build_chat_messages(prompt=None, system=None, messages=messages)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "tool"
        assert msgs[0]["tool_call_id"] == "call-1"
        assert msgs[0]["content"] == "10"

    def test_normalise_tools_from_tool_instances(self):
        kwargs = _normalise_openai_tools({"tools": [add_tool], "temperature": 0.5})
        assert len(kwargs["tools"]) == 1
        assert kwargs["tools"][0]["type"] == "function"
        assert kwargs["tools"][0]["function"]["name"] == "add"
        assert kwargs["temperature"] == 0.5

    def test_normalise_tools_from_anthropic_shape(self):
        anthropic_tool = {
            "name": "add",
            "description": "Add",
            "input_schema": {"type": "object", "properties": {}},
        }
        kwargs = _normalise_openai_tools({"tools": [anthropic_tool]})
        assert kwargs["tools"][0]["type"] == "function"
        assert kwargs["tools"][0]["function"]["name"] == "add"

    def test_normalise_tools_passthrough_openai_shape(self):
        openai_tool = add_tool.to_openai_dict()
        kwargs = _normalise_openai_tools({"tools": [openai_tool]})
        assert kwargs["tools"][0] is openai_tool or kwargs["tools"][0] == openai_tool


class TestOpenAIModel:
    def _make_completion_response(
        self,
        *,
        text: str = "hello",
        finish_reason: str = "stop",
        tool_calls: list[Any] | None = None,
        prompt_tokens: int = 5,
        completion_tokens: int = 3,
    ):
        message = SimpleNamespace(content=text, tool_calls=tool_calls)
        choice = SimpleNamespace(message=message, finish_reason=finish_reason)
        usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            model_dump=lambda: {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )
        return SimpleNamespace(choices=[choice], usage=usage, id="chatcmpl-test")

    @patch("ai_sdk.providers.openai._openai.OpenAI")
    def test_generate_text_basic(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = (
            self._make_completion_response(text="hi there")
        )

        model = OpenAIModel("gpt-4o-mini", api_key="sk-test")
        result = model.generate_text(prompt="Say hi")

        assert result["text"] == "hi there"
        assert result["finish_reason"] == "stop"
        assert result["usage"]["total_tokens"] == 8
        assert result["tool_calls"] is None

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["messages"][-1]["role"] == "user"

    @patch("ai_sdk.providers.openai._openai.OpenAI")
    def test_generate_text_with_tool_instances(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        fn_call = SimpleNamespace(
            id="call-abc",
            function=SimpleNamespace(
                name="add", arguments=json.dumps({"a": 1, "b": 2})
            ),
        )
        mock_client.chat.completions.create.return_value = (
            self._make_completion_response(
                text="", finish_reason="tool_calls", tool_calls=[fn_call]
            )
        )

        model = OpenAIModel("gpt-4o-mini", api_key="sk-test")
        result = model.generate_text(
            prompt="What is 1+2?", tools=[add_tool], tool_choice="auto"
        )

        assert result["finish_reason"] == "tool"
        assert result["tool_calls"] is not None
        assert result["tool_calls"][0]["tool_name"] == "add"
        assert result["tool_calls"][0]["args"] == {"a": 1, "b": 2}

        # Tools must have been converted to OpenAI function format.
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["tools"][0]["type"] == "function"
        assert call_kwargs["tools"][0]["function"]["name"] == "add"

    @patch("ai_sdk.providers.openai._openai.OpenAI")
    def test_generate_text_requires_prompt_or_messages(self, mock_openai_cls):
        mock_openai_cls.return_value = MagicMock()
        model = OpenAIModel("gpt-4o-mini", api_key="sk-test")
        with pytest.raises(ValueError, match="prompt.*messages"):
            model.generate_text()

    @patch("ai_sdk.providers.openai._openai.OpenAI")
    def test_generate_object_uses_parse(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        parsed = AckSchema(ack="yes")
        message = SimpleNamespace(content='{"ack": "yes"}', parsed=parsed)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        usage = SimpleNamespace(
            model_dump=lambda: {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            }
        )
        mock_client.chat.completions.parse.return_value = SimpleNamespace(
            choices=[choice], usage=usage
        )

        model = OpenAIModel("gpt-4o-mini", api_key="sk-test")
        result = model.generate_object(schema=AckSchema, prompt="ack yes")

        assert result["object"] == parsed
        assert result["raw_text"] == '{"ack": "yes"}'
        mock_client.chat.completions.parse.assert_called_once()


# ===========================================================================
# Anthropic provider
# ===========================================================================


class TestAnthropicHelpers:
    def test_extract_system_prompt_only(self):
        system, msgs = _extract_system_and_messages(
            prompt="Hello", system="Be nice", messages=None
        )
        assert system == "Be nice"
        assert msgs == [{"role": "user", "content": "Hello"}]

    def test_extract_system_from_messages(self):
        messages = [
            {"role": "system", "content": "Sys msg"},
            {"role": "user", "content": "Hi"},
        ]
        system, msgs = _extract_system_and_messages(
            prompt=None, system=None, messages=messages
        )
        assert system == "Sys msg"
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_extract_tool_result_as_user_tool_result_block(self):
        messages = [
            {
                "role": "tool",
                "content": [
                    {
                        "toolCallId": "toolu_1",
                        "toolName": "add",
                        "result": "3",
                        "type": "tool-result",
                    }
                ],
            }
        ]
        system, msgs = _extract_system_and_messages(
            prompt=None, system=None, messages=messages
        )
        assert system is None
        assert msgs[0]["role"] == "user"
        block = msgs[0]["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "toolu_1"
        assert block["content"] == "3"

    def test_extract_assistant_openai_tool_calls(self):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "add",
                            "arguments": json.dumps({"a": 1, "b": 2}),
                        },
                    }
                ],
            }
        ]
        _, msgs = _extract_system_and_messages(
            prompt=None, system=None, messages=messages
        )
        assert msgs[0]["role"] == "assistant"
        block = msgs[0]["content"][0]
        assert block["type"] == "tool_use"
        assert block["id"] == "call-1"
        assert block["name"] == "add"
        assert block["input"] == {"a": 1, "b": 2}

    def test_normalise_tools_from_tool_instance(self):
        result = _anthropic_normalise_tools([add_tool])
        assert result is not None
        assert result[0]["name"] == "add"
        assert "input_schema" in result[0]

    def test_normalise_tools_from_openai_shape(self):
        result = _anthropic_normalise_tools([add_tool.to_openai_dict()])
        assert result is not None
        assert result[0]["name"] == "add"
        assert "input_schema" in result[0]

    def test_map_stop_reason(self):
        assert _map_stop_reason("end_turn") == "stop"
        assert _map_stop_reason("tool_use") == "tool"
        assert _map_stop_reason("max_tokens") == "length"
        assert _map_stop_reason("stop_sequence") == "stop"
        assert _map_stop_reason(None) == "unknown"

    def test_prepare_request_kwargs_defaults_max_tokens(self):
        kwargs = _anthropic_prepare_kwargs({}, {})
        assert kwargs["max_tokens"] == 8192

    def test_prepare_request_kwargs_tool_choice_auto(self):
        kwargs = _anthropic_prepare_kwargs(
            {}, {"tool_choice": "auto", "tools": [add_tool]}
        )
        assert kwargs["tool_choice"] == {"type": "auto"}
        assert kwargs["tools"][0]["name"] == "add"

    def test_parse_response_text_only(self):
        text_block = SimpleNamespace(type="text", text="Hello Claude")
        usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        resp = SimpleNamespace(
            content=[text_block],
            stop_reason="end_turn",
            usage=usage,
            model="claude-test",
            id="msg_123",
        )
        result = _anthropic_parse_response(resp)
        assert result["text"] == "Hello Claude"
        assert result["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5
        assert result["usage"]["total_tokens"] == 15
        assert result["tool_calls"] is None
        assert result["provider_metadata"]["anthropic"]["id"] == "msg_123"

    def test_parse_response_with_tool_use(self):
        tool_block = SimpleNamespace(
            type="tool_use", id="toolu_abc", name="add", input={"a": 1, "b": 2}
        )
        resp = SimpleNamespace(
            content=[tool_block],
            stop_reason="tool_use",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            model="claude-test",
            id="msg_456",
        )
        result = _anthropic_parse_response(resp)
        assert result["finish_reason"] == "tool"
        assert result["tool_calls"][0]["tool_call_id"] == "toolu_abc"
        assert result["tool_calls"][0]["tool_name"] == "add"
        assert result["tool_calls"][0]["args"] == {"a": 1, "b": 2}


class TestAnthropicModel:
    @patch("ai_sdk.providers.anthropic._anthropic.Anthropic")
    def test_generate_text_uses_native_messages_api(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        text_block = SimpleNamespace(type="text", text="Hi from Claude")
        usage = SimpleNamespace(input_tokens=4, output_tokens=6)
        mock_client.messages.create.return_value = SimpleNamespace(
            content=[text_block],
            stop_reason="end_turn",
            usage=usage,
            model="claude-sonnet-4-20250514",
            id="msg_test",
        )

        model = AnthropicModel("claude-sonnet-4-20250514", api_key="sk-ant-test")
        result = model.generate_text(prompt="Hello", system="Be brief")

        assert result["text"] == "Hi from Claude"
        assert result["finish_reason"] == "stop"

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"
        assert call_kwargs["system"] == "Be brief"
        assert call_kwargs["messages"] == [{"role": "user", "content": "Hello"}]
        assert call_kwargs["max_tokens"] == 8192

    @patch("ai_sdk.providers.anthropic._anthropic.Anthropic")
    def test_generate_text_with_tools(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        tool_block = SimpleNamespace(
            type="tool_use", id="toolu_1", name="add", input={"a": 3, "b": 4}
        )
        mock_client.messages.create.return_value = SimpleNamespace(
            content=[tool_block],
            stop_reason="tool_use",
            usage=SimpleNamespace(input_tokens=2, output_tokens=2),
            model="claude-test",
            id="msg_tool",
        )

        model = AnthropicModel("claude-test", api_key="sk-ant-test")
        result = model.generate_text(
            prompt="3+4?", tools=[add_tool], tool_choice="auto"
        )

        assert result["finish_reason"] == "tool"
        assert result["tool_calls"][0]["tool_name"] == "add"

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["tools"][0]["name"] == "add"
        assert "input_schema" in call_kwargs["tools"][0]
        assert call_kwargs["tool_choice"] == {"type": "auto"}

    @patch("ai_sdk.providers.anthropic._anthropic.Anthropic")
    def test_generate_text_requires_input(self, mock_anthropic_cls):
        mock_anthropic_cls.return_value = MagicMock()
        model = AnthropicModel("claude-test", api_key="sk-ant-test")
        with pytest.raises(ValueError, match="prompt.*messages"):
            model.generate_text()

    @patch("ai_sdk.providers.anthropic._anthropic.Anthropic")
    def test_generate_object_parses_json(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        text_block = SimpleNamespace(type="text", text='{"ack": "ok"}')
        mock_client.messages.create.return_value = SimpleNamespace(
            content=[text_block],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            model="claude-test",
            id="msg_obj",
        )

        model = AnthropicModel("claude-test", api_key="sk-ant-test")
        result = model.generate_object(schema=AckSchema, prompt="ack")

        assert isinstance(result["object"], AckSchema)
        assert result["object"].ack == "ok"

        # System prompt should contain the schema instruction.
        call_kwargs = mock_client.messages.create.call_args.kwargs
        system_text = call_kwargs["system"]
        assert "JSON" in system_text or "json" in system_text.lower()

    @patch("ai_sdk.providers.anthropic._anthropic.Anthropic")
    def test_uses_env_api_key(self, mock_anthropic_cls):
        mock_anthropic_cls.return_value = MagicMock()
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-from-env"}):
            AnthropicModel("claude-test")
        call_kwargs = mock_anthropic_cls.call_args.kwargs
        assert call_kwargs.get("api_key") == "sk-from-env"


# ===========================================================================
# Gemini provider
# ===========================================================================


class TestGeminiHelpers:
    def test_extract_contents_prompt_only(self):
        system, contents = _extract_system_and_contents(
            prompt="Hello Gemini", system="Be concise", messages=None
        )
        assert system == "Be concise"
        assert len(contents) == 1
        assert contents[0].role == "user"

    def test_extract_system_from_messages(self):
        messages = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Hi"},
        ]
        system, contents = _extract_system_and_contents(
            prompt=None, system=None, messages=messages
        )
        assert system == "Sys"
        assert len(contents) == 1
        assert contents[0].role == "user"

    def test_extract_assistant_tool_calls_as_model_function_call(self):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "add",
                            "arguments": json.dumps({"a": 5, "b": 6}),
                        },
                    }
                ],
            }
        ]
        _, contents = _extract_system_and_contents(
            prompt=None, system=None, messages=messages
        )
        assert contents[0].role == "model"
        parts = contents[0].parts
        assert any(getattr(p, "function_call", None) is not None for p in parts)

    def test_extract_tool_result_as_user_function_response(self):
        messages = [
            {
                "role": "tool",
                "content": [
                    {
                        "toolCallId": "call-1",
                        "toolName": "add",
                        "result": "11",
                        "type": "tool-result",
                    }
                ],
            }
        ]
        _, contents = _extract_system_and_contents(
            prompt=None, system=None, messages=messages
        )
        assert contents[0].role == "user"
        parts = contents[0].parts
        assert any(getattr(p, "function_response", None) is not None for p in parts)

    def test_normalise_tools_from_tool_instance(self):
        result = _gemini_normalise_tools([add_tool])
        assert result is not None
        assert len(result) == 1
        decls = result[0].function_declarations
        assert decls is not None
        assert decls[0].name == "add"

    def test_normalise_tools_from_openai_shape(self):
        result = _gemini_normalise_tools([add_tool.to_openai_dict()])
        assert result is not None
        assert result[0].function_declarations[0].name == "add"

    def test_map_finish_reason(self):
        assert _map_finish_reason("STOP") == "stop"
        assert _map_finish_reason("MAX_TOKENS") == "length"
        assert _map_finish_reason("SAFETY") == "content-filter"
        assert _map_finish_reason(None) == "unknown"
        reason = SimpleNamespace(name="STOP")
        assert _map_finish_reason(reason) == "stop"

    def test_build_config_maps_openai_kwargs(self):
        config, extra = _gemini_build_config(
            system_instruction="Sys",
            default_kwargs={},
            kwargs={
                "temperature": 0.2,
                "max_tokens": 100,
                "tools": [add_tool],
                "tool_choice": "auto",
            },
        )
        assert config.system_instruction == "Sys"
        assert config.temperature == 0.2
        assert config.max_output_tokens == 100
        assert config.tools is not None
        assert config.tool_config is not None

    def test_parse_response_text_only(self):
        part = SimpleNamespace(text="Hello Gemini", function_call=None)
        content = SimpleNamespace(parts=[part])
        candidate = SimpleNamespace(content=content, finish_reason="STOP")
        usage = SimpleNamespace(
            prompt_token_count=7,
            candidates_token_count=4,
            total_token_count=11,
        )
        resp = SimpleNamespace(
            candidates=[candidate],
            usage_metadata=usage,
            model_version="gemini-2.0-flash",
            response_id="resp_1",
            text="Hello Gemini",
        )
        result = _gemini_parse_response(resp)
        assert result["text"] == "Hello Gemini"
        assert result["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 7
        assert result["usage"]["completion_tokens"] == 4
        assert result["usage"]["total_tokens"] == 11
        assert result["tool_calls"] is None

    def test_parse_response_with_function_call(self):
        fc = SimpleNamespace(name="add", args={"a": 1, "b": 2}, id=None)
        part = SimpleNamespace(text=None, function_call=fc)
        content = SimpleNamespace(parts=[part])
        candidate = SimpleNamespace(content=content, finish_reason="STOP")
        resp = SimpleNamespace(
            candidates=[candidate],
            usage_metadata=None,
            model_version="gemini-test",
            response_id="resp_2",
            text=None,
        )

        result = _gemini_parse_response(resp)
        assert result["finish_reason"] == "tool"
        assert result["tool_calls"][0]["tool_name"] == "add"
        assert result["tool_calls"][0]["args"] == {"a": 1, "b": 2}
        assert result["tool_calls"][0]["tool_call_id"].startswith("call_")


class TestGeminiModel:
    @patch("ai_sdk.providers.gemini._genai.Client")
    def test_generate_text_uses_native_generate_content(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        part = SimpleNamespace(text="Hi from Gemini", function_call=None)
        content = SimpleNamespace(parts=[part])
        candidate = SimpleNamespace(content=content, finish_reason="STOP")
        usage = SimpleNamespace(
            prompt_token_count=3,
            candidates_token_count=5,
            total_token_count=8,
        )
        mock_resp = SimpleNamespace(
            candidates=[candidate],
            usage_metadata=usage,
            model_version="gemini-2.0-flash",
            response_id="r1",
            text="Hi from Gemini",
        )
        mock_client.models.generate_content.return_value = mock_resp

        model = GeminiModel("gemini-2.0-flash", api_key="gem-test-key")
        result = model.generate_text(prompt="Hello", system="Be nice")

        assert result["text"] == "Hi from Gemini"
        assert result["finish_reason"] == "stop"
        assert result["usage"]["total_tokens"] == 8

        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        assert call_kwargs["model"] == "gemini-2.0-flash"
        assert call_kwargs["contents"] is not None
        assert call_kwargs["config"] is not None
        assert call_kwargs["config"].system_instruction == "Be nice"

    @patch("ai_sdk.providers.gemini._genai.Client")
    def test_generate_text_with_tools(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        fc = SimpleNamespace(name="add", args={"a": 2, "b": 3}, id="fc_1")
        part = SimpleNamespace(text=None, function_call=fc)
        content = SimpleNamespace(parts=[part])
        candidate = SimpleNamespace(content=content, finish_reason="STOP")
        mock_resp = SimpleNamespace(
            candidates=[candidate],
            usage_metadata=None,
            model_version="gemini-test",
            response_id="r2",
            text=None,
        )
        mock_client.models.generate_content.return_value = mock_resp

        model = GeminiModel("gemini-test", api_key="gem-test-key")
        result = model.generate_text(
            prompt="2+3?", tools=[add_tool], tool_choice="auto"
        )

        assert result["finish_reason"] == "tool"
        assert result["tool_calls"][0]["tool_name"] == "add"
        assert result["tool_calls"][0]["args"] == {"a": 2, "b": 3}

        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        config = call_kwargs["config"]
        assert config.tools is not None
        assert config.tools[0].function_declarations[0].name == "add"

    @patch("ai_sdk.providers.gemini._genai.Client")
    def test_generate_text_requires_input(self, mock_client_cls):
        mock_client_cls.return_value = MagicMock()
        model = GeminiModel("gemini-test", api_key="gem-test-key")
        with pytest.raises(ValueError, match="prompt.*messages"):
            model.generate_text()

    @patch("ai_sdk.providers.gemini._genai.Client")
    def test_generate_object_parses_json(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        part = SimpleNamespace(text='{"ack": "yes"}', function_call=None)
        content = SimpleNamespace(parts=[part])
        candidate = SimpleNamespace(content=content, finish_reason="STOP")
        mock_resp = SimpleNamespace(
            candidates=[candidate],
            usage_metadata=None,
            model_version="gemini-test",
            response_id="r3",
            text='{"ack": "yes"}',
        )
        mock_client.models.generate_content.return_value = mock_resp

        model = GeminiModel("gemini-test", api_key="gem-test-key")
        result = model.generate_object(schema=AckSchema, prompt="ack")

        assert isinstance(result["object"], AckSchema)
        assert result["object"].ack == "yes"

        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        config = call_kwargs["config"]
        assert config.response_mime_type == "application/json"

    @patch("ai_sdk.providers.gemini._genai.Client")
    def test_uses_gemini_api_key_env(self, mock_client_cls):
        mock_client_cls.return_value = MagicMock()
        with patch.dict("os.environ", {"GEMINI_API_KEY": "gem-from-env"}, clear=False):
            GeminiModel("gemini-test")
        assert mock_client_cls.called

    @patch("ai_sdk.providers.gemini._genai.Client")
    def test_factory_ignores_legacy_base_url(self, mock_client_cls):
        """Previous OpenAI-compat implementation accepted base_url; ignore it."""
        from ai_sdk.providers.gemini import gemini

        mock_client_cls.return_value = MagicMock()
        model = gemini("gemini-test", api_key="k", base_url="https://ignored.example")
        assert isinstance(model, GeminiModel)


# ===========================================================================
# Cross-provider integration via generate_text helper
# ===========================================================================


class TestGenerateTextPassesToolInstances:
    """Ensure the high-level helper forwards Tool objects, not pre-baked schemas."""

    def test_tool_objects_reach_provider(self):
        from ai_sdk import generate_text
        from ai_sdk.providers.language_model import LanguageModel

        received_tools: list[Any] = []

        class CaptureModel(LanguageModel):
            def generate_text(
                self, *, prompt=None, system=None, messages=None, **kwargs
            ):
                received_tools.extend(kwargs.get("tools") or [])
                return {
                    "text": "done",
                    "finish_reason": "stop",
                    "usage": {},
                }

            async def stream_text(self, **kwargs):
                raise NotImplementedError

        generate_text(model=CaptureModel(), prompt="test", tools=[add_tool])
        assert len(received_tools) == 1
        assert isinstance(received_tools[0], Tool)
        assert received_tools[0].name == "add"


# ===========================================================================
# Review follow-ups (edge cases / error propagation)
# ===========================================================================


class TestOpenAIToolNormalisationErrors:
    def test_rejects_unknown_tool_type(self):
        with pytest.raises(TypeError, match="Unsupported tool type"):
            _normalise_openai_tools({"tools": ["not-a-tool"]})

    def test_rejects_unknown_dict_shape(self):
        with pytest.raises(TypeError, match="Unsupported tool dict shape"):
            _normalise_openai_tools({"tools": [{"foo": "bar"}]})


class TestAnthropicMessageEdgeCases:
    def test_tool_role_with_string_content(self):
        _, msgs = _extract_system_and_messages(
            prompt=None,
            system=None,
            messages=[
                {
                    "role": "tool",
                    "tool_call_id": "tc-1",
                    "content": "plain result",
                }
            ],
        )
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        blocks = msgs[0]["content"]
        assert blocks[0]["type"] == "tool_result"
        assert blocks[0]["tool_use_id"] == "tc-1"
        assert blocks[0]["content"] == "plain result"

    def test_empty_assistant_message_is_skipped(self):
        _, msgs = _extract_system_and_messages(
            prompt=None,
            system=None,
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": ""},
            ],
        )
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"


class TestGeminiImageAndConfig:
    def test_data_uri_image_becomes_bytes_part(self):
        import base64

        from ai_sdk.providers.gemini import _content_to_parts

        png_b64 = base64.b64encode(b"fakepng").decode()
        data_uri = f"data:image/png;base64,{png_b64}"
        parts = _content_to_parts(
            [{"type": "image", "image": data_uri, "mime_type": "image/png"}]
        )
        assert len(parts) == 1
        part = parts[0]
        # google-genai Part should carry inline bytes, not the raw data-URI string.
        inline = getattr(part, "inline_data", None)
        assert inline is not None
        assert getattr(inline, "data", None) == b"fakepng"

    def test_conflicting_max_token_kwargs_raise(self):
        from ai_sdk.providers.gemini import _build_config

        with pytest.raises(ValueError, match="Conflicting max token"):
            _build_config(
                system_instruction=None,
                default_kwargs={},
                kwargs={"max_tokens": 10, "max_output_tokens": 20},
            )


class TestStreamingErrorPropagation:
    @pytest.mark.asyncio
    @patch("ai_sdk.providers.anthropic._anthropic.Anthropic")
    async def test_anthropic_stream_propagates_producer_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        class BoomStream:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            @property
            def text_stream(self):
                raise RuntimeError("stream failed")

        mock_client.messages.stream.return_value = BoomStream()
        model = AnthropicModel("claude-test", api_key="k")

        with pytest.raises(RuntimeError, match="stream failed"):
            stream = model.stream_text(prompt="hi")
            async for _ in stream:
                pass

    @pytest.mark.asyncio
    @patch("ai_sdk.providers.gemini._genai.Client")
    async def test_gemini_stream_propagates_producer_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content_stream.side_effect = RuntimeError(
            "gemini stream failed"
        )

        model = GeminiModel("gemini-test", api_key="k")
        with pytest.raises(RuntimeError, match="gemini stream failed"):
            stream = model.stream_text(prompt="hi")
            async for _ in stream:
                pass


# ===========================================================================
# Multimodal: image input / file output normalisation
# ===========================================================================


class TestMultimodalHelpers:
    def test_openai_image_part_becomes_image_url(self):
        from ai_sdk.providers._multimodal import normalise_openai_message_content

        content = normalise_openai_message_content(
            [
                {"type": "text", "text": "describe"},
                {"type": "image", "image": b"\x89PNG", "mime_type": "image/png"},
            ]
        )
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "describe"}
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_openai_https_image_url_passthrough(self):
        from ai_sdk.providers._multimodal import openai_content_part_from_sdk

        part = openai_content_part_from_sdk(
            {"type": "image", "image": "https://example.com/a.png"}
        )
        assert part == {
            "type": "image_url",
            "image_url": {"url": "https://example.com/a.png"},
        }

    def test_anthropic_image_base64_block(self):
        from ai_sdk.providers._multimodal import normalise_anthropic_user_content

        content = normalise_anthropic_user_content(
            [
                {"type": "text", "text": "see"},
                {"type": "image", "image": b"img", "mime_type": "image/jpeg"},
            ]
        )
        assert content[0] == {"type": "text", "text": "see"}
        assert content[1]["type"] == "image"
        assert content[1]["source"]["type"] == "base64"
        assert content[1]["source"]["media_type"] == "image/jpeg"

    def test_anthropic_pdf_file_becomes_document(self):
        from ai_sdk.providers._multimodal import anthropic_block_from_sdk_part

        block = anthropic_block_from_sdk_part(
            {"type": "file", "data": b"%PDF-1.4", "mime_type": "application/pdf"}
        )
        assert block["type"] == "document"
        assert block["source"]["media_type"] == "application/pdf"

    def test_gemini_descriptors_for_image_bytes(self):
        from ai_sdk.providers import _multimodal as mm

        descs = mm.gemini_part_descriptors_from_sdk_content(
            [
                {"type": "text", "text": "cap"},
                {"type": "image", "image": b"xyz", "mime_type": "image/webp"},
            ]
        )
        assert descs[0] == {"text": "cap"}
        assert descs[1]["bytes"] == b"xyz"
        assert descs[1]["mime_type"] == "image/webp"

    def test_files_from_openai_message_data_uri(self):
        import base64

        from ai_sdk.providers._multimodal import files_from_openai_message

        b64 = base64.b64encode(b"pic").decode()
        msg = SimpleNamespace(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            ],
            images=None,
        )
        files = files_from_openai_message(msg)
        assert len(files) == 1
        assert files[0]["uint8_array"] == b"pic"
        assert files[0]["mime_type"] == "image/png"

    def test_files_from_anthropic_image_block(self):
        import base64

        from ai_sdk.providers._multimodal import files_from_anthropic_message

        b64 = base64.b64encode(b"out").decode()
        resp = SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="image",
                    source=SimpleNamespace(
                        type="base64", media_type="image/png", data=b64
                    ),
                )
            ]
        )
        files = files_from_anthropic_message(resp)
        assert files[0]["uint8_array"] == b"out"

    def test_files_from_gemini_inline_data(self):
        from ai_sdk.providers._multimodal import files_from_gemini_response

        raw = b"genimg"
        resp = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(
                                text=None,
                                function_call=None,
                                inline_data=SimpleNamespace(
                                    data=raw, mime_type="image/png"
                                ),
                            )
                        ]
                    )
                )
            ]
        )
        files = files_from_gemini_response(resp)
        assert files[0]["uint8_array"] == raw
        assert files[0]["mime_type"] == "image/png"


class TestOpenAIMultimodalIntegration:
    def test_build_chat_messages_normalises_user_images(self):
        from ai_sdk.providers.openai import _build_chat_messages

        msgs = _build_chat_messages(
            prompt=None,
            system=None,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {
                            "type": "image",
                            "image": b"\x00\x01",
                            "mime_type": "image/png",
                        },
                    ],
                }
            ],
        )
        assert msgs[0]["role"] == "user"
        content = msgs[0]["content"]
        assert isinstance(content, list)
        assert content[1]["type"] == "image_url"


class TestAnthropicMultimodalIntegration:
    def test_extract_user_image_messages(self):
        _, msgs = _extract_system_and_messages(
            prompt=None,
            system=None,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {"type": "image", "image": b"ab", "mime_type": "image/png"},
                    ],
                }
            ],
        )
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"][1]["type"] == "image"


class TestGeminiMultimodalIntegration:
    def test_content_to_parts_image_bytes(self):
        from ai_sdk.providers.gemini import _content_to_parts

        parts = _content_to_parts(
            [
                {"type": "text", "text": "hi"},
                {"type": "image", "image": b"data", "mime_type": "image/png"},
            ]
        )
        assert len(parts) == 2
