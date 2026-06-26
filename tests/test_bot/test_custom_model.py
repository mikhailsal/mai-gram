"""Tests for arbitrary ("custom") model parsing and gating helpers."""

from __future__ import annotations

from mai_gram.bot import custom_model
from mai_gram.config_loaders import BotConfig


def _token() -> str:
    return "test-" + "bot-token"


class TestIsUserAllowed:
    def test_none_bot_config_disallows(self) -> None:
        assert custom_model.is_user_allowed(None, "123") is False

    def test_empty_list_disallows(self) -> None:
        cfg = BotConfig(token=_token(), custom_model_allowed_users=[])
        assert custom_model.is_user_allowed(cfg, "123") is False

    def test_listed_user_is_allowed_regardless_of_str_or_int(self) -> None:
        cfg = BotConfig(token=_token(), custom_model_allowed_users=[123, 456])
        assert custom_model.is_user_allowed(cfg, "123") is True
        assert custom_model.is_user_allowed(cfg, 456) is True  # type: ignore[arg-type]

    def test_unlisted_user_is_denied(self) -> None:
        cfg = BotConfig(token=_token(), custom_model_allowed_users=[123])
        assert custom_model.is_user_allowed(cfg, "999") is False


class TestValidateModelName:
    def test_accepts_typical_model_ids(self) -> None:
        assert custom_model.validate_model_name("openai/gpt-5.4-mini")
        assert custom_model.validate_model_name("google/gemma-4-31b-it:free")
        assert custom_model.validate_model_name("vendor/model@alias")

    def test_rejects_empty_or_whitespace_or_too_short(self) -> None:
        assert not custom_model.validate_model_name("")
        assert not custom_model.validate_model_name("   ")
        assert not custom_model.validate_model_name("openai/ gpt")  # space
        assert not custom_model.validate_model_name("x")  # too short


class TestParseModelParams:
    def test_coerces_scalar_types(self) -> None:
        text = "temperature = 0.7\nmax_tokens = 8000\nprovider.allow_fallbacks = false"
        params = custom_model.parse_model_params(text)
        assert params == {
            "temperature": 0.7,
            "max_tokens": 8000,
            "provider": {"allow_fallbacks": False},
        }

    def test_dotted_keys_nest(self) -> None:
        params = custom_model.parse_model_params('reasoning.effort = "high"')
        assert params == {"reasoning": {"effort": "high"}}

    def test_unquoted_string_value_is_kept(self) -> None:
        params = custom_model.parse_model_params("reasoning.effort = high")
        assert params == {"reasoning": {"effort": "high"}}

    def test_json_list_value(self) -> None:
        params = custom_model.parse_model_params(
            'provider.order = ["Google AI Studio", "ai-studio"]'
        )
        assert params == {"provider": {"order": ["Google AI Studio", "ai-studio"]}}

    def test_null_and_bool_and_comments_and_blank_lines(self) -> None:
        text = "# a comment\n\nflag = true\nnothing = null\nbad line without equals"
        params = custom_model.parse_model_params(text)
        assert params == {"flag": True, "nothing": None}

    def test_merges_multiple_dotted_keys_under_same_root(self) -> None:
        text = 'provider.order = ["x"]\nprovider.allow_fallbacks = false'
        params = custom_model.parse_model_params(text)
        assert params == {"provider": {"order": ["x"], "allow_fallbacks": False}}


class TestParseCustomModelInput:
    def test_first_line_is_model_rest_are_params(self) -> None:
        text = "openai/gpt-5.4-mini\nreasoning.effort = high\ntemperature = 0.5"
        model, params = custom_model.parse_custom_model_input(text)
        assert model == "openai/gpt-5.4-mini"
        assert params == {"reasoning": {"effort": "high"}, "temperature": 0.5}

    def test_skips_leading_blank_and_comment_lines(self) -> None:
        text = "\n# pick a model\nopenai/gpt-4o-mini\ntemperature = 1.0"
        model, params = custom_model.parse_custom_model_input(text)
        assert model == "openai/gpt-4o-mini"
        assert params == {"temperature": 1.0}

    def test_model_only_returns_empty_params(self) -> None:
        model, params = custom_model.parse_custom_model_input("openai/gpt-4o-mini")
        assert model == "openai/gpt-4o-mini"
        assert params == {}


class TestMergeAndLoad:
    def test_merge_custom_overrides_base_top_level(self) -> None:
        base = {"temperature": 0.7, "reasoning": {"effort": "low"}}
        custom = {"temperature": 1.5, "top_p": 0.9}
        merged = custom_model.merge_extra_params(base, custom)
        assert merged == {"temperature": 1.5, "reasoning": {"effort": "low"}, "top_p": 0.9}
        # base is not mutated
        assert base["temperature"] == 0.7

    def test_merge_with_none_returns_base(self) -> None:
        base = {"a": 1}
        assert custom_model.merge_extra_params(base, None) is base

    def test_load_custom_params_roundtrip_and_bad_input(self) -> None:
        assert custom_model.load_custom_params(None) is None
        assert custom_model.load_custom_params("") is None
        assert custom_model.load_custom_params("not json") is None
        assert custom_model.load_custom_params("[]") is None  # not a dict
        assert custom_model.load_custom_params('{"temperature": 0.7}') == {"temperature": 0.7}

    def test_format_params_summary(self) -> None:
        assert custom_model.format_params_summary(None) == "(none)"
        assert custom_model.format_params_summary({}) == "(none)"
        assert custom_model.format_params_summary({"b": 1, "a": 2}) == '{"a": 2, "b": 1}'
