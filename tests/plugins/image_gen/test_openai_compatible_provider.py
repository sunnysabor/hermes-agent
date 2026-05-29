"""Tests for generic OpenAI-compatible image generation provider."""

from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import patch

import yaml

import plugins.image_gen.openai_compatible as compat


class TestOpenAICompatibleImageProvider:
    def test_default_model_and_schema(self):
        provider = compat.OpenAICompatibleImageGenProvider()

        assert provider.name == "openai-compatible"
        assert provider.default_model() == compat.DEFAULT_MODEL
        models = provider.list_models()
        assert models[0]["id"] == compat.DEFAULT_MODEL
        schema = provider.get_setup_schema()
        assert schema["env_vars"]
        assert any(item["key"] == "OPENAI_COMPATIBLE_IMAGE_API_KEY" for item in schema["env_vars"])

    def test_resolve_config_prefers_provider_specific_model_base_and_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("OPENAI_COMPATIBLE_IMAGE_MODEL", raising=False)
        monkeypatch.delenv("OPENAI_COMPATIBLE_IMAGE_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_COMPATIBLE_IMAGE_API_KEY", raising=False)
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({
                "image_gen": {
                    "model": "top-level-model",
                    "openai-compatible": {
                        "model": "provider-model",
                        "base_url": "https://images.example.com/v1/",
                        "api_key": "config-key",
                    },
                }
            }),
            encoding="utf-8",
        )

        resolved = compat._resolve_config()

        assert resolved.model == "provider-model"
        assert resolved.base_url == "https://images.example.com/v1"
        assert resolved.api_key == "config-key"

    def test_env_overrides_config_for_scripts(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_MODEL", "env-model")
        monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_BASE_URL", "https://env.example.com/v1")
        monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_API_KEY", "env-key")
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({
                "image_gen": {
                    "openai-compatible": {
                        "model": "provider-model",
                        "base_url": "https://images.example.com/v1",
                        "api_key": "config-key",
                    },
                }
            }),
            encoding="utf-8",
        )

        resolved = compat._resolve_config()

        assert resolved.model == "env-model"
        assert resolved.base_url == "https://env.example.com/v1"
        assert resolved.api_key == "env-key"

    def test_generate_posts_images_generations_and_saves_b64(self, monkeypatch):
        provider = compat.OpenAICompatibleImageGenProvider()
        png_b64 = base64.b64encode(b"fakepng").decode("ascii")
        response = SimpleNamespace(
            status_code=200,
            json=lambda: {"data": [{"b64_json": png_b64, "revised_prompt": "better cat"}]},
            raise_for_status=lambda: None,
            text="",
        )
        monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_API_KEY", "key")
        monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_BASE_URL", "https://api.example.com/v1")
        monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_MODEL", "image-model")

        with patch("plugins.image_gen.openai_compatible.requests.post", return_value=response) as post:
            with patch("plugins.image_gen.openai_compatible.save_b64_image", return_value="/tmp/out.png") as save:
                result = provider.generate("draw a cat", "portrait")

        assert result["success"] is True
        assert result["image"] == "/tmp/out.png"
        assert result["model"] == "image-model"
        assert result["provider"] == "openai-compatible"
        assert result["size"] == "1024x1536"
        assert result["revised_prompt"] == "better cat"
        save.assert_called_once_with(png_b64, prefix="openai_compatible_image-model")
        post.assert_called_once()
        url = post.call_args.args[0]
        assert url == "https://api.example.com/v1/images/generations"
        payload = post.call_args.kwargs["json"]
        assert payload["model"] == "image-model"
        assert payload["prompt"] == "draw a cat"
        assert payload["size"] == "1024x1536"
        assert payload["n"] == 1
        assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer key"

    def test_generate_accepts_url_response(self, monkeypatch):
        provider = compat.OpenAICompatibleImageGenProvider()
        response = SimpleNamespace(
            status_code=200,
            json=lambda: {"data": [{"url": "https://cdn.example.com/out.png"}]},
            raise_for_status=lambda: None,
            text="",
        )
        monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_API_KEY", "key")
        monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_BASE_URL", "https://api.example.com/v1")

        with patch("plugins.image_gen.openai_compatible.requests.post", return_value=response):
            result = provider.generate("draw a cat", "square")

        assert result["success"] is True
        assert result["image"] == "https://cdn.example.com/out.png"
        assert result["size"] == "1024x1024"

    def test_missing_base_url_returns_actionable_error(self, monkeypatch):
        monkeypatch.delenv("OPENAI_COMPATIBLE_IMAGE_BASE_URL", raising=False)
        monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_API_KEY", "key")
        provider = compat.OpenAICompatibleImageGenProvider()

        result = provider.generate("draw a cat")

        assert result["success"] is False
        assert result["error_type"] == "config_required"
        assert "OPENAI_COMPATIBLE_IMAGE_BASE_URL" in result["error"]

    def test_register_calls_context(self):
        class Ctx:
            def __init__(self):
                self.provider = None

            def register_image_gen_provider(self, provider):
                self.provider = provider

        ctx = Ctx()
        compat.register(ctx)

        assert isinstance(ctx.provider, compat.OpenAICompatibleImageGenProvider)
