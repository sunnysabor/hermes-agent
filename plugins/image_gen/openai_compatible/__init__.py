"""Generic OpenAI-compatible image generation backend.

This backend lets Hermes route ``image_generate`` to any service that exposes an
OpenAI-compatible ``POST /images/generations`` endpoint. It is intentionally
small and configurable so users can point Hermes at model-hosted image APIs
without writing a new provider plugin for every vendor.

Configuration precedence (first hit wins):

1. Environment variables, for scripts/tests:
   ``OPENAI_COMPATIBLE_IMAGE_MODEL``, ``OPENAI_COMPATIBLE_IMAGE_BASE_URL``,
   ``OPENAI_COMPATIBLE_IMAGE_API_KEY``.
2. ``image_gen.openai-compatible`` in ``config.yaml``.
3. Top-level ``image_gen.model`` for the model only.
4. Defaults: model ``gpt-image-1``; no default base URL/key.

Example config::

    image_gen:
      provider: openai-compatible
      model: my-image-model
      openai-compatible:
        base_url: https://api.example.com/v1
        # Prefer storing secrets in ~/.hermes/.env instead:
        # api_key: ${OPENAI_COMPATIBLE_IMAGE_API_KEY}

The endpoint must return either ``data[0].b64_json`` or ``data[0].url``.
Base64 output is saved under ``$HERMES_HOME/cache/images/``; URL output is
returned directly for the platform delivery layer to handle.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_b64_image,
    success_response,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-image-1"
DEFAULT_TIMEOUT_SECONDS = 300

_SIZES = {
    "landscape": "1536x1024",
    "square": "1024x1024",
    "portrait": "1024x1536",
}


@dataclass(frozen=True)
class ResolvedConfig:
    model: str
    base_url: str
    api_key: str
    timeout: float


def _load_image_gen_config() -> Dict[str, Any]:
    """Read ``image_gen`` from config.yaml (returns {} on any failure)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen config: %s", exc)
        return {}


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _resolve_config() -> ResolvedConfig:
    """Resolve model/base_url/api_key from env and config."""
    cfg = _load_image_gen_config()
    provider_cfg = (
        cfg.get("openai-compatible")
        if isinstance(cfg.get("openai-compatible"), dict)
        else {}
    )

    env_model = _string(os.environ.get("OPENAI_COMPATIBLE_IMAGE_MODEL"))
    provider_model = _string(provider_cfg.get("model")) if isinstance(provider_cfg, dict) else ""
    top_model = _string(cfg.get("model"))
    model = env_model or provider_model or top_model or DEFAULT_MODEL

    env_base_url = _string(os.environ.get("OPENAI_COMPATIBLE_IMAGE_BASE_URL"))
    provider_base_url = _string(provider_cfg.get("base_url")) if isinstance(provider_cfg, dict) else ""
    base_url = (env_base_url or provider_base_url).rstrip("/")

    env_api_key = _string(os.environ.get("OPENAI_COMPATIBLE_IMAGE_API_KEY"))
    provider_api_key = _string(provider_cfg.get("api_key")) if isinstance(provider_cfg, dict) else ""
    api_key = env_api_key or provider_api_key

    env_timeout = _string(os.environ.get("OPENAI_COMPATIBLE_IMAGE_TIMEOUT"))
    provider_timeout = provider_cfg.get("timeout") if isinstance(provider_cfg, dict) else None
    timeout = _positive_float(env_timeout or provider_timeout, DEFAULT_TIMEOUT_SECONDS)

    return ResolvedConfig(
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
    )


class OpenAICompatibleImageGenProvider(ImageGenProvider):
    """Provider for OpenAI-compatible ``/images/generations`` APIs."""

    @property
    def name(self) -> str:
        return "openai-compatible"

    @property
    def display_name(self) -> str:
        return "OpenAI Compatible"

    def is_available(self) -> bool:
        resolved = _resolve_config()
        return bool(resolved.base_url and resolved.api_key)

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": DEFAULT_MODEL,
                "display": "Configured image model",
                "speed": "varies",
                "strengths": "Any OpenAI-compatible /images/generations endpoint",
                "price": "varies",
            }
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "OpenAI Compatible",
            "badge": "custom",
            "tag": "Route image_generate to any OpenAI-compatible /images/generations API",
            "env_vars": [
                {
                    "key": "OPENAI_COMPATIBLE_IMAGE_API_KEY",
                    "prompt": "OpenAI-compatible image API key",
                },
                {
                    "key": "OPENAI_COMPATIBLE_IMAGE_BASE_URL",
                    "prompt": "OpenAI-compatible base URL (for example https://api.example.com/v1)",
                },
                {
                    "key": "OPENAI_COMPATIBLE_IMAGE_MODEL",
                    "prompt": "Image model id (optional; defaults to image_gen.model or gpt-image-1)",
                    "optional": True,
                },
            ],
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        resolved = _resolve_config()

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider=self.name,
                model=resolved.model,
                aspect_ratio=aspect,
            )
        if not resolved.base_url:
            return error_response(
                error=(
                    "OPENAI_COMPATIBLE_IMAGE_BASE_URL is not set. Configure "
                    "image_gen.openai-compatible.base_url in config.yaml or set "
                    "OPENAI_COMPATIBLE_IMAGE_BASE_URL in ~/.hermes/.env."
                ),
                error_type="config_required",
                provider=self.name,
                model=resolved.model,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        if not resolved.api_key:
            return error_response(
                error=(
                    "OPENAI_COMPATIBLE_IMAGE_API_KEY is not set. Configure it "
                    "with `hermes tools` → Image Generation → OpenAI Compatible "
                    "or add it to ~/.hermes/.env."
                ),
                error_type="auth_required",
                provider=self.name,
                model=resolved.model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        size = _SIZES.get(aspect, _SIZES["square"])
        url = f"{resolved.base_url}/images/generations"
        payload: Dict[str, Any] = {
            "model": resolved.model,
            "prompt": prompt,
            "size": size,
            "n": 1,
        }
        # Some OpenAI-compatible endpoints support these extra knobs; only
        # forward them when the caller/config explicitly supplied them.
        for key in ("quality", "style", "response_format"):
            value = kwargs.get(key)
            if value is not None:
                payload[key] = value

        try:
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {resolved.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=resolved.timeout,
            )
            response.raise_for_status()
            body = response.json()
        except requests.HTTPError as exc:
            exc_response = getattr(exc, "response", None)
            status = getattr(exc_response, "status_code", "unknown")
            text = getattr(exc_response, "text", "")
            return error_response(
                error=f"OpenAI-compatible image API returned HTTP {status}: {text[:500]}",
                error_type="api_error",
                provider=self.name,
                model=resolved.model,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except Exception as exc:
            logger.debug("OpenAI-compatible image generation failed", exc_info=True)
            return error_response(
                error=f"OpenAI-compatible image generation failed: {exc}",
                error_type="api_error",
                provider=self.name,
                model=resolved.model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list) or not data:
            return error_response(
                error="OpenAI-compatible response contained no data[0] image item",
                error_type="empty_response",
                provider=self.name,
                model=resolved.model,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        first = data[0]
        if not isinstance(first, dict):
            return error_response(
                error="OpenAI-compatible response data[0] was not an object",
                error_type="empty_response",
                provider=self.name,
                model=resolved.model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        b64 = _string(first.get("b64_json"))
        image_url = _string(first.get("url"))
        revised_prompt = _string(first.get("revised_prompt"))
        if b64:
            try:
                image_ref = str(save_b64_image(b64, prefix=f"openai_compatible_{resolved.model}"))
            except Exception as exc:
                return error_response(
                    error=f"Could not save image to cache: {exc}",
                    error_type="io_error",
                    provider=self.name,
                    model=resolved.model,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
        elif image_url:
            image_ref = image_url
        else:
            return error_response(
                error="OpenAI-compatible response contained neither b64_json nor url",
                error_type="empty_response",
                provider=self.name,
                model=resolved.model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        extra: Dict[str, Any] = {"size": size}
        if revised_prompt:
            extra["revised_prompt"] = revised_prompt
        return success_response(
            image=image_ref,
            model=resolved.model,
            prompt=prompt,
            aspect_ratio=aspect,
            provider=self.name,
            extra=extra,
        )


def register(ctx) -> None:
    """Plugin entry point — wire provider into the image_gen registry."""
    ctx.register_image_gen_provider(OpenAICompatibleImageGenProvider())
