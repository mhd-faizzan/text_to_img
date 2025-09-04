from __future__ import annotations
import io
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import httpx
from src.config import CLIENT_ID, SESSION_ID, STABILITY_API_KEY

# Supported models mapping
SUPPORTED_MODELS = {
    "large": "sd3.5-large",
    "turbo": "sd3.5-large-turbo",
    "medium": "sd3.5-medium",
    "flash": "sd3.5-flash",
}

@dataclass
class GenerationResult:
    image_bytes: bytes
    content_type: str
    seed: Optional[int]
    response_headers: Dict[str, str]

class StabilityClient:
    def __init__(self, api_key: Optional[str] = None, timeout_seconds: float = 60.0) -> None:
        self.api_key = api_key or STABILITY_API_KEY
        if not self.api_key:
            raise RuntimeError("Missing STABILITY_API_KEY for Stability API client")
        self.client = httpx.Client(timeout=timeout_seconds)

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "image/*",
        }
        if CLIENT_ID:
            headers["X-Client-Id"] = CLIENT_ID
        if SESSION_ID:
            headers["X-Session-Id"] = SESSION_ID
        return headers

    def generate_text_to_image(
        self,
        prompt: str,
        model: str = "sd3",
        aspect_ratio: str = "1:1",
        seed: Optional[int] = None,
        style_preset: Optional[str] = None,
        cfg_scale: Optional[float] = None,
        negative_prompt: Optional[str] = None,
        output_format: str = "png",
    ) -> GenerationResult:
        engine = SUPPORTED_MODELS.get(model, model)
        url = f"https://api.stability.ai/v2beta/stable-image/generate/{engine}"

        data = {
            "prompt": prompt,
            "output_format": output_format,
            "aspect_ratio": aspect_ratio,
        }
        if seed is not None:
            data["seed"] = seed
        if style_preset:
            data["style_preset"] = style_preset
        if cfg_scale is not None:
            data["cfg_scale"] = cfg_scale
        if negative_prompt:
            data["negative_prompt"] = negative_prompt

        resp = self.client.post(url, headers=self._headers(), files={"none": ""}, data=data)
        return self._process_image_response(resp)

    def generate_image_to_image(
        self,
        init_image_bytes: bytes,
        prompt: str,
        model: str = "sd3",
        strength: float = 0.6,
        aspect_ratio: str = "1:1",
        seed: Optional[int] = None,
        style_preset: Optional[str] = None,
        cfg_scale: Optional[float] = None,
        negative_prompt: Optional[str] = None,
        output_format: str = "jpeg",
    ) -> GenerationResult:
        engine = SUPPORTED_MODELS.get(model, model)
        url = f"https://api.stability.ai/v2beta/stable-image/generate/{engine}"

        data = {
            "prompt": prompt,
            "output_format": output_format,
            "strength": str(strength),
            "aspect_ratio": aspect_ratio,
        }
        if seed is not None:
            data["seed"] = str(seed)
        if style_preset:
            data["style_preset"] = style_preset
        if cfg_scale is not None:
            data["cfg_scale"] = str(cfg_scale)
        if negative_prompt:
            data["negative_prompt"] = negative_prompt

        files = {
            "image": ("image.png", io.BytesIO(init_image_bytes), "image/png"),
            "none": "",
        }

        resp = self.client.post(url, headers=self._headers(), data=data, files=files)
        return self._process_image_response(resp)

    def _process_image_response(self, resp: httpx.Response) -> GenerationResult:
        if resp.status_code >= 400:
            raise RuntimeError(f"Stability API error: {self._compose_error_message(resp)}")
        content_type = resp.headers.get("Content-Type", "image/png")
        seed_header = resp.headers.get("X-Seed") or resp.headers.get("Seed")
        seed_value = int(seed_header) if seed_header and seed_header.isdigit() else None
        return GenerationResult(
            image_bytes=resp.content,
            content_type=content_type,
            seed=seed_value,
            response_headers=dict(resp.headers),
        )

    def _compose_error_message(self, resp: httpx.Response) -> str:
        try:
            data = resp.json()
            msg = data.get("message") or data.get("error") or data
            return f"{resp.status_code} {msg}"
        except Exception:
            text = resp.text
            if len(text) > 500:
                text = text[:500] + "…"
            return f"{resp.status_code} {text}"
