from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from src.clients.stability import StabilityClient, GenerationResult


DEFAULT_CFG_BY_MODEL = {
	"turbo": 1.0,
	"flash": 1.0,
	"large": 4.0,
	"medium": 4.0,
}


class GenerateParams(BaseModel):
	mode: str = Field(pattern="^(t2i|i2i)$")
	prompt: str
	model: str = Field(pattern="^(large|turbo|medium|flash)$")
	aspect_ratio: str = Field(default="1:1")
	seed: Optional[int] = None
	style_preset: Optional[str] = None
	cfg_scale: Optional[float] = None
	negative_prompt: Optional[str] = None
	output_format: str = Field(default="png", pattern="^(png|jpeg|webp)$")
	# i2i only
	strength: Optional[float] = Field(default=None, ge=0.0, le=1.0)
	init_image_bytes: Optional[bytes] = None

	@field_validator("cfg_scale", mode="before")
	def _default_cfg(cls, v, values):  # type: ignore[override]
		if v is not None:
			return v
		model = values.get("model", "flash")
		return DEFAULT_CFG_BY_MODEL.get(model, 1.0)


@dataclass
class Workflow:
	client: StabilityClient

	def run(self, params: GenerateParams) -> GenerationResult:
		if params.mode == "t2i":
			return self.client.generate_text_to_image(
				prompt=params.prompt,
				model=params.model,
				aspect_ratio=params.aspect_ratio,
				seed=params.seed,
				style_preset=params.style_preset,
				cfg_scale=params.cfg_scale,
				negative_prompt=params.negative_prompt,
				output_format=params.output_format,
			)
		elif params.mode == "i2i":
			if not params.init_image_bytes:
				raise ValueError("init_image_bytes required for i2i mode")
			return self.client.generate_image_to_image(
				init_image_bytes=params.init_image_bytes,
				prompt=params.prompt,
				model=params.model,
				strength=params.strength or 0.6,
				aspect_ratio=params.aspect_ratio,
				seed=params.seed,
				style_preset=params.style_preset,
				cfg_scale=params.cfg_scale,
				negative_prompt=params.negative_prompt,
				output_format=params.output_format,
			)
		else:
			raise ValueError("Unknown mode")
