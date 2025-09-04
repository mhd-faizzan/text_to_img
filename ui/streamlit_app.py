import io
import time
from typing import Dict, Optional

import os
import sys

# Ensure project root is on sys.path for `src` imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from PIL import Image

from src.clients.stability import StabilityClient
from src.config import DEFAULT_ENGINE, ensure_api_key_present
from src.workflows.generate import GenerateParams, Workflow


st.set_page_config(page_title="SD3.5 Text-to-Image", page_icon="🎨", layout="wide")


def init_state() -> None:
	if "history" not in st.session_state:
		st.session_state.history = []  # list of dicts
	if "last_request_ts" not in st.session_state:
		st.session_state.last_request_ts = 0.0
	if "engine" not in st.session_state:
		st.session_state.engine = DEFAULT_ENGINE


def rate_limited_cooldown(seconds: float = 3.0) -> Optional[float]:
	now = time.time()
	delta = now - st.session_state.last_request_ts
	if delta < seconds:
		return seconds - delta
	return None


def sidebar_controls() -> Dict:
	st.sidebar.header("Controls")
	engine = st.sidebar.selectbox("Engine", ["api", "local"], index=["api", "local"].index(st.session_state.engine))
	st.session_state.engine = engine

	mode = st.sidebar.radio("Mode", ["t2i", "i2i"], horizontal=True)
	model = st.sidebar.selectbox("Model", ["flash", "turbo", "medium", "large"], index=0)
	aspect_ratio = st.sidebar.selectbox("Aspect Ratio", ["1:1", "16:9", "9:16", "4:3", "3:4"]) 
	seed = st.sidebar.number_input("Seed (optional)", min_value=0, step=1, value=0)
	seed = None if seed == 0 else int(seed)
	style_preset = st.sidebar.selectbox(
		"Style Preset (optional)",
		["", "photographic", "digital-art", "cinematic", "anime", "line-art", "lowpoly", "pixel-art"],
		index=0,
	)
	style_preset = style_preset or None
	cfg_scale = st.sidebar.slider("CFG Scale", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
	negative_prompt = st.sidebar.text_input("Negative Prompt")

	strength = None
	init_image_bytes = None
	if mode == "i2i":
		uploaded = st.sidebar.file_uploader("Init Image (<=10MB)", type=["png", "jpg", "jpeg", "webp"])
		if uploaded is not None:
			if uploaded.size > 10 * 1024 * 1024:
				st.sidebar.error("File too large. Max 10 MB.")
			else:
				init_image_bytes = uploaded.read()
		strength = st.sidebar.slider("Strength", min_value=0.0, max_value=1.0, value=0.6, step=0.05)

	prompt = st.text_area("Prompt", placeholder="A cozy cabin in the woods at golden hour, volumetric lighting")
	negative_prompt = negative_prompt or None

	return dict(
		engine=engine,
		mode=mode,
		model=model,
		aspect_ratio=aspect_ratio,
		seed=seed,
		style_preset=style_preset,
		cfg_scale=cfg_scale,
		negative_prompt=negative_prompt,
		strength=strength,
		init_image_bytes=init_image_bytes,
		prompt=prompt,
	)


@st.cache_resource(show_spinner=False)
def get_clients():
	ensure_api_key_present()
	api_client = StabilityClient()
	local = None
	try:
		# Lazy import heavy deps only if installed
		from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
		import torch

		device = "cuda" if torch.cuda.is_available() else "cpu"
		if device == "cuda":
			# SDXL Turbo for super fast previews
			pipe_txt = AutoPipelineForText2Image.from_pretrained(
				"stabilityai/sdxl-turbo", torch_dtype=torch.float16
			).to(device)
			pipe_i2i = AutoPipelineForImage2Image.from_pretrained(
				"stabilityai/sdxl-turbo", torch_dtype=torch.float16
			).to(device)
			local = {"device": device, "t2i": pipe_txt, "i2i": pipe_i2i}
	except Exception:
		local = None
	return api_client, local


def run_local_inference(local, params: Dict) -> Image.Image:
	import torch

	if params["mode"] == "t2i":
		pipe = local["t2i"]
		image = pipe(
			params["prompt"],
			negative_prompt=params.get("negative_prompt"),
			num_inference_steps=4,
			guidance_scale=float(params.get("cfg_scale", 1.0)),
			generator=torch.Generator(device=local["device"]).manual_seed(params.get("seed", 0)) if params.get("seed") else None,
		).images[0]
		return image
	else:
		pipe = local["i2i"]
		init_image = Image.open(io.BytesIO(params["init_image_bytes"]))
		image = pipe(
			prompt=params["prompt"],
			image=init_image,
			strength=float(params.get("strength", 0.6)),
			num_inference_steps=4,
			guidance_scale=float(params.get("cfg_scale", 1.0)),
		).images[0]
		return image


init_state()

st.title("Stable Diffusion 3.5 — Text to Image")
controls = sidebar_controls()

cooldown = rate_limited_cooldown(3.0)
if cooldown is not None:
	st.info(f"Cooling down… wait {cooldown:.1f}s to avoid rate limits.")

col1, col2 = st.columns([2, 1])
with col1:
	generate = st.button("Generate", type="primary", use_container_width=True, disabled=(not controls["prompt"]))
	placeholder = st.empty()

with col2:
	st.subheader("History")
	for i, item in enumerate(reversed(st.session_state.history[-10:]), start=1):
		st.caption(f"#{len(st.session_state.history) - i + 1} — {item['model']} {item['mode']}")
		st.image(item["image"], use_container_width=True)
		st.download_button("Download", data=item["bytes"], file_name=f"sd35_{i}.png", mime="image/png")
		st.divider()

if generate:
	if cooldown is not None:
		st.warning("Please wait a bit before the next request.")
		st.stop()

	with st.spinner("Generating…"):
		api_client, local = get_clients()
		try:
			if controls["engine"] == "local":
				if local is None:
					st.error("Local engine not available. Ensure GPU and diffusers installed.")
					st.stop()
				image = run_local_inference(local, controls)
				buf = io.BytesIO()
				image.save(buf, format="PNG")
				img_bytes = buf.getvalue()
				result_bytes = img_bytes
				content_type = "image/png"
				seed_used = controls.get("seed")
			else:
				workflow = Workflow(client=api_client)
				params = GenerateParams(**{k: v for k, v in controls.items() if k != "engine"})
				result = workflow.run(params)
				result_bytes = result.image_bytes
				content_type = result.content_type
				seed_used = result.seed

			st.session_state.last_request_ts = time.time()

			image = Image.open(io.BytesIO(result_bytes))
			placeholder.image(image, use_container_width=True)
			st.download_button("Download", data=result_bytes, file_name="sd35.png", mime=content_type)
			st.toast("Done!", icon="✅")
			st.session_state.history.append({
				"mode": controls["mode"],
				"model": controls["model"],
				"image": image,
				"bytes": result_bytes,
				"seed": seed_used,
			})
		except RuntimeError as e:
			st.error(f"Stability API error: {e}")
		except Exception as e:
			st.error(f"Unexpected error: {e}")
