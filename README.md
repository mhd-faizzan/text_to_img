# SD3.5 Text-to-Image App (Streamlit)

Production-ready Streamlit app for Stable Diffusion 3.5 (text-to-image and image-to-image) using Stability AI API, with optional local fallback via Hugging Face Diffusers (SDXL-Turbo).

## Features
- Text-to-image and image-to-image modes
- SD3.5 models: large, turbo, medium, flash (default: flash)
- Sidebar controls: model, aspect ratio, seed, style preset, CFG scale, negative prompt
- Image upload (≤10 MB) for i2i and strength slider
- Rate limit: 3s cooldown; friendly errors and prompt guardrails
- Session history with display and download
- Optional local fallback (GPU only) using SDXL-Turbo

## Repo layout
```
src/
  config.py
  clients/stability.py
  workflows/generate.py
ui/
  streamlit_app.py
tests/
  test_payload.py
requirements.txt
.env.example
```

## Setup
1. Python 3.10+
2. Create venv and install deps:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
3. Copy `.env.example` to `.env` and set `STABILITY_API_KEY`.
4. Run locally:
```bash
streamlit run ui/streamlit_app.py
```

On Streamlit Community Cloud, set `STABILITY_API_KEY` in app secrets. The app uses `st.secrets` when available.

## Optional local fallback
Install heavy deps only if you plan to use the local engine (requires GPU):
```bash
pip install diffusers transformers accelerate safetensors
```
Toggle Engine in the sidebar to Local. Uses `stabilityai/sdxl-turbo` with `num_inference_steps=4`.

## Testing
```bash
pip install pytest
pytest -q
```

## Deployment (Streamlit Cloud)
- Push repo to GitHub
- Deploy with entry point `ui/streamlit_app.py`
- Configure `STABILITY_API_KEY` in Secrets
- CPU-only is fine since Stability API does the heavy lifting

## Notes
- API endpoints used:
  - `https://api.stability.ai/v2beta/stable-image/generate/sd3.5`
  - `https://api.stability.ai/v2beta/stable-image/edit/sd3.5`
- Responses return image bytes; headers may include seed
