import os
from typing import Optional

from dotenv import load_dotenv


# Load from .env for local dev; on Streamlit Cloud, use st.secrets
load_dotenv(override=False)


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
	"""Fetch secret from Streamlit secrets if available, else env."""
	try:
		import streamlit as st  # local import to avoid hard dep in tests

		if hasattr(st, "secrets") and key in st.secrets:  # type: ignore[attr-defined]
			value = st.secrets.get(key)  # type: ignore[attr-defined]
			if value is not None:
				return str(value)
	except Exception:
		# Fall back to environment variables
		pass
	return os.getenv(key, default)


STABILITY_API_KEY: Optional[str] = get_secret("STABILITY_API_KEY")
CLIENT_ID: str = get_secret("CLIENT_ID", "sd35-app") or "sd35-app"
SESSION_ID: Optional[str] = get_secret("SESSION_ID")
DEFAULT_ENGINE: str = get_secret("DEFAULT_ENGINE", "api") or "api"  # api | local


def ensure_api_key_present() -> None:
	if not STABILITY_API_KEY:
		raise RuntimeError(
			"STABILITY_API_KEY is not set. Add it to Streamlit secrets or .env"
		)
