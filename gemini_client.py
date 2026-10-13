from __future__ import annotations

from typing import Any

from config import AppConfig


class GeminiClient:
    def __init__(self):
        self._model = None
        if AppConfig.GOOGLE_API_KEY:
            import google.generativeai as genai

            genai.configure(api_key=AppConfig.GOOGLE_API_KEY)
            self._model = genai.GenerativeModel(AppConfig.GEMINI_MODEL)

    def check_connection(self) -> bool:
        if self._model is None:
            return False
        try:
            response = self._model.generate_content(
                "Reply with the single word: connected",
                generation_config={"max_output_tokens": 8, "temperature": 0},
                request_options={"timeout": 30},
            )
            return bool(getattr(response, "text", "").strip())
        except Exception:
            return False

    def generate_text(self, prompt: str) -> str:
        if self._model is None:
            return ""
        response = self._model.generate_content(
            prompt,
            request_options={"timeout": 60},
        )
        return getattr(response, "text", "").strip()
