from __future__ import annotations

import os
from typing import Optional

import groq
import httpx
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"

_CORRECTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You clean up raw OCR output. Fix obvious character-recognition "
            "errors (misread letters, broken words, stray symbols) and "
            "rejoin words split across line breaks. Do NOT summarize, "
            "translate, add information, or change the meaning. If a "
            "passage is ambiguous, leave it as-is rather than guessing. "
            "Return ONLY the corrected text, no commentary.",
        ),
        ("human", "{raw_text}"),
    ]
)


class OCRCorrector:
    """Optional LLM correction pass for raw OCR text.

    This sits between the OCR step and chunking: ImageLoader always produces
    usable text from Tesseract alone, so a missing/rate-limited GROQ_API_KEY
    degrades gracefully to "use the raw OCR text" rather than failing the
    whole upload.
    """

    def __init__(self, model: str | None = None, temperature: float = 0.0):
        self._model_name = model or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
        self._temperature = temperature
        self._chain: Runnable | None = None

    @property
    def _api_key(self) -> Optional[str]:
        raw = os.getenv("GROQ_API_KEY")
        return raw.strip() if raw else None

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    def _get_chain(self) -> Runnable:
        if self._chain is None:
            llm = ChatGroq(
                model=self._model_name,
                temperature=self._temperature,
                api_key=self._api_key,
            )
            self._chain = _CORRECTION_PROMPT | llm | StrOutputParser()
        return self._chain

    def correct(self, raw_text: str) -> str:
        """Best-effort correction. Falls back to the raw OCR text (instead
        of raising) if generation isn't configured or the Groq call fails —
        correction is an enhancement, not a hard dependency of ingestion."""
        if not raw_text.strip() or not self.is_configured:
            return raw_text

        try:
            chain = self._get_chain()
            corrected = chain.invoke({"raw_text": raw_text})
            return corrected.strip() or raw_text
        except (groq.APIStatusError, groq.APIConnectionError, httpx.HTTPError):
            return raw_text