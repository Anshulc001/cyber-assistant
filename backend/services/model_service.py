"""Model service — loads Qwythos-9B and exposes generate / stream methods.

On Colab the model is loaded once at startup and cached in GPU RAM.
Locally (no GPU) the service stays in an unloaded state so the rest of the
app can still start and the health endpoint can report model_loaded=False.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator

from backend.config import settings
from backend.models.schemas import Message

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a helpful, concise, and accurate personal AI assistant. "
    "Answer the user's question directly. If you don't know, say so."
)


class ModelService:
    """Singleton wrapper around the HuggingFace pipeline."""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self.is_loaded: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Download (if needed) and load the model into GPU/CPU memory.

        Called once from the Colab notebook after mounting Drive.
        Safe to call multiple times — subsequent calls are no-ops.
        """
        if self.is_loaded:
            logger.info("Model already loaded — skipping.")
            return

        logger.info("Loading model '%s' …", settings.MODEL_NAME)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            cache_dir = str(settings.models_dir)
            self._tokenizer = AutoTokenizer.from_pretrained(
                settings.MODEL_NAME,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                settings.MODEL_NAME,
                cache_dir=cache_dir,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            self._model.eval()
            self.is_loaded = True
            logger.info("Model loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        message: str,
        history: list[Message],
        context_chunks: list,
    ) -> str:
        parts: list[str] = [f"<|system|>\n{_SYSTEM_PROMPT}"]

        if context_chunks:
            context_text = "\n\n".join(
                f"[Source: {c.source}]\n{c.text}" for c in context_chunks
            )
            parts.append(
                f"<|context|>\nUse the following retrieved context to answer:\n{context_text}"
            )

        for turn in history:
            tag = "<|user|>" if turn.role == "user" else "<|assistant|>"
            parts.append(f"{tag}\n{turn.content}")

        parts.append(f"<|user|>\n{message}")
        parts.append("<|assistant|>")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    async def generate(
        self,
        message: str,
        history: list[Message],
        context_chunks: list,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        """Non-streaming full response. Runs the blocking call in a thread pool."""
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")

        prompt = self._build_prompt(message, history, context_chunks)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._run_inference(prompt, max_new_tokens, temperature, top_p),
        )
        return result

    def _run_inference(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        # Decode only newly generated tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    async def stream(
        self,
        message: str,
        history: list[Message],
        context_chunks: list,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> AsyncGenerator[str, None]:
        """Yield individual tokens using HuggingFace TextIteratorStreamer."""
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")

        import threading

        import torch
        from transformers import TextIteratorStreamer

        prompt = self._build_prompt(message, history, context_chunks)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": self._tokenizer.eos_token_id,
            "streamer": streamer,
        }

        thread = threading.Thread(
            target=lambda: self._model.generate(**gen_kwargs),
            daemon=True,
        )
        thread.start()

        loop = asyncio.get_event_loop()
        for token_text in streamer:
            yield token_text
            # Yield control so the event loop can process other coroutines
            await asyncio.sleep(0)

        thread.join()


model_service = ModelService()
