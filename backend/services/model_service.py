"""Model service — acts as a proxy client to the remote Colab GPU inference server."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncGenerator

import httpx

from backend.config import settings
from backend.models.schemas import Message
from backend.services.db_service import db_service

logger = logging.getLogger(__name__)


class ModelService:
    """Singleton wrapper that proxies LLM inference to Colab."""

    @property
    def is_loaded(self) -> bool:
        """Query remote Colab /health to check if model is loaded and online."""
        colab_url = db_service.get_setting("colab_url")
        if not colab_url:
            logger.warning("Colab Tunnel URL is not configured.")
            return False

        try:
            # Use synchronous HTTP GET with a short timeout to check health
            r = httpx.get(f"{colab_url}/health", timeout=1.5)
            if r.status_code == 200:
                data = r.json()
                return bool(data.get("model_loaded"))
        except Exception as exc:
            logger.debug("Failed to connect to Colab at %s: %s", colab_url, exc)
        return False

    def load(self) -> None:
        """No-op on local PC backend (only loads model on Colab)."""
        logger.info("Local backend load() is a no-op — model runs on Colab.")

    # ------------------------------------------------------------------
    # Message building
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        message: str,
        history: list[Message],
        context_chunks: list,
    ) -> list[dict[str, str]]:
        sys_content = (
            "You are Qwythos, a helpful, concise, and accurate personal AI assistant created by Empero AI.\n\n"
            "FORMATTING REQUIREMENT:\n"
            "You MUST begin your response by writing your step-by-step internal monologue and planning process wrapped inside <think> and </think> tags.\n"
            "Immediately after the </think> tag, you MUST write the separator line '=== RESPONSE ===' on its own line, and then write your final response to the user.\n\n"
            "Example format:\n"
            "<think>\n"
            "1. Analyze user request...\n"
            "2. Formulate plan...\n"
            "</think>\n"
            "=== RESPONSE ===\n"
            "Hello! I'm Qwythos, an AI assistant created by Empero AI..."
        )

        if context_chunks:
            context_text = "\n\n".join(
                f"[Source: {c.source}]\n{c.text}" for c in context_chunks
            )
            sys_content += f"\n\nRelevant context retrieved from your documents:\n\n{context_text}"

        msgs = [{"role": "system", "content": sys_content}]
        for turn in history:
            msgs.append({"role": turn.role.value, "content": turn.content})
        msgs.append({"role": "user", "content": message})
        return msgs

    # ------------------------------------------------------------------
    # Inference Proxy
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
        """Proxy non-streaming completion to remote Colab server."""
        colab_url = db_service.get_setting("colab_url")
        if not colab_url:
            raise RuntimeError("Colab Tunnel URL is not configured in Settings.")

        messages = self._build_messages(message, history, context_chunks)
        payload = {
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(
                f"{colab_url}/chat",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            if r.status_code != 200:
                raise RuntimeError(f"Colab error {r.status_code}: {r.text}")

            # Non-streaming endpoint returns tokens from SSE stream or full JSON depending on config.
            # We parse standard SSE format even for non-streaming to be safe, or just collect tokens.
            full_reply = ""
            # Parse response chunks line by line
            for line in r.text.split("\n"):
                line = line.strip()
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data_json = json.loads(data_str)
                        full_reply += data_json.get("token", "")
                    except Exception:
                        pass
            return full_reply.strip()

    async def stream(
        self,
        message: str,
        history: list[Message],
        context_chunks: list,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> AsyncGenerator[str, None]:
        """Proxy streaming tokens from Colab to local client."""
        colab_url = db_service.get_setting("colab_url")
        if not colab_url:
            raise RuntimeError("Colab Tunnel URL is not configured in Settings.")

        messages = self._build_messages(message, history, context_chunks)
        payload = {
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{colab_url}/chat",
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status_code != 200:
                    err_detail = await response.aread()
                    raise RuntimeError(
                        f"Colab returned error {response.status_code}: "
                        f"{err_detail.decode('utf-8', errors='replace')}"
                    )

                buffer = ""
                async for chunk in response.aiter_text():
                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                data_json = json.loads(data_str)
                                token = data_json.get("token", "")
                                if token:
                                    yield token
                            except Exception:
                                pass


model_service = ModelService()
