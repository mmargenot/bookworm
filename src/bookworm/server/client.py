import logging
import time

import httpx
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from bookworm.server.requests import EmbedRequest
from bookworm.server.responses import EmbedResponse

logger = logging.getLogger(__name__)


class BookwormClient:
    def __init__(
        self,
        embedding_url: str,
        completion_url: str,
        api_key: str,
        model_name: str,
    ):
        self.embedding_url = embedding_url
        self.api_key = api_key
        self.model_name = model_name
        self.headers = {
            "Authorization": f"Bearer {api_key}",
        }
        self.http_client = httpx.AsyncClient(timeout=300)
        self.completion_client = AsyncOpenAI(
            base_url=f"{completion_url}/v1",
            api_key=api_key,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        total_chars = sum(len(t) for t in texts)
        logger.info(f"Embedding {len(texts)} texts ({total_chars} chars)")
        request = EmbedRequest(texts=texts)
        t0 = time.perf_counter()
        response = await self.http_client.post(
            self.embedding_url,
            json=request.model_dump(),
            headers=self.headers,
        )
        response.raise_for_status()
        result = EmbedResponse(**response.json()).embeddings
        dt = time.perf_counter() - t0
        logger.info(f"Received {len(result)} embeddings in {dt:.2f}s")
        return result

    async def complete(
        self,
        messages: list[ChatCompletionMessageParam],
        thinking: bool = False,
    ):
        logger.info(
            f"Completion request: {len(messages)} messages, thinking={thinking}"
        )
        t0 = time.perf_counter()
        extra_body = {"chat_template_kwargs": {"enable_thinking": thinking}}
        result = await self.completion_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            extra_body=extra_body,
        )
        dt = time.perf_counter() - t0
        usage = result.usage
        if usage:
            logger.info(
                f"Completion done in {dt:.2f}s: "
                f"{usage.prompt_tokens} prompt, "
                f"{usage.completion_tokens} completion, "
                f"{usage.total_tokens} total tokens"
            )
        else:
            logger.info(f"Completion done in {dt:.2f}s")
        return result

    async def close(self):
        await self.http_client.aclose()
