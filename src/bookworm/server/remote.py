try:
    import modal
except ImportError:
    raise ImportError(
        "modal is required for remote services. "
        "Install it with: uv sync --group remote"
    )


import os
from fastapi import Header, HTTPException
from bookworm.server.requests import EmbedRequest
from bookworm.server.responses import EmbedResponse

app = modal.App("bookworm-remote-service")

bookworm_secret = modal.Secret.from_name("bookworm-auth")

EMBEDDING_GPU = "A10"

LLM_GPU = "A10"
LLM_MODEL_NAME = "Qwen/Qwen3-8B"
LLM_MAX_MODEL_LEN = 8192

embedding_image = (
    modal.Image.debian_slim()
    .uv_pip_install(
        "sentence-transformers",
        "torch",
        "pydantic",
        "fastapi[standard]",
    )
    .add_local_python_source("bookworm")
)

llm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.13.0",
        "fastapi[standard]",
        "huggingface_hub",
        "hf_transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

model_cache = modal.Volume.from_name(
    "bookworm-model-cache", create_if_missing=True
)

TOOL_CALL_PARSERS: dict[str, str] = {"Qwen/Qwen3": "hermes"}


def get_tool_parser(model_name: str) -> str | None:
    for prefix, parser in TOOL_CALL_PARSERS.items():
        if model_name.startswith(prefix):
            return parser
    return None


@app.cls(image=embedding_image, gpu=EMBEDDING_GPU, secrets=[bookworm_secret])
class ModalEmbeddingService:
    @modal.enter()
    def setup(self):
        from bookworm.services.embedding_service import EmbeddingService
        from bookworm.embedding import BGEM3Embedder

        self.service = EmbeddingService(embedder=BGEM3Embedder())
        self.api_key = os.environ["BOOKWORM_API_KEY"]

    @modal.fastapi_endpoint(method="POST")
    def embed(
        self, request: EmbedRequest, authorization: str = Header()
    ) -> EmbedResponse:
        token = authorization.removeprefix("Bearer ")
        if token != self.api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")
        if not request.texts:
            return EmbedResponse(embeddings=[])
        return EmbedResponse(embeddings=self.service.embed(request.texts))


@app.cls(
    image=llm_image,
    gpu=LLM_GPU,
    secrets=[bookworm_secret],
    volumes={"/root/.cache/huggingface": model_cache},
    scaledown_window=300,
)
class ModalLLMService:
    model_name: str = LLM_MODEL_NAME

    @modal.web_server(port=8000, startup_timeout=1800)
    def serve(self):
        import subprocess

        api_key = os.environ["BOOKWORM_API_KEY"]

        cmd = [
            "vllm",
            "serve",
            self.model_name,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--api-key",
            api_key,
            "--max-model-len",
            str(LLM_MAX_MODEL_LEN),
        ]
        tool_parser = get_tool_parser(self.model_name)
        if tool_parser:
            cmd += [
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                tool_parser,
            ]

        self.proc = subprocess.Popen(cmd)
