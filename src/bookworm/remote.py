try:
    import modal
except ImportError:
    raise ImportError(
        "modal is required for remote services. "
        "Install it with: uv sync --group remote"
    )


app = modal.App("bookworm-remote-service")


EMBEDDING_GPU = "A10"

LLM_GPU = "A10"
LLM_MODEL_NAME = "Qwen/Qwen3-32B"

embedding_image = modal.Image.debian_slim().uv_pip_install(
    "sentence-transformers", "torch"
)

completion_image = modal.Image.debian_slim().uv_pip_install("vllm", "torch")

TOOL_CALL_PARSERS: dict[str, str] = {"Qwen/Qwen3": "hermes"}


def get_tool_parser(model_name: str) -> str | None:
    for prefix, parser in TOOL_CALL_PARSERS.items():
        if model_name.startswith(prefix):
            return parser
    return None


@app.cls(image=embedding_image, gpu=EMBEDDING_GPU)
class ModalEmbeddingService:
    @modal.enter()
    def setup(self):
        from bookworm.services.embedding_service import EmbeddingService
        from bookworm.embedding import BGEM3Embedder

        self.service = EmbeddingService(embedder=BGEM3Embedder())

    @modal.method()
    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.service.embed(texts)


@app.cls(image=completion_image, gpu=LLM_GPU)
class ModalLLMService:
    model_name: str = LLM_MODEL_NAME

    @modal.enter()
    def serve(self):
        import subprocess

        cmd = [
            "vllm",
            "serve",
            self.model_name,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ]
        tool_parser = get_tool_parser(self.model_name)
        if tool_parser:
            cmd += [
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                tool_parser,
            ]

        self.proc = subprocess.Popen(cmd)
