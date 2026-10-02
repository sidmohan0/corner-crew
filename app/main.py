import asyncio
import json
from contextlib import AsyncExitStack, asynccontextmanager, suppress

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

from app.catalog import catalog
from app.config import Settings
from app.memory import MemoryStore
from app.pipeline import Pipeline
from app.schemas import ChatRequest, ChatResponse


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    configs = {
        "anthropic": (
            "https://api.anthropic.com/v1/",
            settings.anthropic_api_key,
            settings.anthropic_model,
        ),
        "openai": ("https://api.openai.com/v1/", settings.openai_api_key, settings.openai_model),
        "gemini": (
            "https://generativelanguage.googleapis.com/v1beta/openai/",
            settings.gemini_api_key,
            settings.gemini_model,
        ),
        "local": (settings.local_base_url, settings.local_api_key, settings.local_model),
        "minicpm": (settings.minicpm_base_url, settings.minicpm_api_key, settings.minicpm_model),
    }
    clients: dict[str, AsyncOpenAI] = {}
    deployments = []
    memory = MemoryStore(settings.memory_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with AsyncExitStack() as stack:
            for provider, (url, key, _) in configs.items():
                if key.get_secret_value():
                    clients[provider] = await stack.enter_async_context(
                        AsyncOpenAI(
                            api_key=key.get_secret_value(),
                            base_url=url,
                            timeout=settings.request_timeout_seconds,
                            max_retries=0,
                        )
                    )
            deployments.extend(catalog(settings, configs, clients))
            yield
        clients.clear()
        deployments.clear()

    app = FastAPI(title="Corner Crew API", version="0.2.0", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/providers")
    async def providers():
        """Configuration status only; this does not probe upstream availability."""
        return {
            name: {"configured": name in clients, "default_model": config[2] or None}
            for name, config in configs.items()
        }

    @app.get("/models")
    async def models():
        """Configured deployment capabilities; region/prices are operator assertions."""
        return [
            dict(provider=d.provider, is_local=d.is_local, **d.metadata.model_dump())
            for d in deployments
        ]

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        return await Pipeline(request, deployments, clients, memory, settings).run()

    @app.post("/chat/stream")
    async def stream_chat(request: ChatRequest):
        """NDJSON: trace events as they occur, then one result or error. No token simulation."""

        async def stream():
            queue = asyncio.Queue()
            pipeline = Pipeline(
                request,
                deployments,
                clients,
                memory,
                settings,
                on_event=lambda event: queue.put_nowait({"type": "trace", "event": event}),
            )

            async def produce():
                try:
                    result = await pipeline.run()
                    await queue.put({"type": "result", "data": result.model_dump()})
                except HTTPException as exc:
                    await queue.put(
                        {"type": "error", "status": exc.status_code, "detail": exc.detail}
                    )
                except Exception:  # noqa: BLE001 -- always terminate the stream with a sanitized error
                    await queue.put(
                        {"type": "error", "status": 500, "detail": "Unexpected pipeline failure."}
                    )
                finally:
                    queue.put_nowait(None)

            task = asyncio.create_task(produce())
            try:
                while True:
                    event = await queue.get()
                    if event is None:
                        break
                    yield json.dumps(event) + "\n"
            finally:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

        return StreamingResponse(
            stream(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


app = create_app()
