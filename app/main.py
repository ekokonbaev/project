# main.py
import uvicorn
import logging
from typing import Optional
import asyncio
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

LOG = logging.getLogger("uvicorn.error")
logging.basicConfig(level=logging.INFO)

OPENROUTER_API_KEY = "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"
SECRET_KEY = "replace_with_secret_key"

@asynccontextmanager
async def lifespan(app: FastAPI):
    LOG.info("Starting app; registered routes:")
    for route in app.routes:
        LOG.info("%s %s", getattr(route, "methods", None), getattr(route, "path", None))
    yield
    LOG.info("Shutting down")

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True)

class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = "meta-llama/llama-3.3-70b-instruct:free"
    max_tokens: Optional[int] = 512

class ChatResponse(BaseModel):
    response: str

@app.get("/health")
async def health():
    return {"status": "ok"}

async def call_openrouter(prompt: str, model: str, max_tokens: int) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Ты — Орион by Ekokonbaev."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
    }

    # простой retry для 429/5xx
    attempts = 3
    backoff = 0.8
    async with httpx.AsyncClient(timeout=60.0) as client:
        for i in range(attempts):
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                try:
                    choices = data.get("choices")
                    if choices and isinstance(choices, list):
                        content = choices[0].get("message", {}).get("content") or choices[0].get("text")
                        return content if isinstance(content, str) else str(content)
                    return str(data)
                except Exception:
                    LOG.error("Invalid model response: %s", data)
                    raise HTTPException(status_code=502, detail="Invalid model response format")
            if resp.status_code in (429, 502, 503):
                LOG.warning("Upstream %s (attempt %d/%d): %s", resp.status_code, i+1, attempts, resp.text[:200])
                await asyncio.sleep(backoff * (2 ** i))
                continue
            # other error
            LOG.error("OpenRouter returned %s: %s", resp.status_code, resp.text)
            raise HTTPException(status_code=502, detail="Upstream error")
    raise HTTPException(status_code=503, detail="Upstream unavailable after retries")

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, x_secret: Optional[str] = Header(None)):
    if x_secret != SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")
    result = await call_openrouter(req.message, req.model, req.max_tokens)
    return ChatResponse(response=result)

if __name__ == "__main__":
    
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
