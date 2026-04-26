import os
import sys

from contextlib import asynccontextmanager
from dotenv import load_dotenv
import aiohttp
import httpx
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket
from whatsapp.router import router as whatsapp_router
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger as _logger
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Logging ────────────────────────────────────────────────────────────────────
def _log_filter(record):
    if record["level"].no >= 20:  # INFO and above — always show
        return True
    msg = record["message"]
    return any(kw in msg for kw in ["Generating TTS:", "TTFB:", "type='data'"])

_logger.remove()
_logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level:<8}</level> | {message}",
    colorize=True,
    filter=_log_filter,
)

load_dotenv(override=True)


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    from helpers.db import init_db, close_db
    await init_db()
    app.state.call_metadata = {}
    app.state.session = aiohttp.ClientSession()
    yield
    await app.state.session.close()
    await close_db()


app = FastAPI(title="Sri Venkanna Hero Bot", version="1.0.0", lifespan=lifespan)
app.include_router(whatsapp_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ws_url(host: str) -> str:
    return f"wss://{host}/ws"


# ── Outbound — initiate call ───────────────────────────────────────────────────

class OutboundCallRequest(BaseModel):
    phone_number: str
    customer_name: str | None = None


async def _make_vobiz_call(session: aiohttp.ClientSession, to_number: str, answer_url: str):
    auth_id     = os.getenv("VOBIZ_AUTH_ID",    "")
    auth_token  = os.getenv("VOBIZ_AUTH_TOKEN", "")
    from_number = os.getenv("VOBIZ_PHONE_NUMBER", "")

    headers = {
        "Content-Type": "application/json",
        "X-Auth-ID":    auth_id,
        "X-Auth-Token": auth_token,
    }
    data = {
        "to":            to_number,
        "from":          from_number,
        "answer_url":    answer_url,
        "answer_method": "POST",
    }
    url = f"https://api.vobiz.ai/api/v1/Account/{auth_id}/Call/"

    async with session.post(url, headers=headers, json=data) as resp:
        if resp.status != 201:
            error = await resp.text()
            raise Exception(f"VoBiz API error ({resp.status}): {error}")
        return await resp.json()


@app.post("/start")
async def start_outbound_call(body: OutboundCallRequest, request: Request) -> JSONResponse:
    host       = request.headers.get("host", "")
    protocol   = "https" if not host.startswith(("localhost", "127.0.0.1")) else "http"
    answer_url = f"{protocol}://{host}/answer"

    try:
        result    = await _make_vobiz_call(request.app.state.session, body.phone_number, answer_url)
        call_uuid = result.get("request_uuid") or result.get("call_uuid") or "unknown"
    except Exception as e:
        _logger.error(f"[OUTBOUND] Failed to initiate call: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    from helpers.db import insert_call
    await insert_call(
        call_uuid     = call_uuid,
        from_number   = os.getenv("VOBIZ_PHONE_NUMBER", ""),
        to_number     = body.phone_number,
        customer_name = body.customer_name,
    )

    app.state.call_metadata[call_uuid] = {
        "from_number":   os.getenv("VOBIZ_PHONE_NUMBER", ""),
        "to_number":     body.phone_number,
        "customer_name": body.customer_name,
        "db_inserted":   True,
    }

    _logger.info(f"[OUTBOUND] Call initiated → {body.phone_number} | uuid={call_uuid}")
    return JSONResponse({"call_uuid": call_uuid, "status": "call_initiated", "phone_number": body.phone_number})


# ── Answer hook ────────────────────────────────────────────────────────────────

@app.api_route("/answer", methods=["GET", "POST"])
async def answer(
    request: Request,
    CallUUID: str = Query(None),
) -> HTMLResponse:
    from_number = ""
    to_number   = ""

    if request.method == "POST":
        form = await request.form()
        if not CallUUID:
            CallUUID = form.get("CallUUID", "")
        from_number = form.get("From", "")
        to_number   = form.get("To",   "")

    if CallUUID:
        existing = app.state.call_metadata.get(CallUUID, {})
        existing.update({"from_number": from_number, "to_number": to_number})
        app.state.call_metadata[CallUUID] = existing

    host            = request.headers.get("host", "")
    ws_url          = f"{_ws_url(host)}?call_uuid={CallUUID}" if CallUUID else _ws_url(host)
    protocol        = "https" if not host.startswith(("localhost", "127.0.0.1")) else "http"
    record_callback = f"{protocol}://{host}/recording-ready"

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Record fileFormat="mp3" maxLength="3600" recordSession="true" callbackUrl="{record_callback}" callbackMethod="POST">
    </Record>
    <Stream bidirectional="true" audioTrack="inbound" contentType="audio/x-mulaw;rate=8000" keepCallAlive="true">
        {ws_url}
    </Stream>
</Response>"""

    _logger.info(f"[{CallUUID}] Answer XML → ws={ws_url}")
    return HTMLResponse(content=xml, media_type="application/xml")


# ── WebSocket — pipecat pipeline ───────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    call_uuid: str = Query(None),
):
    await websocket.accept()
    _logger.info(f"WebSocket connected — call_uuid={call_uuid}")

    metadata      = app.state.call_metadata.pop(call_uuid, {})
    from_number   = metadata.get("from_number",   "unknown")
    to_number     = metadata.get("to_number",     "unknown")
    customer_name = metadata.get("customer_name", None)

    from helpers.db import insert_call, insert_transcript, update_call_ended
    if not metadata.get("db_inserted"):
        await insert_call(call_uuid, from_number, to_number, customer_name)

    transcript: list = []

    try:
        from helpers.bot import bot
        from pipecat.runner.types import WebSocketRunnerArguments

        runner_args = WebSocketRunnerArguments(websocket=websocket)
        await bot(runner_args, customer_name=customer_name, transcript_out=transcript)

    except Exception as e:
        _logger.error(f"[{call_uuid}] Bot error: {e}")
        await websocket.close()

    finally:
        await update_call_ended(call_uuid)
        if transcript:
            await insert_transcript(call_uuid, transcript)
            _logger.info(f"[{call_uuid}] Transcript saved — {len(transcript)} turns")

        try:
            from whatsapp.client import send_template_message
            await send_template_message(
                to=to_number,
                template_name=os.getenv("WHATSAPP_TEMPLATE_NAME", ""),
                components=[{
                    "type": "body",
                    "parameters": [{"type": "text", "text": customer_name or "Customer"}],
                }],
            )
            from helpers.db import mark_whatsapp_sent
            await mark_whatsapp_sent(call_uuid)
        except Exception as e:
            _logger.error(f"[{call_uuid}] WhatsApp send failed: {e}")


# ── Recording callback ─────────────────────────────────────────────────────────

_RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), "..", "recordings")
os.makedirs(_RECORDINGS_DIR, exist_ok=True)


@app.api_route("/recording-ready", methods=["GET", "POST"])
async def recording_ready(request: Request) -> HTMLResponse:
    data          = await request.form()
    recording_url = str(data.get("RecordUrl", "") or "")
    call_uuid     = str(data.get("CallUUID",  "") or "")

    if recording_url and call_uuid:
        from helpers.db import update_call_recording
        try:
            auth_id    = os.getenv("VOBIZ_AUTH_ID",    "")
            auth_token = os.getenv("VOBIZ_AUTH_TOKEN", "")
            dl_headers = {"X-Auth-ID": auth_id, "X-Auth-Token": auth_token}
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    recording_url,
                    headers=dl_headers,
                    follow_redirects=True,
                    timeout=60,
                )
            local_path = os.path.join(_RECORDINGS_DIR, f"{call_uuid}.mp3")
            with open(local_path, "wb") as f:
                f.write(resp.content)
            local_url = f"/recordings/{call_uuid}.mp3"
            await update_call_recording(call_uuid, local_url)
            _logger.info(f"[{call_uuid}] Recording saved → {local_url} ({len(resp.content)} bytes)")
        except Exception as e:
            _logger.error(f"[{call_uuid}] Recording download failed: {e}")
            await update_call_recording(call_uuid, recording_url)

    return HTMLResponse(content="<Response></Response>", media_type="application/xml")


# ── Logs API ───────────────────────────────────────────────────────────────────

@app.get("/logs")
async def get_logs() -> JSONResponse:
    from helpers.db import get_calls
    return JSONResponse(await get_calls())


@app.get("/logs/{call_uuid}")
async def get_log_detail(call_uuid: str) -> JSONResponse:
    from helpers.db import get_call, get_transcript
    call = await get_call(call_uuid)
    if not call:
        return JSONResponse({"error": "Not found"}, status_code=404)
    transcript = await get_transcript(call_uuid)
    return JSONResponse({"call": call, "transcript": transcript})


class StatusUpdate(BaseModel):
    status: str


@app.patch("/logs/{call_uuid}/status")
async def update_status(call_uuid: str, body: StatusUpdate) -> JSONResponse:
    from helpers.db import update_call_status
    await update_call_status(call_uuid, body.status)
    return JSONResponse({"ok": True})


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})
