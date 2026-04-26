import os
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from loguru import logger

from whatsapp.models import WhatsAppWebhookPayload
from whatsapp.client import send_text_message

router = APIRouter(prefix="/webhook/whatsapp", tags=["whatsapp"])


@router.get("")
async def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
) -> PlainTextResponse:
    verify_token = os.getenv("WHATSAPP_VERIFY_TOKEN", "")
    if hub_mode == "subscribe" and hub_verify_token == verify_token:
        logger.info("[WhatsApp] Webhook verified")
        return PlainTextResponse(content=hub_challenge)
    logger.warning("[WhatsApp] Webhook verification failed")
    return PlainTextResponse(content="Forbidden", status_code=403)


@router.post("")
async def receive_message(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        payload = WhatsAppWebhookPayload(**body)
    except Exception as e:
        logger.error(f"[WhatsApp] Payload parse error: {e}")
        return JSONResponse({"status": "error"}, status_code=400)

    for entry in payload.entry:
        for change in entry.changes:
            messages = change.value.messages
            if not messages:
                continue
            for msg in messages:
                if msg.get("type") != "text":
                    continue
                from_number = msg.get("from", "")
                text = msg.get("text", {}).get("body", "")
                logger.info(f"[WhatsApp] Incoming from {from_number}: {text}")
                await _handle_reply(from_number, text, request)

    return JSONResponse({"status": "ok"})


async def _handle_reply(from_number: str, text: str, request: Request) -> None:
    try:
        from helpers.db import get_call_by_phone
        call = await get_call_by_phone(from_number)
        customer_name = call.get("customer_name", "") if call else ""

        reply = await _llm_reply(text, customer_name)
        await send_text_message(from_number, reply)
    except Exception as e:
        logger.error(f"[WhatsApp] Reply error: {e}")


async def _llm_reply(user_message: str, customer_name: str) -> str:
    import os
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=os.getenv("LOCAL_LLM_API_KEY", "local"),
        base_url=os.getenv("LOCAL_LLM_URL", ""),
    )

    system = (
        "You are Raajesh, a friendly sales executive at Sri Venkanna Hero Dealership, "
        "R.C. Puram, Hyderabad. A customer has replied to your WhatsApp message. "
        "Reply in English, warmly and briefly (2-3 sentences max). "
        "Never quote prices or EMI amounts — say the support team will share details. "
        "Encourage them to visit the showroom for a test ride."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]

    resp = await client.chat.completions.create(
        model=os.getenv("LOCAL_LLM_MODEL", ""),
        messages=messages,
        max_tokens=150,
    )
    return resp.choices[0].message.content.strip()
