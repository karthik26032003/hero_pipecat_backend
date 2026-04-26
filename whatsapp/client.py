import os
import httpx
from loguru import logger


GRAPH_API_VERSION = "v25.0"
GRAPH_API_BASE = f"https://graph.facebook.com/{GRAPH_API_VERSION}"


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {os.getenv('WHATSAPP_TOKEN', '')}",
        "Content-Type": "application/json",
    }


def _phone_number_id() -> str:
    return os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")


async def send_template_message(
    to: str,
    template_name: str,
    language_code: str = "en",
    components: list | None = None,
) -> dict:
    url = f"{GRAPH_API_BASE}/{_phone_number_id()}/messages"
    payload: dict = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "template",
        "template": {
            "name": template_name,
            "language": {"code": language_code},
        },
    }
    if components:
        payload["template"]["components"] = components

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=_headers(), json=payload, timeout=15)
        data = resp.json()
        if resp.status_code != 200:
            logger.error(f"[WhatsApp] Template send failed: {data}")
        else:
            logger.info(f"[WhatsApp] Template sent to {to}: {data}")
        return data


async def send_text_message(to: str, text: str) -> dict:
    url = f"{GRAPH_API_BASE}/{_phone_number_id()}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text},
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=_headers(), json=payload, timeout=15)
        data = resp.json()
        if resp.status_code != 200:
            logger.error(f"[WhatsApp] Text send failed: {data}")
        else:
            logger.info(f"[WhatsApp] Text sent to {to}")
        return data
