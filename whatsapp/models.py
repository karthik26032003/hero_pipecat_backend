from pydantic import BaseModel
from typing import Any


class WhatsAppMessage(BaseModel):
    from_: str
    id: str
    timestamp: str
    type: str
    text: dict[str, Any] | None = None

    class Config:
        populate_by_name = True


class WhatsAppContact(BaseModel):
    profile: dict[str, Any]
    wa_id: str


class WhatsAppValue(BaseModel):
    messaging_product: str
    metadata: dict[str, Any]
    contacts: list[WhatsAppContact] | None = None
    messages: list[dict[str, Any]] | None = None
    statuses: list[dict[str, Any]] | None = None


class WhatsAppChange(BaseModel):
    value: WhatsAppValue
    field: str


class WhatsAppEntry(BaseModel):
    id: str
    changes: list[WhatsAppChange]


class WhatsAppWebhookPayload(BaseModel):
    object: str
    entry: list[WhatsAppEntry]
