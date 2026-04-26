import os
from dotenv import load_dotenv
from loguru import logger
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    TextFrame,
    TTSSpeakFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import WebSocketRunnerArguments
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.vobiz import VobizFrameSerializer
from pipecat.transcriptions.language import Language
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

load_dotenv(override=True)

SYSTEM_PROMPT = """
You are Raajesh, a warm and friendly sales executive at Sri Venkanna Hero dealership, R.C. Puram, Hyderabad.

You are making an outbound call to a customer who has previously shown interest in buying a Hero two-wheeler. Your job is to have a genuine, human conversation — not read a script. Think of it like catching up with someone and naturally guiding them towards visiting the showroom.

CONVERSATION FLOW:
Start by introducing yourself briefly and ask them if they are looking to buy a two-wheeler. If they say yes, ask which vehicle they have in mind. Once they mention a model, casually share that the showroom currently has up to ₹4,000 off on purchases this month.

Gently encourage them to visit the showroom for a test ride. If they show interest, ask which area they are from and whether they are planning to pay by cash or finance. Help them understand the advantage of buying from our showroom — without being pushy.

── Telugu rules ──
Style: Casual, natural Telugu mixed with English words — the way people actually speak in Hyderabad. Never formal or pure Telugu. Sentences must be grammatically correct and easy for customers to understand.
Script: Telugu script only for all Telugu words — CRITICAL: no Roman transliteration ever, it degrades TTS quality. Use only Telugu Unicode characters (U+0C00–U+0C7F) for Telugu words. NEVER use Tamil, Gujarati, Devanagari, Kannada, or any other Indian script.
Sentence endings: use the correct ending for each sentence type — questions end with ?, exclamations end with !, statements end with । (danda). NEVER end any sentence with a plain period (.) — it causes fatal TTS splitting errors. Example: "నమస్తే సర్ । మీకు టైమ్ ఉందా? చాలా బాగుంది!"
Sentence length: keep each sentence under 20 words — long sentences cause unnatural TTS breathing.

Always be to the point, keep your responses short, to the point. Be conversational. 

SPEECH STYLE:
Your responses are read aloud by a voice engine. Write them the way you would actually say them — with warmth, emotion, and natural rhythm. Use these techniques:

- Use , (comma) for a short pause mid-sentence, Use ! (exclamation) to show warmth and enthusiasm
- Vary your sentence rhythm — don't make every sentence the same length or structure.
- NEVER use … or ... (ellipsis/dots) — they cause TTS splitting errors.
- NEVER use line breaks or blank lines — they create empty voice chunks that crash the engine.

PRICING AND NUMBERS:
Never quote any price, EMI amount, interest rate, down payment, or any specific figure. If the customer asks about pricing, costs, EMI, or anything money-related, tell them warmly that our support team will get in touch with the exact details. Do not guess or make up numbers. Always write numbers with commas: ₹4,000 not ₹4000.

UNKNOWN QUESTIONS:
If the customer asks something you don't know — about the vehicle, dealership, offers, or schemes — do not guess. Simply tell them you will check with the support team and they will get back with the right information. Keep it natural and reassuring, not robotic.

Never use emojis or special symbols — they will be read aloud and sound strange.
Never speak for more than 5 minutes total. Never make up information you don't have.
"""


class TranscriptionLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if (
            isinstance(frame, TranscriptionFrame)
            and direction == FrameDirection.DOWNSTREAM
        ):
            logger.info(f"[STT] User said: {frame.text}")
        await self.push_frame(frame, direction)


class BotResponseLogger(FrameProcessor):
    """Collects LLM text chunks and logs the complete bot response."""

    def __init__(self):
        super().__init__()
        self._buffer = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, TextFrame):
                self._buffer.append(frame.text)
            elif isinstance(frame, LLMFullResponseEndFrame) and self._buffer:
                full_text = "".join(self._buffer).strip()
                if full_text:
                    logger.info(f"[BOT] {full_text}")
                self._buffer = []
        await self.push_frame(frame, direction)


# ── Bot pipeline ───────────────────────────────────────────────────────────────


async def run_bot(
    transport: BaseTransport,
    handle_sigint: bool,
    customer_name: str | None = None,
    transcript_out: list | None = None,
):
    greeting = f"హలో, నేను Raajesh. హీరో వెంకన్న షోరూమ్ నుంచి కాల్ చేస్తున్నాను. ఇది {customer_name} గారేనా మాట్లాడేది ?"

    llm = OpenAILLMService(
        api_key="local",
        base_url=os.getenv("LOCAL_LLM_URL", "http://164.52.198.104:8049/v1"),
        model="google/gemma-4-26B-A4B-it",
    )

    stt = SarvamSTTService(
        api_key=os.getenv("SARVAM_API_KEY", ""),
        settings=SarvamSTTService.Settings(  # type: ignore
            model="saarika:v2.5",
            vad_signals=True,
        ),
    )

    
    tts = SarvamTTSService(
        api_key=os.getenv("SARVAM_API_KEY", ""),
        settings=SarvamTTSService.Settings(
            voice="shubh",
            model="bulbul:v3-beta",
            language=Language.TE,
            temperature=0.9,
            pace=1.0
        ),
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "begin"},
        {"role": "assistant", "content": greeting},
    ]

    context = LLMContext(messages)  # type: ignore

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(user_turn_stop_timeout=0.2),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            TranscriptionLogger(),
            user_aggregator,
            llm,
            BotResponseLogger(),
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            enable_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Outbound call connected — customer: {customer_name or 'unknown'}")
        await task.queue_frames([TTSSpeakFrame(text=greeting)])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Call disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)
    try:
        await runner.run(task)
    finally:
        if transcript_out is not None:
            for msg in context._messages[2:]:  # skip system + seeded "begin"
                role = msg.get("role", "")
                content = msg.get("content", "")
                if (
                    role in ("user", "assistant")
                    and isinstance(content, str)
                    and content.strip()
                ):
                    transcript_out.append(
                        {"role": role, "text": content, "index": len(transcript_out)}
                    )
            logger.info(f"Transcript captured — {len(transcript_out)} turns")



# ── Entry point ────────────────────────────────────────────────────────────────


async def bot(
    runner_args: WebSocketRunnerArguments,
    customer_name: str | None = None,
    transcript_out: list | None = None,
):
    transport_type, call_data = await parse_telephony_websocket(runner_args.websocket)
    logger.info(f"Transport: {transport_type}")

    serializer = VobizFrameSerializer(
        stream_id=call_data["stream_id"],
        call_id=call_data["call_id"],
        auth_id=os.getenv("VOBIZ_AUTH_ID", ""),
        auth_token=os.getenv("VOBIZ_AUTH_TOKEN", ""),
        params=VobizFrameSerializer.InputParams(
            vobiz_sample_rate=8000,
            auto_hang_up=True,
        ),
    )

    transport = FastAPIWebsocketTransport(
        websocket=runner_args.websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=serializer,
        ),
    )

    await run_bot(
        transport,
        runner_args.handle_sigint,
        customer_name=customer_name,
        transcript_out=transcript_out,
    )
