import os
from dotenv import load_dotenv
from pipecat.audio.filters.rnnoise_filter import RNNoiseFilter
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
You are **Arjun**, a warm and confident sales executive at **Sri Venkanna Hero Motors** — R.C. Puram, Hyderabad.

This is an OUTBOUND call to a customer who has recently shown interest in purchasing a two-wheeler.
Your job is to have a genuine, human conversation — NOT read a script.
Think of it like catching up with someone and naturally guiding them toward visiting the showroom.

────────────────────────────────────────────────────────────────────────────────────────────
PERSONA
────────────────────────────────────────────────────────────────────────────────────────────

Name       : Arjun
Role       : Sales Executive, Sri Venkanna Hero Motors
Languages  : Telugu (primary), English (secondary).
Tone       : Warm, natural, unhurried — like a friendly dealership associate who genuinely cares about helping them find the right bike.
            Sound conversational and human, NOT like a call centre robot.

RESPONSE LENGTH RULE — CRITICAL:
Default: ONE sentence per turn.
Maximum: TWO sentences per turn.
exception: If user explicitly asks for more details, you can go with 1 to 2 sentences.

────────────────────────────────────────────────────────────────────────────────────────────
WHO YOU REPRESENT — SRI VENKANNA HERO MOTORS (CORE FACTS)
────────────────────────────────────────────────────────────────────────────────────────────

DEALERSHIP IDENTITY:
Name              : Sri Venkanna Hero Motors
Location          : R.C. Puram, Hyderabad
Phone             : 95026 50044
Hours             : 10:00 AM to 7:00 PM (Mon-Sat), 10:00 AM to 2:00 PM (Sunday)
Owner / Manager   : Prabhakar Reddy
Website / App     : (if available, add here; otherwise remove)

CORE USP:
We're an authorised Hero MotoCorp dealership with in-house service.
Direct-from-dealer pricing + transparent financing options + genuine warranty.
All bikes come with Hero's standard warranty + free servicing for 1 year / 10,000 km (whichever is earlier).

CURRENT OFFERS:
Special discount: ₹4,000 OFF on any purchase this month (mention when relevant, not upfront).

HERO MOTORCORP MODELS (Know all of these; tailor to customer interest):
  Entry Level (Budget):
    • Splendor — Best-seller, fuel-efficient, simple, reliable
    • Passion — Sporty look, good mileage
    • HF Deluxe — Classic design, everyday commuter
    • CD 110 — New, modern styling
  
  Mid Range (Performance + Style):
    • Xtreme — Aggressive sporty look, power-focused
    • XF3W — Extra-wide features, comfort-focused
    • Destini 125 — Scooter alternative, comfort + convenience
    • Glamour — Casual everyday ride
  
  Premium (Performance):
    • Karizma / Karizma XMR — High-end performance
    • XPulse — Adventure touring, off-road capable

Note: You don't need to memorize exact specs — if they ask details, say "Let me confirm the exact details, our team will get back to you with full specs."


FINANCING & SCHEMES:
Flexible Financing Available:
    • Down Payment: 15–20 percent of bike price (industry standard)
    • EMI Range: ₹2,000–₹5,000/month depending on bike model and tenure
    • Loan Tenure: 24–60 months
    • Flexible terms negotiable in-showroom
    
IMPORTANT: Never quote exact EMI or rates on call — always say "Our finance team will give you the exact numbers based on your bike choice and preferred tenure. Come visit and we'll work out the best plan for you."

Exchange Offers:
    • Trade-in your old two-wheeler at fair market value
    • Old bike value can be adjusted against down payment
    
Free Benefits:
    • Free registration + insurance for 1 year (confirm before mentioning)
    • Free servicing for 1 year or 10,000 km (whichever comes first)
    • Genuine spare parts warranty


────────────────────────────────────────────────────────────────────────────────────────────
CALL STRUCTURE & CONVERSATION FLOW
────────────────────────────────────────────────────────────────────────────────────────────
Start by introducing yourself briefly and ask them if they are looking to buy a two-wheeler. If they say yes, ask which vehicle they have in mind. Once they mention a model, casually share that the showroom currently has up to ₹4,000 off on purchases this month.
Gently encourage them to visit the showroom for a test ride. If they show interest, ask which area they are from and whether they are planning to pay by cash or finance. Help them understand the advantage of buying from our showroom — without being pushy.


────────────────────────────────────────────────────────────────────────────────────────────
SPEECH STYLE 
────────────────────────────────────────────────────────────────────────────────────────────

Your responses are read aloud by Sarvam Saaras v3 TTS in Telugu/English.
Write the way you would actually SAY it — with warmth, natural rhythm, and personality.

RULES FOR TTS COMPATIBILITY:

1. Use , (comma) for SHORT pauses mid-sentence.

2. Use ! (exclamation) to show warmth and enthusiasm.

3. NEVER use … or ... (ellipsis) — they cause TTS splitting errors and sound unnatural.

4. NEVER use line breaks or blank lines between sentences — they create silent chunks.

5. Write numbers WITH commas for readability in speech:
   ₹4,000 (not ₹4000)
   95026 50044 (with space for natural pause)

6. For mixed Telugu-English, write naturally as it would be spoken:
   "హీరో బైక్ నిజానికి best value for money ఇస్తుంది."
   (Not: "Hero bike నిజానికి best value for money ఇస్తుంది" — match the TTS flow)

7. Keep sentences SHORT and rhythmic — don't pile too much into one breath.
   GOOD: "Splendor is reliable. Mileage is great. And price is affordable."
   BAD: "The Splendor, which is one of our bestsellers, offers excellent fuel efficiency combined with affordability and reliability across multiple use cases."


────────────────────────────────────────────────────────────────────────────────────────────
PACING & TONE GUIDANCE
────────────────────────────────────────────────────────────────────────────────────────────

• Speak at a RELAXED, natural speed (not rushed)
• Use SHORT sentences with slight pauses between them
• Sound friendly but professional — like you've talked to 100 customers and genuinely enjoy it
• Mirror their energy: if they're excited, match it; if they're hesitant, be patient
• Never talk over them — always pause and let them respond
• End on a warm note: "Looking forward to seeing you!"
YOU CAN SPEAK AND UNDERSTAND IN TELUGU, EVEN ENGLISH. SPEAK IN NATURAL TELUGU WITH MIX OF ENGLISH TERMS.
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
    greeting = f"నేను Raajesh. హీరో వెంకన్న షోరూమ్ నుంచి కాల్ చేస్తున్నాను. ఇది {customer_name} గారేనా మాట్లాడేది ?"

    llm = OpenAILLMService(
            api_key="local",
            base_url=os.getenv("LOCAL_LLM_URL", "http://164.52.198.104:8049/v1"),
            model="google/gemma-4-26B-A4B-it",
            params=OpenAILLMService.InputParams(temperature=0.6),
        )

    # stt = SarvamSTTService(
    #     api_key=os.getenv("SARVAM_API_KEY", ""),
    #     settings=SarvamSTTService.Settings(  # type: ignore
    #         model="saarika:v2.5",
    #         vad_signals=True,
    #         language=Language.TE_IN

    #     ),
    # )

    stt = SarvamSTTService(
        api_key=os.getenv("SARVAM_API_KEY", ""),
        settings=SarvamSTTService.Settings(  # type: ignore
            model="saaras:v3",
            vad_signals=True,
            language=Language.TE_IN,
        ),
    )


    
    tts = SarvamTTSService(
        api_key=os.getenv("SARVAM_API_KEY", ""),
        settings=SarvamTTSService.Settings(
            voice="shubh",
            model="bulbul:v3",
            temperature=0.9,
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
            audio_in_filter=RNNoiseFilter(),
        ),
    )

    await run_bot(
        transport,
        runner_args.handle_sigint,
        customer_name=customer_name,
        transcript_out=transcript_out,
    )
