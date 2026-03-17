import json
import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from meeting_models import MeetingReport


load_dotenv()  # Load GROQ_API_KEY from environment


SYSTEM_PROMPT = """
You are an expert Meeting Intelligence engine.
Your job is to transform a raw meeting transcript into a STRICT JSON object
representing a structured meeting report.

The JSON MUST conform EXACTLY to the following Pydantic schema (no extra keys):

MeetingReport:
- summary: string
- discussion_points: list of strings
- decisions: list of strings
- action_items: list of objects with fields:
    - task: string
    - owner: string
    - deadline: string

=== IMPLICIT INFERENCE RULES (CRITICAL) ===
- Identify tasks even when they are expressed informally, e.g.:
  - "I'll handle that."
  - "Sarah, please check this by Friday."
  - "Can you send the deck later today?"
- For EVERY action item, infer:
  - owner: who is responsible (based on names, pronouns like "I", "you", "we",
    role mentions, or prior context in the conversation).
  - deadline: infer an explicit due date or time frame from phrases like
    "by Friday", "this week", "before the demo", "ASAP".
    If there is truly no clue, use `"no explicit deadline mentioned"`.
- NEVER leave owner or deadline empty. Always infer a best-effort value.

=== OUTPUT RULES ===
- Output ONLY valid JSON. Do NOT wrap it in markdown fences.
- Do NOT include comments in the JSON.
- All strings must be double-quoted.
- If you are unsure about some details, make a reasonable, clearly-labeled best guess.
"""


def _build_llm() -> ChatGroq:
    """
    Construct the Groq-backed LLM client targeting Llama-3-70B.
    Expects GROQ_API_KEY to be set in the environment (loaded via python-dotenv).
    """
    # Groq model IDs can be deprecated; keep it configurable via env var.
    # Current production default (per Groq docs) is "llama-3.3-70b-versatile".
    model_id = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    return ChatGroq(
        model=model_id,
        temperature=0,
        max_tokens=2048,
    )


REVIEWER_PROMPT = """
You are a critical Reviewer Agent for a Meeting Intelligence system.

Your job is to carefully audit and, if necessary, CORRECT an existing MeetingReport
based on the actual speaker-attributed transcript of the meeting.

The MeetingReport JSON MUST always conform EXACTLY to the schema:

MeetingReport:
- summary: string
- discussion_points: list of strings
- decisions: list of strings
- action_items: list of objects with fields:
    - task: string
    - owner: string
    - deadline: string

=== REVIEW TASK (CRITICAL) ===
- Compare each action item with the transcript and check:
  - Did the person who is listed as "owner" actually volunteer or get assigned
    that task in the conversation?
  - Are deadlines consistent with what is said (e.g., "by Friday", "next week")?
- If you find mismatches, FIX the JSON so that:
  - Each task is assigned to the most appropriate person based on who spoke.
  - Deadlines reflect the spoken time frames as closely as possible.
- Preserve all useful information from the original report; only change fields
  that are clearly wrong or missing.

=== OUTPUT RULES ===
- Output ONLY a valid MeetingReport JSON object (no extra keys).
- Do NOT wrap it in markdown fences.
- Do NOT include comments.
- All strings must be double-quoted.
"""


def extract_meeting_report(transcript: str) -> MeetingReport:
    """
    Call the Groq LLM to transform a raw transcript into a MeetingReport.
    """
    if not transcript or not transcript.strip():
        raise ValueError("Transcript is empty.")

    llm = _build_llm()

    messages = [
        SystemMessage(content=SYSTEM_PROMPT.strip()),
        HumanMessage(
            content=(
                "Here is the raw meeting transcript. "
                "Follow all instructions in the system prompt and respond with JSON only.\n\n"
                f"TRANSCRIPT:\n{transcript}"
            )
        ),
    ]

    response = llm.invoke(messages)
    raw_content = response.content if isinstance(response.content, str) else str(response.content)

    try:
        parsed: Dict[str, Any] = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        # Best-effort recovery: try to locate the first and last JSON braces.
        start = raw_content.find("{")
        end = raw_content.rfind("}")
        if start != -1 and end != -1 and start < end:
            try:
                parsed = json.loads(raw_content[start : end + 1])
            except json.JSONDecodeError:
                raise ValueError(f"LLM returned non-JSON output and recovery failed: {exc}") from exc
        else:
            raise ValueError(f"LLM returned non-JSON output: {exc}") from exc

    # Validate and coerce to our strict Pydantic schema.
    return MeetingReport.model_validate(parsed)


def review_and_fix_meeting_report(
    transcript: str,
    report: MeetingReport,
) -> MeetingReport:
    """
    Reviewer Agent:

    Takes the original speaker-attributed transcript and an initial MeetingReport,
    asks the Groq LLM to verify that action items align with who actually spoke,
    and returns a corrected MeetingReport JSON if needed.
    """
    if not transcript or not transcript.strip():
        raise ValueError("Transcript is empty for reviewer agent.")

    llm = _build_llm()

    messages = [
        SystemMessage(content=REVIEWER_PROMPT.strip()),
        HumanMessage(
            content=(
                "Here is the SPEAKER-ATTRIBUTED TRANSCRIPT of the meeting, followed by the "
                "CURRENT MeetingReport JSON. Carefully review and, if needed, correct the "
                "MeetingReport so that action item owners and deadlines are consistent with "
                "what was actually said.\n\n"
                "=== TRANSCRIPT START ===\n"
                f"{transcript}\n"
                "=== TRANSCRIPT END ===\n\n"
                "=== CURRENT MEETINGREPORT JSON START ===\n"
                f"{report.model_dump_json()}\n"
                "=== CURRENT MEETINGREPORT JSON END ===\n\n"
                "Respond ONLY with the corrected MeetingReport JSON."
            )
        ),
    ]

    response = llm.invoke(messages)
    raw_content = response.content if isinstance(response.content, str) else str(response.content)

    try:
        parsed: Dict[str, Any] = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        # Best-effort recovery similar to extract_meeting_report.
        start = raw_content.find("{")
        end = raw_content.rfind("}")
        if start != -1 and end != -1 and start < end:
            try:
                parsed = json.loads(raw_content[start : end + 1])
            except json.JSONDecodeError:
                raise ValueError(f"Reviewer LLM returned non-JSON output and recovery failed: {exc}") from exc
        else:
            raise ValueError(f"Reviewer LLM returned non-JSON output: {exc}") from exc

    return MeetingReport.model_validate(parsed)


def answer_meeting_question(
    question: str,
    meeting_report: dict[str, Any],
    transcript: str,
    topic_segments: list[dict[str, Any]] | None = None,
) -> str:
    """
    Lightweight QA helper for `/query_meeting`.

    Uses the existing Groq client and provides a structured context composed of:
    - MeetingReport JSON (already distilled).
    - Speaker-attributed transcript (for nuance and attribution).
    - Optional topic segments for fast navigation.
    """
    if not question or not question.strip():
        raise ValueError("Question is empty.")

    llm = _build_llm()

    context_json = json.dumps(
        {
            "meeting_report": meeting_report,
            "topic_segments": topic_segments or [],
        },
        ensure_ascii=False,
    )

    messages = [
        SystemMessage(
            content=(
                "You are an assistant that answers precise questions about a single meeting.\n"
                "- Use the structured MeetingReport JSON as your primary source of truth for "
                "summary, decisions, and action items.\n"
                "- Use the speaker-attributed transcript only when extra nuance is required.\n"
                "- Respond concisely in 1–3 sentences with professional, specific language."
            )
        ),
        HumanMessage(
            content=(
                "Here is the structured context (JSON) followed by the raw transcript.\n\n"
                "=== CONTEXT JSON START ===\n"
                f"{context_json}\n"
                "=== CONTEXT JSON END ===\n\n"
                "=== SPEAKER-ATTRIBUTED TRANSCRIPT START ===\n"
                f"{transcript}\n"
                "=== SPEAKER-ATTRIBUTED TRANSCRIPT END ===\n\n"
                f"QUESTION: {question}"
            )
        ),
    ]

    response = llm.invoke(messages)
    raw_content = response.content if isinstance(response.content, str) else str(response.content)
    return raw_content.strip()

