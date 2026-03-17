from dataclasses import dataclass
from typing import List
import re


@dataclass
class TopicSegment:
    start_timestamp: str
    end_timestamp: str
    topic_summary: str
    raw_block: str


TIMESTAMP_PATTERN = re.compile(
    r"^\s*(?:\[(?P<t1>\d{1,2}:\d{2}(?::\d{2})?)\]|(?P<t2>\d{1,2}:\d{2}(?::\d{2})?))\s*-?\s*(?P<rest>.*)$"
)


def _normalize_timestamp(ts: str | None) -> str:
    """
    Normalize timestamps to mm:ss or hh:mm:ss-like strings for consistency.
    If none is provided, we return a synthetic '00:00' which can later be
    replaced or enriched by upstream tooling.
    """
    if not ts:
        return "00:00"
    return ts.strip()


def segment_transcript_by_topic(transcript: str) -> List[TopicSegment]:
    """
    Divide the transcript into logical topic blocks using simple heuristics:

    - If a line starts with a timestamp like `[00:01:23]` or `00:01`, it is
      treated as a new potential segment boundary.
    - Otherwise, long pauses between timestamps are approximated by line count.
    - Each segment is summarized very roughly using the first non-empty line.

    This function is intentionally lightweight and does not call any LLMs.
    It is meant to be a pre-processing / helper function that can be
    composed with the main Meeting Intelligence pipeline.
    """
    if not transcript or not transcript.strip():
        return []

    lines = transcript.splitlines()

    segments: List[TopicSegment] = []
    current_block_lines: List[str] = []
    current_start_ts: str | None = None

    for line in lines:
        match = TIMESTAMP_PATTERN.match(line)
        if match:
            # Flush any existing block before starting a new one.
            if current_block_lines:
                raw_block = "\n".join(current_block_lines).strip()
                if raw_block:
                    topic_summary = _summarize_block(raw_block)
                    segments.append(
                        TopicSegment(
                            start_timestamp=_normalize_timestamp(current_start_ts),
                            end_timestamp=_normalize_timestamp(match.group("t1") or match.group("t2")),
                            topic_summary=topic_summary,
                            raw_block=raw_block,
                        )
                    )

            current_block_lines = [match.group("rest").strip()]
            current_start_ts = match.group("t1") or match.group("t2")
        else:
            current_block_lines.append(line)

    # Flush the last block, using the last known timestamp as both start and end.
    if current_block_lines:
        raw_block = "\n".join(current_block_lines).strip()
        if raw_block:
            topic_summary = _summarize_block(raw_block)
            segments.append(
                TopicSegment(
                    start_timestamp=_normalize_timestamp(current_start_ts),
                    end_timestamp=_normalize_timestamp(current_start_ts),
                    topic_summary=topic_summary,
                    raw_block=raw_block,
                )
            )

    return segments


def _summarize_block(block: str) -> str:
    """
    Very small heuristic summarizer: uses the first informative line
    as a 'topic title'. This keeps the function deterministic and
    avoids extra model calls.
    """
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Remove common speaker prefixes like "Alice:" or "Bob -"
        stripped = re.sub(r"^[A-Za-z0-9_\- ]{1,40}:\s*", "", stripped)
        return stripped[:200]
    return "General discussion"

