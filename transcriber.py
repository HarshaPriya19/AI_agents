from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import whisper
from pyannote.audio import Pipeline

from segmentation import segment_transcript_by_topic, TopicSegment


@dataclass
class SpeakerSegment:
    """
    Lightweight container for a single speaker-attributed utterance.
    """

    start: float
    end: float
    speaker: str
    text: str


@dataclass
class TranscriptionResult:
    """
    Combined result of ASR + diarization + lightweight topic segmentation.
    """

    attributed_transcript: str
    speaker_segments: List[SpeakerSegment]
    topic_segments: List[TopicSegment]
    diarization_status: str = "enabled"
    diarization_error: str | None = None


_WHISPER_MODEL: Optional[whisper.Whisper] = None
_DIARIZATION_PIPELINE: Optional[Pipeline] = None


def _load_whisper_model(model_name: str = "base") -> whisper.Whisper:
    """
    Lazily load the Whisper model once per process.
    """
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        _WHISPER_MODEL = whisper.load_model(model_name)
    return _WHISPER_MODEL


def _load_diarization_pipeline() -> Pipeline:
    """
    Lazily load the pyannote.audio diarization pipeline.

    This expects a valid Hugging Face token in the PYANNOTE_AUTH_TOKEN
    environment variable. Refer to pyannote.audio documentation for how
    to obtain a token and accept the model license.
    """
    global _DIARIZATION_PIPELINE
    if _DIARIZATION_PIPELINE is None:
        auth_token = os.getenv("PYANNOTE_AUTH_TOKEN")
        if not auth_token:
            raise RuntimeError(
                "PYANNOTE_AUTH_TOKEN is not set. "
                "Set it to a valid Hugging Face token to enable speaker diarization."
            )

        # pyannote.audio has changed its auth parameter name across versions.
        # Try the most common signatures in order for compatibility.
        # Note: pyannote diarization models are often "gated" on Hugging Face.
        # You must accept the model's license terms on the HF page for your account
        # and ensure the token has access. You can override the model id via env:
        #   PYANNOTE_DIARIZATION_MODEL=pyannote/speaker-diarization-3.1
        model_id = os.getenv("PYANNOTE_DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
        last_exc: Exception | None = None
        for kwargs in (
            {"token": auth_token},
            {"hf_token": auth_token},
            {"use_auth_token": auth_token},
        ):
            try:
                _DIARIZATION_PIPELINE = Pipeline.from_pretrained(model_id, **kwargs)
                break
            except TypeError as exc:
                last_exc = exc
                _DIARIZATION_PIPELINE = None

        if _DIARIZATION_PIPELINE is None:
            raise RuntimeError(
                "Failed to load pyannote diarization pipeline. "
                "Common causes:\n"
                "- The Hugging Face repo is gated and your account has not accepted the model license.\n"
                "- The token does not have access to the gated repo.\n"
                "- pyannote.audio version mismatch.\n\n"
                f"Model: {model_id}\n"
                f"Last error: {last_exc}"
            )
    return _DIARIZATION_PIPELINE


def _format_timestamp(seconds: float) -> str:
    """
    Format a float number of seconds as hh:mm:ss or mm:ss.
    """
    if seconds < 0:
        seconds = 0.0

    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _normalize_speaker_label(raw_label: str) -> str:
    """
    Map pyannote labels like 'SPEAKER_00' to 'Speaker 0'.
    """
    raw = raw_label.strip()
    if raw.upper().startswith("SPEAKER_"):
        try:
            idx = int(raw.split("_", maxsplit=1)[1])
            return f"Speaker {idx}"
        except (IndexError, ValueError):
            pass
    return raw


def _build_speaker_segments(
    whisper_segments: List[dict],
    diarization_annotation,
) -> List[SpeakerSegment]:
    """
    Align Whisper time-coded segments with diarization turns
    by assigning each Whisper segment to the speaker active
    at its midpoint.
    """
    speaker_segments: List[SpeakerSegment] = []

    # Pre-materialize diarization tracks for simple midpoint lookup.
    diarization_tracks: List[Tuple[Tuple[float, float], str]] = []
    for segment, _, label in diarization_annotation.itertracks(yield_label=True):
        diarization_tracks.append(((segment.start, segment.end), str(label)))

    for seg in whisper_segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        midpoint = (start + end) / 2.0
        assigned_label: Optional[str] = None
        for (seg_start, seg_end), label in diarization_tracks:
            if seg_start <= midpoint <= seg_end:
                assigned_label = label
                break

        speaker = _normalize_speaker_label(assigned_label or "Speaker ?")
        speaker_segments.append(
            SpeakerSegment(
                start=start,
                end=end,
                speaker=speaker,
                text=text,
            )
        )

    return speaker_segments


def _build_attributed_transcript(speaker_segments: List[SpeakerSegment]) -> str:
    """
    Create a single attributed transcript string where each line starts with
    a timestamp and speaker label, e.g.:

    [00:05] Speaker 1: I think we should approve the budget.

    This format is intentionally compatible with the simple topic segmentation
    heuristics in segmentation.segment_transcript_by_topic (which looks for
    timestamp prefixes).
    """
    lines: List[str] = []
    for seg in speaker_segments:
        ts = _format_timestamp(seg.start)
        line = f"[{ts}] {seg.speaker}: {seg.text}"
        lines.append(line)
    return "\n".join(lines)


def transcribe_and_diarize(audio_path: str, whisper_model: str = "base") -> TranscriptionResult:
    """
    High-level entry point used by the Flask API:

    - Runs Whisper locally for speech-to-text with timestamps.
    - Runs pyannote.audio for speaker diarization.
    - Aligns both to produce speaker-attributed segments.
    - Builds a single attributed transcript string that preserves
      conversational nuance for downstream LLM processing.
    - Segments that transcript into lightweight topic blocks with timestamps.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    ffmpeg_available = shutil.which("ffmpeg") is not None

    model = _load_whisper_model(whisper_model)

    diarization_enabled = (os.getenv("DIARIZATION_ENABLED") or "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    diarization_strict = (os.getenv("DIARIZATION_STRICT") or "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    # 1) Local ASR with Whisper.
    # If ffmpeg is not available (common on Windows hackathon machines),
    # we can still transcribe WAV files by loading them with soundfile and
    # passing the waveform directly to Whisper.
    if ffmpeg_available:
        asr_result = model.transcribe(
            audio_path,
            verbose=False,
            word_timestamps=False,
        )
    else:
        _, ext = os.path.splitext(audio_path.lower())
        if ext != ".wav":
            raise RuntimeError(
                "FFmpeg is not installed, so only .wav files can be processed right now. "
                "Please upload a .wav file (PCM) or install FFmpeg to support .mp3/.m4a."
            )

        audio_np, sr = sf.read(audio_path, always_2d=True)
        # Convert (time, channels) -> mono float32
        mono = audio_np.mean(axis=1).astype("float32", copy=False)

        # Whisper expects 16kHz audio. Resample if needed using linear interpolation.
        target_sr = 16000
        if int(sr) != target_sr:
            duration = mono.shape[0] / float(sr)
            old_t = np.linspace(0.0, duration, num=mono.shape[0], endpoint=False)
            new_len = int(round(duration * target_sr))
            new_t = np.linspace(0.0, duration, num=new_len, endpoint=False)
            mono = np.interp(new_t, old_t, mono).astype("float32", copy=False)

        asr_result = model.transcribe(
            mono,
            verbose=False,
            word_timestamps=False,
        )
    whisper_segments: List[dict] = asr_result.get("segments", []) or []

    diarization_status = "disabled"
    diarization_error: str | None = None

    speaker_segments: List[SpeakerSegment]
    if diarization_enabled:
        try:
            diarization_pipeline = _load_diarization_pipeline()

            # 2) Speaker diarization with pyannote.
            # On some Windows setups, pyannote's optional torchcodec-based decoding warns/fails.
            # To make this robust, we preload audio ourselves and pass in-memory waveform.
            waveform_np, sample_rate = sf.read(audio_path, always_2d=True)
            # soundfile returns shape (time, channels); pyannote expects (channels, time)
            waveform = torch.from_numpy(np.transpose(waveform_np).astype("float32", copy=False))
            diarization = diarization_pipeline({"waveform": waveform, "sample_rate": int(sample_rate)})

            # 3) Align ASR segments with diarization.
            speaker_segments = _build_speaker_segments(whisper_segments, diarization)
            diarization_status = "enabled"
        except Exception as exc:  # noqa: BLE001 - fallback for demo resiliency
            diarization_error = str(exc)
            if diarization_strict:
                raise
            # Fallback: keep the app working even if HF gating/network blocks diarization.
            diarization_status = "fallback_single_speaker"
            speaker_segments = [
                SpeakerSegment(
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", float(seg.get("start", 0.0)))),
                    speaker="Speaker 0",
                    text=(seg.get("text") or "").strip(),
                )
                for seg in whisper_segments
                if (seg.get("text") or "").strip()
            ]
    else:
        # Explicitly disabled diarization: still produce a deterministic transcript.
        speaker_segments = [
            SpeakerSegment(
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", float(seg.get("start", 0.0)))),
                speaker="Speaker 0",
                text=(seg.get("text") or "").strip(),
            )
            for seg in whisper_segments
            if (seg.get("text") or "").strip()
        ]

    # 4) Build an attributed transcript compatible with Phase 1 pipeline.
    attributed_transcript = _build_attributed_transcript(speaker_segments)

    # 5) Topic-level segmentation using existing heuristics.
    topic_segments = segment_transcript_by_topic(attributed_transcript)

    return TranscriptionResult(
        attributed_transcript=attributed_transcript,
        speaker_segments=speaker_segments,
        topic_segments=topic_segments,
        diarization_status=diarization_status,
        diarization_error=diarization_error,
    )

