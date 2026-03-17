"""
Small helper script to download a ~30 second sample audio clip
for local testing of the transcription + diarization pipeline.

Usage (from the project root):

    python download_sample_audio.py

By default this saves the file as `sample_meeting.wav` in the
current directory. You can change the URL below to point to any
short, two-speaker meeting-style recording you prefer.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Final

import numpy as np
import soundfile as sf

import requests

try:
    import pyttsx3
except Exception:  # noqa: BLE001
    pyttsx3 = None  # type: ignore[assignment]


# Optional network sources (best-effort). Some networks block these hosts.
SPEECH_WAV_URLS: Final[list[str]] = [
    "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav",
    "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0011_8k.wav",
]

OUTPUT_FILENAME: Final[str] = "sample_meeting.wav"
CLIP_SECONDS: Final[int] = 30
TARGET_SR: Final[int] = 16000


def main() -> None:
    out_path = pathlib.Path(OUTPUT_FILENAME).resolve()
    tmp1 = out_path.with_suffix(".part1.wav")
    tmp2 = out_path.with_suffix(".part2.wav")
    print(f"Preparing sample audio at {out_path} ...")

    try:
        # Always-works offline fallback: generate a synthetic "speech-like" sample instantly.
        # This is not real speech, but it exercises the full upload -> pipeline -> email path.
        def _synth_voice(seed: int, seconds: float) -> np.ndarray:
            rng = np.random.default_rng(seed)
            n = int(seconds * TARGET_SR)
            t = np.arange(n, dtype=np.float32) / float(TARGET_SR)
            # Build a formant-ish signal (sum of 3 sines) with slowly varying frequency.
            base = 120.0 + 30.0 * np.sin(2 * np.pi * 0.8 * t) + float(rng.uniform(-10, 10))
            f1 = base
            f2 = base * 2.0 + 50.0
            f3 = base * 3.1 + 90.0
            phase1 = 2 * np.pi * np.cumsum(f1) / TARGET_SR
            phase2 = 2 * np.pi * np.cumsum(f2) / TARGET_SR
            phase3 = 2 * np.pi * np.cumsum(f3) / TARGET_SR
            sig = (
                0.55 * np.sin(phase1)
                + 0.25 * np.sin(phase2)
                + 0.15 * np.sin(phase3)
            ).astype("float32", copy=False)
            # Envelope to mimic syllables.
            env = (0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t)) ** 2
            noise = 0.02 * rng.standard_normal(n).astype("float32", copy=False)
            out = (sig * env + noise).astype("float32", copy=False)
            out /= max(1e-6, float(np.max(np.abs(out))))
            return out

        def _build_synthetic_sample() -> None:
            seg_len_s = 6.0
            pause = np.zeros(int(0.3 * TARGET_SR), dtype="float32")
            v1 = _synth_voice(1, seg_len_s)
            v2 = _synth_voice(2, seg_len_s)
            blocks: list[np.ndarray] = []
            total_needed = int(CLIP_SECONDS * TARGET_SR)
            while sum(b.shape[0] for b in blocks) < total_needed:
                blocks.append(v1)
                blocks.append(pause)
                blocks.append(v2)
                blocks.append(pause)
            out = np.concatenate(blocks)[:total_needed]
            sf.write(str(out_path), out, TARGET_SR)

        # Preferred approach (no network, no ffmpeg): generate two voices locally using system TTS.
        if pyttsx3 is not None:
            try:
                print("Generating local two-voice sample using system TTS ...")
                engine = pyttsx3.init()
                voices = engine.getProperty("voices") or []
                if len(voices) >= 2:
                    voice_a = voices[0].id
                    voice_b = voices[1].id
                elif len(voices) == 1:
                    voice_a = voices[0].id
                    voice_b = voices[0].id
                else:
                    raise RuntimeError("No TTS voices available on this machine.")

                a_wav = out_path.with_suffix(".tts_a.wav")
                b_wav = out_path.with_suffix(".tts_b.wav")

                text_a = (
                    "Hi, this is speaker one. Let's review the budget and decide next steps. "
                    "I can send the updated spreadsheet by Friday."
                )
                text_b = (
                    "Thanks. This is speaker two. I agree with approving the budget. "
                    "Please schedule a follow up meeting next week to confirm timelines."
                )

                engine.setProperty("voice", voice_a)
                engine.save_to_file(text_a, str(a_wav))
                engine.runAndWait()

                engine.setProperty("voice", voice_b)
                engine.save_to_file(text_b, str(b_wav))
                engine.runAndWait()

                a1, sr1 = sf.read(str(a_wav), always_2d=True)
                a2, sr2 = sf.read(str(b_wav), always_2d=True)

                a_wav.unlink(missing_ok=True)
                b_wav.unlink(missing_ok=True)

                def _to_mono_float32(x: np.ndarray) -> np.ndarray:
                    return x.mean(axis=1).astype("float32", copy=False)

                def _resample_linear(mono: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
                    if int(sr) == int(target_sr):
                        return mono
                    duration = mono.shape[0] / float(sr)
                    old_t = np.linspace(0.0, duration, num=mono.shape[0], endpoint=False)
                    new_len = int(round(duration * target_sr))
                    new_t = np.linspace(0.0, duration, num=new_len, endpoint=False)
                    return np.interp(new_t, old_t, mono).astype("float32", copy=False)

                m1 = _resample_linear(_to_mono_float32(a1), int(sr1), TARGET_SR)
                m2 = _resample_linear(_to_mono_float32(a2), int(sr2), TARGET_SR)

                seg_len = int(7 * TARGET_SR)
                pause = np.zeros(int(0.25 * TARGET_SR), dtype="float32")

                def _take(x: np.ndarray, n: int) -> np.ndarray:
                    if x.shape[0] >= n:
                        return x[:n]
                    reps = int(np.ceil(n / x.shape[0]))
                    y = np.tile(x, reps)
                    return y[:n]

                blocks: list[np.ndarray] = []
                total_needed = int(CLIP_SECONDS * TARGET_SR)
                while sum(b.shape[0] for b in blocks) < total_needed:
                    blocks.append(_take(m1, seg_len))
                    blocks.append(pause)
                    blocks.append(_take(m2, seg_len))
                    blocks.append(pause)

                out = np.concatenate(blocks)[:total_needed]
                sf.write(str(out_path), out, TARGET_SR)
                print(f"Sample generated. Saved 30s clip to {out_path}")
                return
            except Exception as exc:  # noqa: BLE001
                print(f"Local TTS generation failed ({exc}). Falling back to network download...")

        print(f"Downloading sample audio parts to {tmp1} and {tmp2} ...")

        def _download(url: str, dest: pathlib.Path) -> None:
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Accept": "*/*",
            }
            with requests.get(url, stream=True, timeout=120, headers=headers) as resp:
                resp.raise_for_status()
                with dest.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)

        # Download two different clips (try multiple URLs if some hosts block).
        last_exc: Exception | None = None
        for i, url in enumerate(SPEECH_WAV_URLS):
            try:
                _download(url, tmp1)
                last_exc = None
                start_idx = i + 1
                break
            except Exception as exc:
                last_exc = exc
                if tmp1.exists():
                    tmp1.unlink(missing_ok=True)
        if last_exc is not None:
            raise last_exc

        last_exc = None
        for url in SPEECH_WAV_URLS[start_idx:]:
            try:
                _download(url, tmp2)
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                if tmp2.exists():
                    tmp2.unlink(missing_ok=True)
        if last_exc is not None:
            raise last_exc

        a1, sr1 = sf.read(str(tmp1), always_2d=True)
        a2, sr2 = sf.read(str(tmp2), always_2d=True)

        def _to_mono_float32(x: np.ndarray) -> np.ndarray:
            return x.mean(axis=1).astype("float32", copy=False)

        def _resample_linear(mono: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
            if int(sr) == int(target_sr):
                return mono
            duration = mono.shape[0] / float(sr)
            old_t = np.linspace(0.0, duration, num=mono.shape[0], endpoint=False)
            new_len = int(round(duration * target_sr))
            new_t = np.linspace(0.0, duration, num=new_len, endpoint=False)
            return np.interp(new_t, old_t, mono).astype("float32", copy=False)

        m1 = _resample_linear(_to_mono_float32(a1), int(sr1), TARGET_SR)
        m2 = _resample_linear(_to_mono_float32(a2), int(sr2), TARGET_SR)

        # Build a ~30 second clip by repeating/alternating segments.
        # Pattern: speaker1 6s, pause 0.3s, speaker2 6s, pause, repeat...
        seg_len = int(6 * TARGET_SR)
        pause = np.zeros(int(0.3 * TARGET_SR), dtype="float32")

        def _take(x: np.ndarray, n: int) -> np.ndarray:
            if x.shape[0] >= n:
                return x[:n]
            # repeat if too short
            reps = int(np.ceil(n / x.shape[0]))
            y = np.tile(x, reps)
            return y[:n]

        blocks: list[np.ndarray] = []
        total_needed = int(CLIP_SECONDS * TARGET_SR)
        while sum(b.shape[0] for b in blocks) < total_needed:
            blocks.append(_take(m1, seg_len))
            blocks.append(pause)
            blocks.append(_take(m2, seg_len))
            blocks.append(pause)

        out = np.concatenate(blocks)[:total_needed]
        sf.write(str(out_path), out, TARGET_SR)
        return
    except Exception as exc:  # noqa: BLE001 - simple CLI script
        print(f"Failed to download sample audio: {exc}", file=sys.stderr)
        print("Falling back to offline synthetic sample generation...")
        try:
            _build_synthetic_sample()
            print(f"Sample generated. Saved 30s clip to {out_path}")
            return
        except Exception as exc2:  # noqa: BLE001
            print(f"Offline synthetic generation also failed: {exc2}", file=sys.stderr)
            sys.exit(1)
    finally:
        if tmp1.exists():
            tmp1.unlink(missing_ok=True)
        if tmp2.exists():
            tmp2.unlink(missing_ok=True)

    print(f"Download complete. Saved 30s clip to {out_path}")


if __name__ == "__main__":
    main()

