import os
import tempfile
import warnings

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from werkzeug.utils import secure_filename

# pyannote.audio may emit very verbose torchcodec/FFmpeg warnings on Windows
# during import. These are non-fatal for our in-memory audio path, but they can
# obscure Flask's startup URL in the console.
warnings.filterwarnings(
    "ignore",
    message=r".*torchcodec is not installed correctly.*",
    category=UserWarning,
)

from llm_client import (
    extract_meeting_report,
    review_and_fix_meeting_report,
    answer_meeting_question,
)
from segmentation import segment_transcript_by_topic
from transcriber import transcribe_and_diarize
from meeting_models import MeetingReport
from workflows import (
    send_telegram_summary,
    send_email_draft,
    create_calendar_reminder,
)


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", static_url_path="")
    CORS(app)  # Enable CORS for all routes and origins by default.

    @app.route("/health", methods=["GET"])
    def health() -> tuple[dict, int]:
        return {"status": "ok"}, 200

    @app.route("/")
    def index() -> "tuple[str, int] | str":
        """
        Serve the main dashboard HTML.
        """
        return send_from_directory(app.static_folder, "index.html")

    @app.route("/auth")
    def auth() -> "tuple[str, int] | str":
        """
        Lightweight sign-in page (client-side only).
        """
        return send_from_directory(app.static_folder, "auth.html")

    @app.route("/process_audio", methods=["POST"])
    def process_audio():
        """
        Accept an audio file (e.g., .mp3 or .wav), run local ASR + diarization,
        then feed the speaker-attributed transcript into the existing
        Meeting Intelligence pipeline.

        The response includes:
        - meeting_report: strict MeetingReport JSON (Phase 1)
        - topic_segments: lightweight topic blocks with timestamps
        - speaker_attributed_transcript: full text with speaker labels
        """
        if "file" not in request.files:
            return jsonify({"error": "No file part in request under field name 'file'."}), 400

        file_storage = request.files["file"]
        if not file_storage or file_storage.filename == "":
            return jsonify({"error": "No selected file."}), 400

        # Optional delivery email captured from the /auth page.
        # This is sent as multipart form field "user_email".
        user_email = (request.form.get("user_email") or "").strip()

        filename = secure_filename(file_storage.filename)
        _, ext = os.path.splitext(filename.lower())
        if ext not in {".mp3", ".wav", ".m4a", ".flac"}:
            return jsonify({"error": "Unsupported audio format. Use .mp3, .wav, .m4a, or .flac."}), 400

        # Persist the uploaded file to a temporary location for the underlying
        # audio libraries to consume.
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            temp_path = tmp.name
            file_storage.save(temp_path)

        try:
            transcription_result = transcribe_and_diarize(temp_path)
            attributed_transcript = transcription_result.attributed_transcript

            # Run the existing LLM-based Meeting Intelligence extraction on the
            # speaker-attributed transcript to preserve conversational nuance.
            report = extract_meeting_report(attributed_transcript)

            # We already computed topic_segments inside transcriber.py to ensure
            # alignment with timestamps; reuse that here rather than recomputing.
            segments_payload = [
                {
                    "start_timestamp": seg.start_timestamp,
                    "end_timestamp": seg.end_timestamp,
                    "topic_summary": seg.topic_summary,
                }
                for seg in transcription_result.topic_segments
            ]

            # Optional: auto-send email right after processing.
            auto_email_enabled = (os.getenv("AUTO_EMAIL_ON_PROCESS_AUDIO") or "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            email_result = None
            if auto_email_enabled:
                # If user_email is provided, send to that address; otherwise fallback
                # to EMAIL_TO env or the default in workflows.py.
                email_result_obj = send_email_draft(report, to_address=user_email or None)
                email_result = {
                    "success": email_result_obj.success,
                    "details": email_result_obj.details,
                }

            return (
                jsonify(
                    {
                        "meeting_report": report.model_dump(),
                        "topic_segments": segments_payload,
                        "speaker_attributed_transcript": attributed_transcript,
                        "diarization_status": transcription_result.diarization_status,
                        "diarization_error": transcription_result.diarization_error,
                        "auto_email_result": email_result,
                    }
                ),
                200,
            )
        except Exception as exc:  # noqa: BLE001 - surface errors in JSON for debugging
            return jsonify({"error": str(exc)}), 500
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                # Best-effort cleanup; ignore failures.
                pass

    @app.route("/process_transcript", methods=["POST"])
    def process_transcript():
        """
        Accept raw text, run the Meeting Intelligence extraction pipeline,
        and return a strict JSON MeetingReport plus lightweight topic segments.
        """
        data = request.get_json(silent=True) or {}
        transcript = data.get("transcript") or data.get("text") or ""

        if not isinstance(transcript, str) or not transcript.strip():
            return jsonify({"error": "Field 'transcript' (non-empty string) is required."}), 400

        try:
            report = extract_meeting_report(transcript)
        except Exception as exc:  # noqa: BLE001 - surface errors in JSON for debugging
            return jsonify({"error": str(exc)}), 500

        # Topic segmentation is not part of the strict MeetingReport schema
        # but is useful for downstream UI/analytics, so we add it as an
        # auxiliary field in the API response.
        segments = segment_transcript_by_topic(transcript)
        segments_payload = [
            {
                "start_timestamp": seg.start_timestamp,
                "end_timestamp": seg.end_timestamp,
                "topic_summary": seg.topic_summary,
            }
            for seg in segments
        ]

        return (
            jsonify(
                {
                    "meeting_report": report.model_dump(),
                    "topic_segments": segments_payload,
                }
            ),
            200,
        )

    @app.route("/execute_workflow", methods=["POST"])
    def execute_workflow():
        """
        Execute an automation workflow based on a MeetingReport and target.

        Expected JSON payload:
        {
            "meeting_report": { ... MeetingReport JSON ... },
            "target": "telegram" | "email" | "calendar",
            "speaker_attributed_transcript": "..."  // required for reviewer agent
        }

        Flow:
        - Validate and construct MeetingReport from JSON.
        - Run Reviewer Agent to verify/fix action item owners/deadlines
          based on the actual speaker-attributed transcript.
        - Trigger the chosen workflow using the corrected MeetingReport.
        """
        data = request.get_json(silent=True) or {}
        target = (data.get("target") or "").lower()
        report_payload = data.get("meeting_report") or {}
        transcript = data.get("speaker_attributed_transcript") or ""

        if target not in {"telegram", "email", "calendar"}:
            return jsonify({"error": "Invalid 'target'. Use 'telegram', 'email', or 'calendar'."}), 400

        if not transcript or not isinstance(transcript, str):
            return jsonify({"error": "'speaker_attributed_transcript' (non-empty string) is required."}), 400

        try:
            report = MeetingReport.model_validate(report_payload)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"Invalid MeetingReport payload: {exc}"}), 400

        try:
            # Reviewer Agent: correct action items before executing workflows.
            reviewed_report = review_and_fix_meeting_report(transcript, report)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"Reviewer agent failed: {exc}"}), 500

        try:
            if target == "telegram":
                result = send_telegram_summary(reviewed_report)
            elif target == "email":
                result = send_email_draft(reviewed_report)
            else:  # calendar
                result = create_calendar_reminder(reviewed_report)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"Workflow execution failed: {exc}"}), 500

        return jsonify(
            {
                "target": target,
                "meeting_report": reviewed_report.model_dump(),
                "workflow_result": {
                    "success": result.success,
                    "details": result.details,
                },
            }
        ), 200

    @app.route("/query_meeting", methods=["POST"])
    def query_meeting():
        """
        Lightweight QA endpoint used by the dashboard's query bar.

        Expects JSON payload:
        {
            "question": "...",
            "meeting_report": { ... },
            "speaker_attributed_transcript": "...",
            "topic_segments": [ ... ]  // optional
        }
        """
        data = request.get_json(silent=True) or {}
        question = (data.get("question") or "").strip()
        meeting_report = data.get("meeting_report") or {}
        transcript = data.get("speaker_attributed_transcript") or ""
        topic_segments = data.get("topic_segments") or []

        if not question:
            return jsonify({"error": "Field 'question' (non-empty string) is required."}), 400

        if not isinstance(transcript, str) or not transcript.strip():
            return jsonify(
                {"error": "'speaker_attributed_transcript' (non-empty string) is required."}
            ), 400

        try:
            answer = answer_meeting_question(
                question=question,
                meeting_report=meeting_report,
                transcript=transcript,
                topic_segments=topic_segments,
            )
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc)}), 500

        return jsonify({"answer": answer}), 200

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(host="0.0.0.0", port=5000, debug=True)
