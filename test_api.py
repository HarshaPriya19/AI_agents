import json
from typing import Any, Dict

from app import create_app


def _validate_meeting_report_structure(payload: Dict[str, Any]) -> None:
    assert "summary" in payload and isinstance(payload["summary"], str)
    assert "discussion_points" in payload and isinstance(payload["discussion_points"], list)
    assert "decisions" in payload and isinstance(payload["decisions"], list)
    assert "action_items" in payload and isinstance(payload["action_items"], list)

    for item in payload["action_items"]:
        assert isinstance(item, dict)
        assert "task" in item and isinstance(item["task"], str)
        assert "owner" in item and isinstance(item["owner"], str)
        assert "deadline" in item and isinstance(item["deadline"], str)


def run_smoke_test() -> None:
    """
    Simple smoke test runner that can be executed via:
        python test_api.py

    It uses Flask's test client, so no separate server process is required.
    """
    app = create_app()
    client = app.test_client()

    with open("sample_transcript.txt", "r", encoding="utf-8") as f:
        transcript = f.read()

    response = client.post(
        "/process_transcript",
        data=json.dumps({"transcript": transcript}),
        content_type="application/json",
    )

    print(f"Status code: {response.status_code}")
    assert response.status_code == 200, response.data

    data = response.get_json()
    assert isinstance(data, dict)
    assert "meeting_report" in data
    assert "topic_segments" in data

    meeting_report = data["meeting_report"]
    _validate_meeting_report_structure(meeting_report)

    topic_segments = data["topic_segments"]
    assert isinstance(topic_segments, list)
    if topic_segments:
        seg = topic_segments[0]
        assert "start_timestamp" in seg
        assert "end_timestamp" in seg
        assert "topic_summary" in seg

    print("API response structure is valid.")


if __name__ == "__main__":
    run_smoke_test()

