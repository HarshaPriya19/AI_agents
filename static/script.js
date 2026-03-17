// Call the backend on the same host/port serving this page.
// This avoids "Failed to fetch" when accessing the dashboard via LAN IP.
const API_BASE = window.location.origin;

// Hackathon demo mode:
// When enabled, clicking "Process Audio" will NOT call the backend.
// Instead it will populate the dashboard with realistic example data.
const DEMO_MODE = true;

let selectedFile = null;
let latestMeetingReport = null;
let latestSpeakerTranscript = null;
let latestTopicSegments = null;

const audioInput = document.getElementById("audio-input");
const processBtn = document.getElementById("process-audio-btn");
const loader = document.getElementById("loader");

const meetingTitleEl = document.getElementById("meeting-title");
const meetingMetaEl = document.getElementById("meeting-meta");
const summaryTextEl = document.getElementById("summary-text");
const actionItemsBodyEl = document.getElementById("action-items-body");
const timelineEl = document.getElementById("timeline");
const transcriptEl = document.getElementById("transcript");
const queryForm = document.getElementById("query-form");
const queryInput = document.getElementById("query-input");
const queryAnswerEl = document.getElementById("query-answer");

function setLoading(isLoading) {
  if (isLoading) {
    loader.classList.remove("hidden");
    processBtn.disabled = true;
  } else {
    loader.classList.add("hidden");
    processBtn.disabled = !selectedFile;
  }
}

audioInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) {
    selectedFile = null;
    processBtn.disabled = true;
    meetingMetaEl.textContent = "Upload an audio file to begin.";
    return;
  }
  selectedFile = file;
  processBtn.disabled = false;
  meetingMetaEl.textContent = `Selected file: ${file.name}`;
});

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function loadDemoData() {
  const now = new Date();
  const dateStr = now.toLocaleString();

  latestMeetingReport = {
    summary:
      "The team reviewed Q2 budget proposals, clarified ownership for the launch readiness checklist, and aligned on a revised demo timeline. Risks were identified around vendor onboarding and documentation, and next steps were assigned with clear deadlines.",
    discussion_points: [
      "Budget approval criteria and contingency plan",
      "Demo timeline and stakeholder expectations",
      "Vendor onboarding status and access provisioning",
      "Documentation gaps for the new workflow",
    ],
    decisions: [
      "Approved the Q2 budget with a 5% contingency buffer.",
      "Moved the customer demo to next Friday to allow time for QA and documentation updates.",
    ],
    action_items: [
      {
        task: "Send the updated budget spreadsheet and contingency breakdown to leadership.",
        owner: "Harsha",
        deadline: "EOD today",
      },
      {
        task: "Schedule a 30-minute follow-up sync to confirm the revised demo agenda and roles.",
        owner: "Speaker 1",
        deadline: "by Wednesday",
      },
      {
        task: "Finish vendor onboarding checklist and confirm access to the analytics dashboard.",
        owner: "Speaker 0",
        deadline: "by Thursday",
      },
      {
        task: "Update the runbook and add missing steps for the new transcription workflow.",
        owner: "Speaker 2",
        deadline: "before the demo (next Friday)",
      },
    ],
  };

  latestTopicSegments = [
    {
      start_timestamp: "00:00",
      end_timestamp: "02:10",
      topic_summary: "Budget overview, constraints, and approval criteria.",
    },
    {
      start_timestamp: "02:10",
      end_timestamp: "05:05",
      topic_summary: "Decision: approve budget with contingency and assign spreadsheet follow-up.",
    },
    {
      start_timestamp: "05:05",
      end_timestamp: "08:20",
      topic_summary: "Demo timeline risks, dependencies, and revised date.",
    },
    {
      start_timestamp: "08:20",
      end_timestamp: "10:45",
      topic_summary: "Vendor onboarding + documentation gaps; action items and deadlines.",
    },
  ];

  latestSpeakerTranscript = [
    "[00:00] Speaker 0: Let's start with the Q2 budget. The main concern is keeping a contingency for vendor costs.",
    "[00:22] Speaker 1: Agreed. I think we can approve it if we keep a 5% buffer and track spend weekly.",
    "[01:05] Speaker 0: Great. I'll send the updated spreadsheet to leadership by end of day.",
    "[02:10] Speaker 2: On the demo timeline, QA needs two extra days. I'd recommend moving to next Friday.",
    "[03:00] Speaker 1: That works. I'll schedule a follow-up sync by Wednesday to align on the agenda.",
    "[04:05] Speaker 0: Vendor onboarding is still pending analytics access. I'll confirm permissions by Thursday.",
    "[05:15] Speaker 2: I'll update the runbook and fill the documentation gaps before the demo.",
  ].join("\n");

  return { dateStr };
}

processBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  if (DEMO_MODE) {
    setLoading(true);
    queryAnswerEl.textContent = "";
    try {
      // Simulate realistic processing delay for presentation.
      await sleep(1200);
      const { dateStr } = loadDemoData();

      meetingTitleEl.textContent = "Team Weekly Sync (Demo)";
      meetingMetaEl.textContent = `${dateStr} • Demo Mode`;

      updateDashboardFromResponse();
      return;
    } finally {
      setLoading(false);
    }
  }

  const formData = new FormData();
  formData.append("file", selectedFile);
  // If the user signed in on /auth, include their email so the backend can
  // deliver the summary automatically without extra clicks.
  const userEmail = (localStorage.getItem("mi_user_email") || "").trim();
  if (userEmail) {
    formData.append("user_email", userEmail);
  }

  setLoading(true);
  queryAnswerEl.textContent = "";

  try {
    const res = await fetch(`${API_BASE}/process_audio`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || "Failed to process audio.");
    }

    const data = await res.json();
    latestMeetingReport = data.meeting_report || null;
    latestSpeakerTranscript = data.speaker_attributed_transcript || "";
    latestTopicSegments = data.topic_segments || [];

    updateDashboardFromResponse();

    // If the backend attempted auto-email, surface a minimal status to the user.
    if (data.auto_email_result) {
      const ok = !!data.auto_email_result.success;
      if (!ok) {
        console.warn("Auto-email did not send:", data.auto_email_result);
      }
    }
  } catch (err) {
    console.error(err);
    alert(`Error processing audio: ${err.message}`);
  } finally {
    setLoading(false);
  }
});

function updateDashboardFromResponse() {
  const now = new Date();
  meetingTitleEl.textContent = "AI Meeting Intelligence";
  meetingMetaEl.textContent = now.toLocaleString();

  if (latestMeetingReport && latestMeetingReport.summary) {
    summaryTextEl.classList.remove("empty-state");
    summaryTextEl.textContent = latestMeetingReport.summary;
  } else {
    summaryTextEl.classList.add("empty-state");
    summaryTextEl.textContent =
      "No summary available. The MeetingReport JSON is missing the summary field.";
  }

  renderActionItems();
  renderTimeline();
  renderTranscript();
}

function renderActionItems() {
  while (actionItemsBodyEl.firstChild) {
    actionItemsBodyEl.removeChild(actionItemsBodyEl.firstChild);
  }

  const items = latestMeetingReport?.action_items || [];

  if (!items.length) {
    const row = document.createElement("tr");
    row.className = "empty-row";
    const cell = document.createElement("td");
    cell.colSpan = 5;
    cell.textContent = "No action items identified.";
    row.appendChild(cell);
    actionItemsBodyEl.appendChild(row);
    return;
  }

  items.forEach((item, idx) => {
    const tr = document.createElement("tr");

    const idxTd = document.createElement("td");
    idxTd.textContent = String(idx + 1);

    const taskTd = document.createElement("td");
    taskTd.textContent = item.task || "";

    const ownerTd = document.createElement("td");
    ownerTd.textContent = item.owner || "";

    const deadlineTd = document.createElement("td");
    deadlineTd.textContent = item.deadline || "";

    const wfTd = document.createElement("td");
    const btnContainer = document.createElement("div");
    btnContainer.className = "action-workflow-buttons";

    const telegramBtn = document.createElement("button");
    telegramBtn.className = "icon-btn telegram";
    telegramBtn.title = "Send to Telegram";
    telegramBtn.innerHTML = '<i class="fa-solid fa-paper-plane"></i>';
    telegramBtn.addEventListener("click", () =>
      triggerWorkflow("telegram")
    );

    const emailBtn = document.createElement("button");
    emailBtn.className = "icon-btn email";
    emailBtn.title = "Draft Email";
    emailBtn.innerHTML = '<i class="fa-solid fa-envelope-open-text"></i>';
    emailBtn.addEventListener("click", () => triggerWorkflow("email"));

    const calendarBtn = document.createElement("button");
    calendarBtn.className = "icon-btn calendar";
    calendarBtn.title = "Create Calendar Reminder";
    calendarBtn.innerHTML = '<i class="fa-solid fa-calendar-plus"></i>';
    calendarBtn.addEventListener("click", () =>
      triggerWorkflow("calendar")
    );

    btnContainer.appendChild(telegramBtn);
    btnContainer.appendChild(emailBtn);
    btnContainer.appendChild(calendarBtn);

    wfTd.appendChild(btnContainer);

    tr.appendChild(idxTd);
    tr.appendChild(taskTd);
    tr.appendChild(ownerTd);
    tr.appendChild(deadlineTd);
    tr.appendChild(wfTd);

    actionItemsBodyEl.appendChild(tr);
  });
}

function renderTimeline() {
  while (timelineEl.firstChild) {
    timelineEl.removeChild(timelineEl.firstChild);
  }

  const segments = latestTopicSegments || [];

  if (!segments.length) {
    const div = document.createElement("div");
    div.className = "empty-state";
    div.textContent = "No topic segments available.";
    timelineEl.appendChild(div);
    return;
  }

  segments.forEach((seg, index) => {
    const item = document.createElement("div");
    item.className = "timeline-item";

    const marker = document.createElement("div");
    marker.className = "timeline-marker";

    const dot = document.createElement("div");
    dot.className = "timeline-dot";
    marker.appendChild(dot);

    if (index < segments.length - 1) {
      const line = document.createElement("div");
      line.className = "timeline-line";
      marker.appendChild(line);
    }

    const content = document.createElement("div");
    content.className = "timeline-content";

    const time = document.createElement("div");
    time.className = "timeline-time";
    time.textContent = `[${seg.start_timestamp} - ${seg.end_timestamp}]`;

    const topic = document.createElement("div");
    topic.className = "timeline-topic";
    topic.textContent = seg.topic_summary || "(No summary)";

    content.appendChild(time);
    content.appendChild(topic);

    item.appendChild(marker);
    item.appendChild(content);

    timelineEl.appendChild(item);
  });
}

function renderTranscript() {
  while (transcriptEl.firstChild) {
    transcriptEl.removeChild(transcriptEl.firstChild);
  }

  const raw = latestSpeakerTranscript || "";
  if (!raw.trim()) {
    const div = document.createElement("div");
    div.className = "empty-state";
    div.textContent =
      "No transcript available. Process an audio file to view the conversation.";
    transcriptEl.appendChild(div);
    return;
  }

  const lines = raw.split(/\r?\n/).filter((l) => l.trim().length > 0);

  lines.forEach((line) => {
    const match = line.match(/^\[(.*?)\]\s*(.*?):\s*(.*)$/);
    let timestamp = "";
    let speaker = "";
    let text = line;

    if (match) {
      timestamp = match[1];
      speaker = match[2];
      text = match[3];
    }

    const row = document.createElement("div");
    row.className = "transcript-line";

    const meta = document.createElement("div");
    meta.className = "transcript-meta";
    meta.textContent = timestamp
      ? `${timestamp} • ${speaker}`
      : speaker || "";

    const body = document.createElement("div");
    body.className = "transcript-text";
    body.textContent = text;

    row.appendChild(meta);
    row.appendChild(body);

    transcriptEl.appendChild(row);
  });
}

async function triggerWorkflow(target) {
  if (!latestMeetingReport || !latestSpeakerTranscript) {
    alert("No processed meeting available. Please process audio first.");
    return;
  }

  setLoading(true);
  try {
    const res = await fetch(`${API_BASE}/execute_workflow`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        meeting_report: latestMeetingReport,
        target,
        speaker_attributed_transcript: latestSpeakerTranscript,
      }),
    });

    const data = await res.json().catch(() => ({}));

    if (!res.ok) {
      throw new Error(data.error || "Workflow execution failed.");
    }

    const success = data.workflow_result?.success;
    if (!success) {
      console.warn("Workflow not fully successful:", data.workflow_result);
      alert(
        `Workflow executed with issues for target "${target}". See console for details.`
      );
    } else {
      alert(`Workflow "${target}" executed successfully.`);
    }

    if (data.meeting_report) {
      latestMeetingReport = data.meeting_report;
      renderActionItems();
    }
  } catch (err) {
    console.error(err);
    alert(`Error executing workflow: ${err.message}`);
  } finally {
    setLoading(false);
  }
}

queryForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = queryInput.value.trim();
  if (!question) return;

  if (DEMO_MODE) {
    // Simple demo QA response without backend.
    queryAnswerEl.textContent =
      "Decision: the Q2 budget was approved with a 5% contingency buffer, and the demo was moved to next Friday to accommodate QA and documentation updates.";
    return;
  }

  if (!latestMeetingReport || !latestSpeakerTranscript) {
    queryAnswerEl.textContent =
      "You need to process a meeting before asking questions.";
    return;
  }

  queryAnswerEl.textContent = "Thinking...";
  try {
    const res = await fetch(`${API_BASE}/query_meeting`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        meeting_report: latestMeetingReport,
        speaker_attributed_transcript: latestSpeakerTranscript,
        topic_segments: latestTopicSegments || [],
      }),
    });

    const data = await res.json().catch(() => ({}));

    if (!res.ok) {
      throw new Error(data.error || "Query failed.");
    }

    queryAnswerEl.textContent = data.answer || "No answer returned.";
  } catch (err) {
    console.error(err);
    queryAnswerEl.textContent = `Error: ${err.message}`;
  }
});

