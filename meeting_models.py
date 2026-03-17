from typing import List

from pydantic import BaseModel, Field


class ActionItem(BaseModel):
    task: str = Field(..., description="Concrete, actionable task inferred from the meeting")
    owner: str = Field(..., description="Person responsible for the task, inferred from context")
    deadline: str = Field(
        ...,
        description=(
            "Due date or time frame for the task, inferred from context. "
            "If no clear deadline exists, use a best-effort natural language description like "
            "'as soon as possible' or 'no explicit deadline mentioned'."
        ),
    )


class MeetingReport(BaseModel):
    """
    Core schema for structured meeting intelligence required by DeployIt 2026.
    """

    summary: str = Field(..., description="Concise overview of the entire meeting")
    discussion_points: List[str] = Field(
        default_factory=list,
        description="Key topics, questions, and issues discussed during the meeting.",
    )
    decisions: List[str] = Field(
        default_factory=list,
        description="Important commitments, approvals, and decisions that were made.",
    )
    action_items: List[ActionItem] = Field(
        default_factory=list,
        description="Concrete tasks with inferred owners and deadlines.",
    )

