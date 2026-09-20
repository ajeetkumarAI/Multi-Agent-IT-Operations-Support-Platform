"""Domain models shared by the support agents."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class KnowledgeArtifact:
    title: str
    source: str
    summary: str


@dataclass(frozen=True)
class FollowUpRequest:
    missing_fields: list[str]
    prompt: str


@dataclass(frozen=True)
class HumanEscalation:
    team: str
    reason: str
    expertise_required: list[str]
    business_rules: list[str]


@dataclass(frozen=True)
class SupportRequest:
    customer_id: str
    summary: str
    details: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SupportOutcome:
    status: str
    intent: str
    knowledge_artifacts: list[KnowledgeArtifact]
    recommended_actions: list[str]
    follow_up: FollowUpRequest | None = None
    escalation: HumanEscalation | None = None


@dataclass(frozen=True)
class IntentProfile:
    name: str
    keywords: tuple[str, ...]
    required_fields: tuple[str, ...]
    knowledge_artifacts: tuple[KnowledgeArtifact, ...]
    resolution_actions: tuple[str, ...]
    escalation_team: str
    expertise_required: tuple[str, ...]
    confidence: float
    requires_human_handoff: bool = False