from __future__ import annotations

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


class IntentClassifierAgent:
    def __init__(self) -> None:
        self._profiles = (
            IntentProfile(
                name="account_unlock",
                keywords=("unlock", "locked", "cannot login", "login failed"),
                required_fields=("account_id", "channel"),
                knowledge_artifacts=(
                    KnowledgeArtifact(
                        title="Digital Banking Unlock Playbook",
                        source="kb://digital-banking/unlock-playbook",
                        summary="Steps for verifying lock reason, initiating unlock, and resetting MFA.",
                    ),
                ),
                resolution_actions=(
                    "Verify the lock reason from the digital banking audit trail.",
                    "Initiate the self-service account unlock workflow.",
                    "Send MFA reset guidance if the customer is still blocked after unlock.",
                ),
                escalation_team="Digital Banking Support",
                expertise_required=("authentication operations",),
                confidence=0.92,
            ),
            IntentProfile(
                name="loan_closure",
                keywords=("loan closure", "close loan", "foreclosure", "loan payoff"),
                required_fields=("loan_id", "closure_date"),
                knowledge_artifacts=(
                    KnowledgeArtifact(
                        title="Loan Closure SOP",
                        source="kb://lending/loan-closure-sop",
                        summary="Documents and timeline required to close a retail loan.",
                    ),
                ),
                resolution_actions=(
                    "Share the foreclosure statement request process.",
                    "Create a servicing task for payoff computation.",
                ),
                escalation_team="Retail Lending Operations",
                expertise_required=("loan servicing",),
                confidence=0.8,
            ),
            IntentProfile(
                name="card_dispute",
                keywords=("chargeback", "dispute", "unauthorized transaction", "fraudulent transaction"),
                required_fields=("account_id", "transaction_id", "transaction_date"),
                knowledge_artifacts=(
                    KnowledgeArtifact(
                        title="Card Dispute Handling Standard",
                        source="kb://payments/card-disputes",
                        summary="Investigation standards, provisional credit checks, and regulatory timelines.",
                    ),
                ),
                resolution_actions=(
                    "Collect card dispute evidence and verify provisional credit eligibility.",
                ),
                escalation_team="Fraud and Disputes Desk",
                expertise_required=("card disputes", "fraud review"),
                confidence=0.92,
                requires_human_handoff=True,
            ),
        )

    def classify(self, request: SupportRequest) -> IntentProfile | None:
        content = f"{request.summary} {request.details}".lower()
        matches: list[tuple[int, float, float, IntentProfile]] = []
        for profile in self._profiles:
            match_count = sum(1 for keyword in profile.keywords if keyword in content)
            if match_count:
                match_ratio = match_count / len(profile.keywords)
                matches.append((match_count, match_ratio, profile.confidence, profile))
        if not matches:
            return None

        top_match_count, top_match_ratio, top_confidence, top_profile = max(
            matches,
            key=lambda item: (item[0], item[1], item[2]),
        )
        top_matches = [
            profile
            for match_count, match_ratio, confidence, profile in matches
            if (
                match_count == top_match_count
                and match_ratio == top_match_ratio
                and confidence == top_confidence
            )
        ]
        if len(top_matches) > 1:
            return None

        return top_profile


class InformationGatheringAgent:
    def gather(self, profile: IntentProfile | None, request: SupportRequest) -> FollowUpRequest | None:
        if profile is None:
            return FollowUpRequest(
                missing_fields=["intent_details"],
                prompt=(
                    "Please share more detail about the request, including the impacted product, "
                    "customer goal, and any reference number so the right specialist workflow can be selected."
                ),
            )

        missing_fields = [
            field_name
            for field_name in profile.required_fields
            if self._is_missing_value(request.metadata, field_name)
        ]
        if not missing_fields:
            return None

        return FollowUpRequest(
            missing_fields=missing_fields,
            prompt=(
                "Additional information is required before the request can be completed: "
                + ", ".join(missing_fields)
                + "."
            ),
        )

    @staticmethod
    def _is_missing_value(metadata: dict[str, Any], field_name: str) -> bool:
        if field_name not in metadata:
            return True

        value = metadata.get(field_name)
        if value is None:
            return True

        return isinstance(value, str) and not value.strip()


class KnowledgeRetrievalAgent:
    def retrieve(self, profile: IntentProfile | None) -> list[KnowledgeArtifact]:
        if profile is None:
            return []
        return list(profile.knowledge_artifacts)


class ResolutionAgent:
    def resolve(self, profile: IntentProfile, request: SupportRequest) -> list[str] | None:
        if profile.requires_human_handoff:
            return None

        if str(request.metadata.get("severity", "")).lower() == "critical":
            return None

        if request.metadata.get("requires_human"):
            return None

        return list(profile.resolution_actions)


class EscalationAgent:
    def escalate(
        self,
        profile: IntentProfile | None,
        request: SupportRequest,
        reason: str,
    ) -> HumanEscalation:
        if profile is None:
            return HumanEscalation(
                team="General Support Queue",
                reason=reason,
                expertise_required=["triage"],
                business_rules=[
                    "Escalate when intent confidence is too low for automated handling.",
                    "Route ambiguous BFSI requests to a human triage specialist.",
                ],
            )

        business_rules = [
            "Escalate when the workflow requires specialist judgment or regulated review.",
            "Preserve retrieved context so the receiving team can continue without re-triage.",
        ]
        if request.metadata.get("requires_human"):
            business_rules.append("Customer or system explicitly requested human review.")
        if str(request.metadata.get("severity", "")).lower() == "critical":
            business_rules.append("Critical-severity requests must be handled by the specialist desk.")

        return HumanEscalation(
            team=profile.escalation_team,
            reason=reason,
            expertise_required=list(profile.expertise_required),
            business_rules=business_rules,
        )


class MultiAgentSupportPlatform:
    def __init__(self) -> None:
        self.intent_classifier = IntentClassifierAgent()
        self.information_gatherer = InformationGatheringAgent()
        self.knowledge_retriever = KnowledgeRetrievalAgent()
        self.resolution_agent = ResolutionAgent()
        self.escalation_agent = EscalationAgent()

    def process_request(self, request: SupportRequest) -> SupportOutcome:
        profile = self.intent_classifier.classify(request)
        if profile is None:
            if self._requires_human_handoff(request):
                return self._build_unknown_outcome(
                    status="escalated",
                    escalation=self.escalation_agent.escalate(
                        profile=None,
                        request=request,
                        reason="The request could not be classified with enough confidence for automated handling.",
                    ),
                )

            follow_up = self.information_gatherer.gather(profile, request)
            return self._build_unknown_outcome(
                status="needs_information",
                follow_up=follow_up,
            )

        knowledge_artifacts = self.knowledge_retriever.retrieve(profile)
        follow_up = self.information_gatherer.gather(profile, request)
        if follow_up is not None:
            if self._requires_human_handoff(request):
                escalation = self.escalation_agent.escalate(
                    profile=profile,
                    request=request,
                    reason="The request matched a known workflow but requires immediate specialist handling.",
                )
                return SupportOutcome(
                    status="escalated",
                    intent=profile.name,
                    knowledge_artifacts=knowledge_artifacts,
                    recommended_actions=[],
                    escalation=escalation,
                )

            return SupportOutcome(
                status="needs_information",
                intent=profile.name,
                knowledge_artifacts=knowledge_artifacts,
                recommended_actions=[],
                follow_up=follow_up,
            )

        resolution = self.resolution_agent.resolve(profile, request)
        if resolution is not None:
            return SupportOutcome(
                status="resolved",
                intent=profile.name,
                knowledge_artifacts=knowledge_artifacts,
                recommended_actions=resolution,
            )

        escalation = self.escalation_agent.escalate(
            profile=profile,
            request=request,
            reason="The request needs specialist review based on confidence, severity, or regulatory handling rules.",
        )
        return SupportOutcome(
            status="escalated",
            intent=profile.name,
            knowledge_artifacts=knowledge_artifacts,
            recommended_actions=[],
            escalation=escalation,
        )

    @staticmethod
    def _requires_human_handoff(request: SupportRequest) -> bool:
        if request.metadata.get("requires_human"):
            return True

        return str(request.metadata.get("severity", "")).lower() == "critical"

    @staticmethod
    def _build_unknown_outcome(
        status: str,
        follow_up: FollowUpRequest | None = None,
        escalation: HumanEscalation | None = None,
    ) -> SupportOutcome:
        return SupportOutcome(
            status=status,
            intent="unknown",
            knowledge_artifacts=[],
            recommended_actions=[],
            follow_up=follow_up,
            escalation=escalation,
        )
