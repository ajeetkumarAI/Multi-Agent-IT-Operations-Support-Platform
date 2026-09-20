"""Specialized agents used by the support workflow."""

from .models import (
    FollowUpRequest,
    HumanEscalation,
    IntentProfile,
    KnowledgeArtifact,
    SupportRequest,
)


class UserValidationAgent:
    def validate(self, request: SupportRequest) -> FollowUpRequest | None:
        if not request.customer_id.strip():
            return FollowUpRequest(
                missing_fields=["customer_id"],
                prompt="Please provide a valid customer ID before support can continue.",
            )

        if request.metadata.get("identity_verified") is False:
            return FollowUpRequest(
                missing_fields=["identity_verification"],
                prompt="We could not verify this customer ID. Please provide the requested verification details.",
            )

        return None


class IntentClassifierAgent:
    def __init__(self) -> None:
        self._profiles = (
            IntentProfile(
                name="account_unlock",
                keywords=("unlock", "locked", "cannot login", "login failed"),
                required_fields=("account_id", "channel"),
                knowledge_artifacts=(KnowledgeArtifact(
                    title="Digital Banking Unlock Playbook",
                    source="kb://digital-banking/unlock-playbook",
                    summary="Steps for verifying lock reason, initiating unlock, and resetting MFA.",
                ),),
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
                knowledge_artifacts=(KnowledgeArtifact(
                    title="Loan Closure SOP",
                    source="kb://lending/loan-closure-sop",
                    summary="Documents and timeline required to close a retail loan.",
                ),),
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
                knowledge_artifacts=(KnowledgeArtifact(
                    title="Card Dispute Handling Standard",
                    source="kb://payments/card-disputes",
                    summary="Investigation standards, provisional credit checks, and regulatory timelines.",
                ),),
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
        matches: list[tuple[int, float, IntentProfile]] = []
        for profile in self._profiles:
            match_count = sum(1 for keyword in profile.keywords if keyword in content)
            if match_count:
                matches.append((match_count, profile.confidence, profile))
        if not matches:
            return None

        top_match_count, top_confidence, top_profile = max(matches, key=lambda item: (item[0], item[1]))
        top_matches = [
            profile
            for match_count, confidence, profile in matches
            if match_count == top_match_count and confidence == top_confidence
        ]
        if len(top_matches) > 1:
            return None
        return top_profile

    def get_profile(self, name: str) -> IntentProfile | None:
        return next((profile for profile in self._profiles if profile.name == name), None)


class IntentUnderstandingAgent:
    def __init__(self, classifier: IntentClassifierAgent) -> None:
        self.classifier = classifier

    def understand(self, request: SupportRequest) -> IntentProfile | None:
        confirmed_intent = request.metadata.get("confirmed_intent")
        if isinstance(confirmed_intent, str):
            profile = self.classifier.get_profile(confirmed_intent)
            if profile is not None:
                return profile
        return self.classifier.classify(request)

    def request_confirmation(self, profile: IntentProfile | None, request: SupportRequest) -> FollowUpRequest | None:
        if profile is None or request.metadata.get("intent_confirmed") is not False:
            return None
        return FollowUpRequest(
            missing_fields=["intent_confirmation"],
            prompt=(
                f"I understand that you need help with {profile.name.replace('_', ' ')}. "
                "Is this understanding correct? Reply with confirmation or describe what should be changed."
            ),
        )


class InputValidationAgent:
    def validate(self, profile: IntentProfile | None, request: SupportRequest) -> FollowUpRequest | None:
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
            if field_name not in request.metadata or request.metadata.get(field_name) is None
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


class InformationGatheringAgent(InputValidationAgent):
    """Backward-compatible name for the input-gathering stage."""

    gather = InputValidationAgent.validate


class KnowledgeRetrievalAgent:
    def retrieve(self, profile: IntentProfile | None, attempt: int = 1) -> list[KnowledgeArtifact]:
        if profile is None:
            return []
        return list(profile.knowledge_artifacts)


class KnowledgeValidationAgent:
    def validate(self, profile: IntentProfile | None, artifacts: list[KnowledgeArtifact]) -> bool:
        if profile is None or not artifacts:
            return False
        expected_sources = {artifact.source for artifact in profile.knowledge_artifacts}
        return all(
            artifact.source in expected_sources
            and bool(artifact.title.strip())
            and bool(artifact.summary.strip())
            for artifact in artifacts
        )


class ResolutionAgent:
    def resolve(self, profile: IntentProfile, request: SupportRequest) -> list[str] | None:
        if profile.requires_human_handoff:
            return None
        if str(request.metadata.get("severity", "")).lower() in {"high", "critical"}:
            return None
        if request.metadata.get("requires_human"):
            return None
        return list(profile.resolution_actions)


class EscalationAgent:
    def escalate(self, profile: IntentProfile | None, request: SupportRequest, reason: str) -> HumanEscalation:
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
        if str(request.metadata.get("severity", "")).lower() in {"high", "critical"}:
            business_rules.append("High-severity requests must be handled by the specialist desk.")

        return HumanEscalation(
            team=profile.escalation_team,
            reason=reason,
            expertise_required=list(profile.expertise_required),
            business_rules=business_rules,
        )