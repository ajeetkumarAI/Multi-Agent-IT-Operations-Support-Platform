"""Main orchestration agent for the support platform."""

from .agents import _is_truthy_flag
from .models import FollowUpRequest, HumanEscalation, SupportOutcome, SupportRequest
from .corpus import CorpusQA, CorpusRecord, CorpusRepository
from .document_qa import DocumentStore
from .tools import AgenticRAGTool, FAQTool, MetadataCorpusTool, TextToSQLTool, ToolAnswer
from .subagents import (
    EscalationAgent,
    InputValidationAgent,
    IntentClassifierAgent,
    IntentUnderstandingAgent,
    KnowledgeRetrievalAgent,
    KnowledgeValidationAgent,
    ResolutionAgent,
    UserValidationAgent,
)


class MainOrchestratorAgent:
    """Coordinates specialized subagents and owns the request lifecycle."""

    def __init__(
        self,
        max_knowledge_attempts: int = 3,
        corpus_repository: CorpusRepository | None = None,
        document_store: DocumentStore | None = None,
    ) -> None:
        if max_knowledge_attempts < 1:
            raise ValueError("max_knowledge_attempts must be at least 1")

        self.max_knowledge_attempts = max_knowledge_attempts
        self.user_validator = UserValidationAgent()
        self.intent_classifier = IntentClassifierAgent()
        self.intent_understanding_agent = IntentUnderstandingAgent(self.intent_classifier)
        self.input_validator = InputValidationAgent()
        self.knowledge_retriever = KnowledgeRetrievalAgent()
        self.knowledge_validator = KnowledgeValidationAgent()
        self.resolution_agent = ResolutionAgent()
        self.escalation_agent = EscalationAgent()
        self.corpus_repository = corpus_repository or CorpusRepository()
        self.corpus_repository.seed()
        self.corpus_qa = CorpusQA(self.corpus_repository)
        self.metadata_tool = MetadataCorpusTool(self.corpus_repository)
        self.faq_tool = FAQTool(self.corpus_repository)
        self.sql_tool = TextToSQLTool(self.corpus_repository)
        self.document_store = document_store
        self.rag_tool = AgenticRAGTool(document_store) if document_store is not None else None

    def answer_question(self, question: str) -> ToolAnswer:
        """Route a question to metadata, agentic RAG, or read-only text-to-SQL."""
        normalized = question.lower()
        faq_answer = self.faq_tool.run(question)
        if faq_answer.corpus_records and self._is_simple_faq_question(normalized):
            return faq_answer

        if self.rag_tool is not None and any(
            marker in normalized for marker in ("document", "pdf", "file", "manual", "runbook")
        ):
            rag_answer = self.rag_tool.run(question)
            if rag_answer.document_chunks:
                return rag_answer

        if any(marker in normalized for marker in ("how many", "count", "list all", "show records", "record count")):
            return self.sql_tool.run(question)

        return self.metadata_tool.run(question)

    @staticmethod
    def _is_simple_faq_question(question: str) -> bool:
        complex_markers = (
            "count", "how many", "list all", "compare", "trend", "report", "document",
            "pdf", "file", "runbook", "policy evidence", "database",
        )
        return not any(marker in question for marker in complex_markers)

    def answer_operational_question(
        self,
        question: str,
        limit: int = 3,
    ) -> tuple[str, list[CorpusRecord]]:
        """Retrieve, validate, and answer an operational corpus question."""
        if not question.strip():
            return "Please provide an operational question.", []
        result = self.metadata_tool.run(question)
        return result.answer, result.corpus_records[:limit]

    def process_request(self, request: SupportRequest) -> SupportOutcome:
        identity_follow_up = self.user_validator.validate(request)
        if identity_follow_up is not None:
            return self._build_unknown_outcome("needs_information", follow_up=identity_follow_up)

        profile = self.intent_understanding_agent.understand(request)
        if profile is None:
            if self._requires_human_handoff(request):
                return self._build_unknown_outcome(
                    "escalated",
                    escalation=self.escalation_agent.escalate(
                        None, request,
                        "The request could not be classified with enough confidence for automated handling.",
                    ),
                )
            return self._build_unknown_outcome(
                "needs_information",
                follow_up=self.input_validator.validate(profile, request),
            )

        confirmation = self.intent_understanding_agent.request_confirmation(profile, request)
        if confirmation is not None:
            return SupportOutcome("needs_confirmation", profile.name, [], [], follow_up=confirmation)

        knowledge_artifacts = self.knowledge_retriever.retrieve(profile, attempt=1)
        follow_up = self.input_validator.validate(profile, request)
        if follow_up is not None:
            if self._requires_human_handoff(request):
                return SupportOutcome(
                    "escalated", profile.name, knowledge_artifacts, [],
                    escalation=self.escalation_agent.escalate(
                        profile, request,
                        "The request matched a known workflow but requires immediate specialist handling.",
                    ),
                )
            return SupportOutcome("needs_information", profile.name, knowledge_artifacts, [], follow_up=follow_up)

        if not self.knowledge_validator.validate(profile, knowledge_artifacts):
            knowledge_artifacts = []
        for attempt in range(2, self.max_knowledge_attempts + 1):
            candidate_artifacts = self.knowledge_retriever.retrieve(profile, attempt=attempt)
            if self.knowledge_validator.validate(profile, candidate_artifacts):
                knowledge_artifacts = candidate_artifacts
                break

        if not knowledge_artifacts:
            reason = "Relevant knowledge could not be validated after the configured retrieval attempts."
            if self._requires_human_handoff(request):
                return SupportOutcome(
                    "escalated", profile.name, [], [],
                    escalation=self.escalation_agent.escalate(profile, request, reason),
                )
            return SupportOutcome(
                "needs_information", profile.name, [], [],
                follow_up=FollowUpRequest(
                    ["relevant_support_context"],
                    "I could not validate relevant support information for this request. Please provide more context or a reference number so I can continue.",
                ),
            )

        resolution = self.resolution_agent.resolve(profile, request)
        if resolution is not None:
            return SupportOutcome("resolved", profile.name, knowledge_artifacts, resolution)

        return SupportOutcome(
            "escalated", profile.name, knowledge_artifacts, [],
            escalation=self.escalation_agent.escalate(
                profile, request,
                "The request needs specialist review based on confidence, severity, or regulatory handling rules.",
            ),
        )

    @staticmethod
    def _requires_human_handoff(request: SupportRequest) -> bool:
        return _is_truthy_flag(request.metadata.get("requires_human")) or str(
            request.metadata.get("severity", "")
        ).lower() in {
            "high", "critical",
        }

    @staticmethod
    def _build_unknown_outcome(
        status: str,
        follow_up: FollowUpRequest | None = None,
        escalation: HumanEscalation | None = None,
    ) -> SupportOutcome:
        return SupportOutcome(status, "unknown", [], [], follow_up=follow_up, escalation=escalation)


MultiAgentSupportPlatform = MainOrchestratorAgent