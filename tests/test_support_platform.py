import unittest
from threading import Thread
from tempfile import TemporaryDirectory
from pathlib import Path

from src.support_platform.database import KnowledgeRepository
from src.support_platform.document_qa import DocumentChunk, DocumentQA, DocumentStore
from src.support_platform.tools import TextToSQLTool
from src.support_platform.corpus import CorpusQA, CorpusRepository, load_sop_faq_file
from src.support_platform.llm import OpenAILLM
from src.support_platform.subagents.intent.agent import IntentClassifierAgent, IntentUnderstandingAgent
from support_platform import KnowledgeArtifact, KnowledgeRetrievalAgent, MultiAgentSupportPlatform, SupportRequest


class MultiAgentSupportPlatformTests(unittest.TestCase):
    def setUp(self) -> None:
        self.platform = MultiAgentSupportPlatform()

    def test_resolves_account_unlock_request_when_required_context_exists(self) -> None:
        outcome = self.platform.process_request(
            SupportRequest(
                customer_id="CUST-100",
                summary="Unable to unlock mobile banking account",
                details="Customer cannot login after too many failed attempts and needs access restored.",
                metadata={"account_id": "ACCT-1", "channel": "mobile"},
            )
        )

        self.assertEqual(outcome.status, "resolved")
        self.assertEqual(outcome.intent, "account_unlock")
        self.assertTrue(outcome.knowledge_artifacts)
        self.assertIn("Initiate the self-service account unlock workflow.", outcome.recommended_actions)
        self.assertIsNone(outcome.escalation)

    def test_requests_missing_information_for_incomplete_loan_closure_request(self) -> None:
        outcome = self.platform.process_request(
            SupportRequest(
                customer_id="CUST-200",
                summary="Need loan closure",
                details="Please help me close my loan account.",
                metadata={"loan_id": "LN-9"},
            )
        )

        self.assertEqual(outcome.status, "needs_information")
        self.assertEqual(outcome.intent, "loan_closure")
        self.assertEqual(outcome.follow_up.missing_fields, ["closure_date"])
        self.assertIn("closure_date", outcome.follow_up.prompt)

    def test_escalates_card_dispute_to_specialist_team(self) -> None:
        outcome = self.platform.process_request(
            SupportRequest(
                customer_id="CUST-300",
                summary="Unauthorized transaction dispute",
                details="A card transaction was not performed by the customer and needs investigation.",
                metadata={
                    "account_id": "ACCT-7",
                    "transaction_id": "TXN-55",
                    "transaction_date": "2026-09-19",
                },
            )
        )

        self.assertEqual(outcome.status, "escalated")
        self.assertEqual(outcome.intent, "card_dispute")
        self.assertEqual(outcome.escalation.team, "Fraud and Disputes Desk")
        self.assertIn("card disputes", outcome.escalation.expertise_required)
        self.assertTrue(outcome.knowledge_artifacts)

    def test_requests_more_detail_for_unknown_intent(self) -> None:
        outcome = self.platform.process_request(
            SupportRequest(
                customer_id="CUST-400",
                summary="Need help",
                details="Customer asks for support but the product and issue are unclear.",
                metadata={},
            )
        )

        self.assertEqual(outcome.status, "needs_information")
        self.assertEqual(outcome.intent, "unknown")
        self.assertEqual(outcome.follow_up.missing_fields, ["intent_details"])
        self.assertIn("impacted product", outcome.follow_up.prompt)

    def test_escalates_high_severity_unknown_intent_to_human_triage(self) -> None:
        outcome = self.platform.process_request(
            SupportRequest(
                customer_id="CUST-500",
                summary="System issue",
                details="Urgent help needed but the case details are still incomplete.",
                metadata={"severity": "critical"},
            )
        )

        self.assertEqual(outcome.status, "escalated")
        self.assertEqual(outcome.intent, "unknown")
        self.assertEqual(outcome.escalation.team, "General Support Queue")
        self.assertIn("triage", outcome.escalation.expertise_required)

    def test_escalates_urgent_known_intent_even_when_information_is_missing(self) -> None:
        outcome = self.platform.process_request(
            SupportRequest(
                customer_id="CUST-600",
                summary="Account locked",
                details="Customer cannot login and needs immediate support.",
                metadata={"account_id": "ACCT-99", "severity": "critical"},
            )
        )

        self.assertEqual(outcome.status, "escalated")
        self.assertEqual(outcome.intent, "account_unlock")
        self.assertEqual(outcome.escalation.team, "Digital Banking Support")
        self.assertIsNone(outcome.follow_up)

    def test_treats_zero_value_metadata_as_present(self) -> None:
        outcome = self.platform.process_request(
            SupportRequest(
                customer_id="CUST-700",
                summary="Cannot login because account is locked",
                details="The customer needs help unlocking internet banking access.",
                metadata={"account_id": 0, "channel": "internet"},
            )
        )

        self.assertEqual(outcome.status, "resolved")
        self.assertEqual(outcome.intent, "account_unlock")

    def test_requests_clarification_when_multiple_intents_match_equally(self) -> None:
        outcome = self.platform.process_request(
            SupportRequest(
                customer_id="CUST-800",
                summary="Cannot login after unauthorized transaction dispute",
                details="The account is locked, login failed, and there is a fraudulent transaction to dispute.",
                metadata={},
            )
        )

        self.assertEqual(outcome.status, "needs_information")
        self.assertEqual(outcome.intent, "unknown")
        self.assertIn("impacted product", outcome.follow_up.prompt)

    def test_requires_customer_id_before_agent_orchestration_starts(self) -> None:
        outcome = self.platform.process_request(
            SupportRequest(
                customer_id="",
                summary="Cannot login because account is locked",
                details="The customer needs help unlocking internet banking access.",
                metadata={"account_id": "ACCT-1", "channel": "mobile"},
            )
        )

        self.assertEqual(outcome.status, "needs_information")
        self.assertEqual(outcome.follow_up.missing_fields, ["customer_id"])

    def test_pauses_for_user_confirmation_when_intent_is_not_confirmed(self) -> None:
        outcome = self.platform.process_request(
            SupportRequest(
                customer_id="CUST-901",
                summary="Unable to unlock mobile banking account",
                details="Customer cannot login after too many failed attempts.",
                metadata={
                    "account_id": "ACCT-1",
                    "channel": "mobile",
                    "intent_confirmed": False,
                },
            )
        )

        self.assertEqual(outcome.status, "needs_confirmation")
        self.assertEqual(outcome.intent, "account_unlock")
        self.assertIn("Is this understanding correct?", outcome.follow_up.prompt)

    def test_confirmed_intent_feedback_selects_the_requested_workflow(self) -> None:
        outcome = self.platform.process_request(
            SupportRequest(
                customer_id="CUST-902",
                summary="The request was unclear initially",
                details="Customer confirms this is an account unlock request.",
                metadata={
                    "account_id": "ACCT-2",
                    "channel": "mobile",
                    "confirmed_intent": "account_unlock",
                    "intent_confirmed": True,
                },
            )
        )

        self.assertEqual(outcome.status, "resolved")
        self.assertEqual(outcome.intent, "account_unlock")

    def test_retries_until_retrieved_knowledge_is_relevant(self) -> None:
        class EventuallyRelevantRetriever(KnowledgeRetrievalAgent):
            def __init__(self) -> None:
                self.attempts: list[int] = []

            def retrieve(self, profile, attempt=1):
                self.attempts.append(attempt)
                if attempt < 3:
                    return [KnowledgeArtifact("Wrong", "kb://wrong", "Wrong workflow")]
                return super().retrieve(profile, attempt)

        retriever = EventuallyRelevantRetriever()
        self.platform.knowledge_retriever = retriever

        outcome = self.platform.process_request(
            SupportRequest(
                customer_id="CUST-903",
                summary="Unable to unlock mobile banking account",
                details="Customer cannot login after too many failed attempts.",
                metadata={"account_id": "ACCT-3", "channel": "mobile"},
            )
        )

        self.assertEqual(outcome.status, "resolved")
        self.assertEqual(retriever.attempts, [1, 2, 3])

    def test_requests_more_context_when_retrieval_cannot_be_validated(self) -> None:
        class EmptyRetriever(KnowledgeRetrievalAgent):
            def retrieve(self, profile, attempt=1):
                return []

        self.platform.knowledge_retriever = EmptyRetriever()
        outcome = self.platform.process_request(
            SupportRequest(
                customer_id="CUST-904",
                summary="Unable to unlock mobile banking account",
                details="Customer cannot login after too many failed attempts.",
                metadata={"account_id": "ACCT-4", "channel": "mobile"},
            )
        )

        self.assertEqual(outcome.status, "needs_information")
        self.assertEqual(outcome.follow_up.missing_fields, ["relevant_support_context"])

    def test_knowledge_repository_persists_artifacts_in_sqlite(self) -> None:
        with TemporaryDirectory() as directory:
            database_path = str(Path(directory) / "support.db")
            repository = KnowledgeRepository(database_path)
            artifact = KnowledgeArtifact("Unlock guide", "kb://unlock", "Unlock steps")
            repository.save("account_unlock", artifact)

            reopened = KnowledgeRepository(database_path)
            self.assertEqual(reopened.search("account_unlock"), [artifact])
            repository.close()
            reopened.close()

    def test_intent_confirmation_uses_injected_openai_compatible_model(self) -> None:
        class FakeChatModel:
            def invoke(self, messages):
                self.messages = messages
                return type("Response", (), {"content": "Please confirm the account unlock request."})()

        llm = OpenAILLM(FakeChatModel())
        classifier = IntentClassifierAgent()
        intent_agent = IntentUnderstandingAgent(classifier, llm=llm)
        request = SupportRequest(
            customer_id="CUST-905",
            summary="Account locked",
            details="Please unlock my account.",
            metadata={"intent_confirmed": False},
        )

        profile = intent_agent.understand(request)
        follow_up = intent_agent.request_confirmation(profile, request)

        self.assertEqual(follow_up.prompt, "Please confirm the account unlock request.")

    def test_document_qa_indexes_and_answers_from_uploaded_text(self) -> None:
        qa = DocumentQA(DocumentStore())
        chunk_count = qa.ingest(
            "policy.txt",
            b"The cancellation policy allows customers to cancel within thirty days. Refunds are processed within five business days.",
        )

        answer, sources = qa.answer("How many days do customers have to cancel?")

        self.assertEqual(chunk_count, 1)
        self.assertIn("thirty days", answer)
        self.assertEqual(sources[0].document_name, "policy.txt")

    def test_backend_corpus_answers_safeword_question_with_poc_and_approval(self) -> None:
        repository = CorpusRepository()
        repository.seed()
        answer, records = CorpusQA(repository).answer("My SafeWord token is not working. How do I activate it?")

        self.assertEqual(records[0].record_id, "SAFEWORD-001")
        self.assertIn("IAM Operations POC", answer)
        self.assertIn("replacement token", answer)
        repository.close()

    def test_backend_corpus_answers_how_to_raise_cmp(self) -> None:
        repository = CorpusRepository()
        repository.seed()
        answer, records = CorpusQA(repository).answer("How do I raise a CMP request for an application issue?")

        self.assertEqual(records[0].record_id, "CMP-001")
        self.assertIn("Service Portal", answer)
        self.assertIn("Service Desk POC", answer)
        repository.close()

    def test_backend_corpus_routes_access_question_to_approval_path(self) -> None:
        repository = CorpusRepository()
        repository.seed()
        answer, records = CorpusQA(repository).answer(
            "How do I request application entitlement for a production role and who approves it?"
        )

        self.assertEqual(records[0].record_id, "ACCESS-001")
        self.assertIn("least-privilege", answer)
        self.assertIn("application-owner approval", answer)
        repository.close()

    def test_backend_corpus_answers_batch_failure_with_rerun_control(self) -> None:
        repository = CorpusRepository()
        repository.seed()
        answer, records = CorpusQA(repository).answer(
            "The overnight batch job failed. Can I rerun it and who must approve?"
        )

        self.assertEqual(records[0].record_id, "BATCH-001")
        self.assertIn("duplicate-processing", answer)
        self.assertIn("Batch Operations POC", answer)
        repository.close()

    def test_backend_corpus_answers_certificate_renewal_question(self) -> None:
        repository = CorpusRepository()
        repository.seed()
        answer, records = CorpusQA(repository).answer(
            "A TLS certificate is expiring. How do I renew it safely?"
        )

        self.assertEqual(records[0].record_id, "CERT-001")
        self.assertIn("private keys", answer)
        self.assertIn("change-control approval", answer)
        repository.close()

    def test_maintained_sop_faq_file_is_loaded_into_backend_corpus(self) -> None:
        records = load_sop_faq_file()

        self.assertTrue(records)
        self.assertIn("FAQ-CMP-002", {record.record_id for record in records})

    def test_specific_entitlement_question_beats_generic_ownership_record(self) -> None:
        repository = CorpusRepository()
        repository.seed()

        records = repository.search(
            "How do I request application entitlement for a production role and who approves it?"
        )

        self.assertEqual(records[0].record_id, "ACCESS-001")
        repository.close()

    def test_main_orchestrator_answers_operational_question_from_corpus(self) -> None:
        repository = CorpusRepository()
        orchestrator = MultiAgentSupportPlatform(corpus_repository=repository)

        answer, records = orchestrator.answer_operational_question(
            "How do I raise a CMP request for an application issue?"
        )

        self.assertEqual(records[0].record_id, "CMP-001")
        self.assertIn("Service Portal", answer)
        repository.close()

    def test_main_orchestrator_routes_document_question_to_agentic_rag(self) -> None:
        document_store = DocumentStore()
        document_store.replace_document(
            "access-runbook.pdf",
            [
                DocumentChunk("access-runbook.pdf", 0, "Production access requires manager and application owner approval.")
            ],
        )
        orchestrator = MultiAgentSupportPlatform(document_store=document_store)

        result = orchestrator.answer_question("What does the access runbook document say about production approval?")

        self.assertEqual(result.tool, "agentic_rag")
        self.assertIn("Production access", result.answer)
        document_store.close()

    def test_text_to_sql_tool_is_read_only_and_returns_metadata_records(self) -> None:
        repository = CorpusRepository()
        repository.seed()
        tool = TextToSQLTool(repository)

        result = tool.run("List all certificate records")

        self.assertEqual(result.tool, "text_to_sql")
        self.assertTrue(any(record.record_id == "CERT-001" for record in result.corpus_records))
        with self.assertRaises(ValueError):
            tool._execute_read_only("DELETE FROM support_corpus", ())
        repository.close()

    def test_main_orchestrator_routes_simple_question_to_faq_first(self) -> None:
        repository = CorpusRepository()
        orchestrator = MultiAgentSupportPlatform(corpus_repository=repository)

        result = orchestrator.answer_question("How do I activate my SafeWord token?")

        self.assertEqual(result.tool, "faq")
        self.assertTrue(result.corpus_records[0].source.startswith("faq://"))
        self.assertEqual(len(result.corpus_records), 1)
        repository.close()

    def test_main_orchestrator_uses_sql_for_structured_question_before_faq(self) -> None:
        repository = CorpusRepository()
        orchestrator = MultiAgentSupportPlatform(corpus_repository=repository)

        result = orchestrator.answer_question("List all certificate records")

        self.assertEqual(result.tool, "text_to_sql")
        repository.close()

    def test_corpus_repository_can_be_reused_across_streamlit_threads(self) -> None:
        repository = CorpusRepository()
        repository.seed()
        results: list[str] = []

        def read_from_rerun_thread() -> None:
            results.append(repository.search("SafeWord token activation")[0].record_id)

        thread = Thread(target=read_from_rerun_thread)
        thread.start()
        thread.join()
        repository.close()

        self.assertEqual(results, ["SAFEWORD-001"])


if __name__ == "__main__":
    unittest.main()
