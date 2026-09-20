import unittest

from support_platform import MultiAgentSupportPlatform, SupportRequest


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


if __name__ == "__main__":
    unittest.main()
