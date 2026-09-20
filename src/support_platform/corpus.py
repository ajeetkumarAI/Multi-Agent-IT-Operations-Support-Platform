"""Backend-maintained operational support corpus and grounded QA."""

from __future__ import annotations

import json
import re
import sqlite3
from threading import RLock
from dataclasses import dataclass
from pathlib import Path

from .llm import OpenAILLM


@dataclass(frozen=True)
class CorpusRecord:
    record_id: str
    category: str
    title: str
    keywords: tuple[str, ...]
    answer: str
    procedure: tuple[str, ...]
    poc: str
    approval: str
    source: str


DEFAULT_CORPUS = (
    CorpusRecord("SAFEWORD-001", "access", "SafeWord token activation and recovery", ("safeword", "token", "activate", "activation", "otp", "not working", "reset"), "For a SafeWord token that is not working, first verify the user ID and token serial in IAM. Re-sync the token once, then activate or re-enrol it through the SafeWord self-service portal. Do not share the token seed or OTP with support.", ("Confirm the user ID and token serial.", "Check token status in IAM.", "Use token re-sync once.", "If still blocked, raise a CMP request with the error and timestamp."), "IAM Operations POC: iam-operations@example.com", "Manager approval is required for a replacement token; standard re-sync does not require approval.", "kb://iam/safeword-token-runbook"),
    CorpusRecord("CMP-001", "process", "How to raise a CMP request", ("cmp", "ticket", "raise", "request", "incident", "change", "service request"), "Raise a CMP request in the Service Portal using the CMP catalog item. Select the affected application, add the user ID, business impact, exact error, timestamp, screenshots, and required-by date. Submit it to the application support queue.", ("Open Service Portal.", "Choose CMP from the catalog.", "Complete impact and evidence fields.", "Submit to the application support queue.", "Record the CMP number for follow-up."), "Service Desk POC: servicedesk@example.com", "Application-owner approval is required for access changes and production changes.", "kb://service-management/cmp-request-guide"),
    CorpusRecord("POC-001", "ownership", "Application POC and approval routing", ("poc", "owner", "approval", "approver", "who", "contact", "application"), "The application POC is listed in the service catalog under Application Owner. For access approval, the user's line manager approves first and the application owner provides the second approval when the entitlement is privileged or production-scoped.", ("Find the application in the service catalog.", "Identify the Application Owner.", "Obtain line-manager approval.", "Obtain application-owner approval for privileged or production access."), "Service Catalog POC: service-catalog@example.com", "Privileged and production access requires both line-manager and application-owner approval.", "kb://governance/application-ownership-and-approval"),
    CorpusRecord("INSTALL-001", "installation", "Approved application installation", ("install", "installation", "setup", "software", "download", "configure", "access"), "Use the approved software catalog or managed desktop portal to install the application. Do not download installers from public links. If the application is absent from the catalog, raise a CMP service request with the business justification and device details.", ("Search the managed software catalog.", "Install the approved package.", "Restart if requested.", "Raise a CMP request when the package is unavailable."), "Endpoint Support POC: endpoint-support@example.com", "Software installation requiring elevated privileges needs endpoint-admin approval.", "kb://endpoint/approved-application-installation"),
    CorpusRecord("ISSUE-001", "troubleshooting", "General application issue resolution", ("issue", "error", "not working", "problem", "resolve", "fix", "failed"), "Capture the exact error, time, user ID, application name, and business impact. Retry the documented self-service step once. If the issue remains, raise a CMP incident and attach screenshots or logs without including secrets.", ("Capture evidence.", "Check the application runbook.", "Retry the documented fix once.", "Raise CMP with evidence if unresolved."), "Service Desk POC: servicedesk@example.com", "The application support queue decides whether an owner or change approval is needed.", "kb://service-management/application-issue-triage"),
    CorpusRecord("ACCESS-001", "access", "Application access and entitlement request", ("access", "entitlement", "permission", "role", "rights", "provision", "user access"), "Request application access through the approved access catalog. Select the least-privilege role, provide the business justification and required-by date, and attach the manager approval. Privileged or production roles require application-owner approval as well.", ("Search the access catalog for the application.", "Select the least-privilege role.", "Add business justification and required-by date.", "Obtain line-manager approval.", "Obtain application-owner approval for privileged or production access.", "Track the resulting CMP or access request number."), "Access Management POC: access-management@example.com", "Line-manager approval is required for standard access; application-owner and control-owner approval are required for privileged or production access.", "kb://access/application-entitlement-request"),
    CorpusRecord("MFA-001", "identity", "MFA enrollment and replacement", ("mfa", "multi factor", "2fa", "otp", "phone", "lost", "replacement", "enrol"), "Use the identity portal to enroll a registered authenticator. If the device is lost or the authenticator is locked, contact the Identity Operations queue and raise a CMP request for secure reset or replacement. Never send OTP values or recovery codes to support.", ("Verify the user identity using the approved process.", "Check registered authenticators in IAM.", "Use self-service enrollment when available.", "Raise a secure reset or replacement request if the device is lost or locked.", "Confirm the new authenticator works without sharing secrets."), "Identity Operations POC: iam-operations@example.com", "A replacement or reset requires identity verification; manager approval may be required for exceptional access.", "kb://iam/mfa-enrollment-and-replacement"),
    CorpusRecord("VPN-001", "connectivity", "Remote access and VPN troubleshooting", ("vpn", "remote", "network", "connect", "connection", "offsite", "tunnel"), "Confirm the user has approved remote-access entitlement and a supported device. Reconnect using the managed VPN client, capture the error and timestamp, and raise a CMP incident if the tunnel still fails. Do not install an unapproved VPN client.", ("Confirm remote-access entitlement.", "Check device posture and managed-client version.", "Reconnect using the managed VPN client.", "Capture error, timestamp, and network location.", "Raise CMP to Network or Endpoint Support if unresolved."), "Network Operations POC: network-operations@example.com", "Remote-access entitlement requires manager and application-owner approval where the target system is privileged.", "kb://network/remote-access-vpn-troubleshooting"),
    CorpusRecord("CMP-STATUS-001", "process", "CMP request status and reassignment", ("cmp", "status", "update", "pending", "stuck", "queue", "reassign", "follow up"), "Use the CMP number to check status in the Service Portal. Review the assigned queue, pending approval, and requester comments before following up. If the item is routed incorrectly, request reassignment through the Service Desk rather than opening duplicates.", ("Open the Service Portal.", "Search using the CMP number.", "Check current queue and pending approver.", "Add a concise update with new evidence.", "Request reassignment if the queue is incorrect."), "Service Desk POC: servicedesk@example.com", "The current application owner or queue manager approves reassignment; do not bypass required controls.", "kb://service-management/cmp-status-and-reassignment"),
    CorpusRecord("APPROVAL-001", "governance", "Production change and emergency approval", ("production", "change", "emergency", "approval", "approver", "release", "deploy", "control"), "Raise a change request with implementation, validation, rollback, risk, and outage details. Standard production changes follow the normal approval route. Emergency changes require the emergency approver and post-implementation review according to the bank's change policy.", ("Describe implementation and validation steps.", "Document rollback and customer impact.", "Attach test evidence and the implementation window.", "Obtain application-owner and change-control approval.", "Complete post-implementation review for emergency changes."), "Change Management POC: change-management@example.com", "Production changes require application-owner and change-control approval; emergency approval must follow the emergency policy.", "kb://change-management/production-change-approval"),
    CorpusRecord("POC-002", "ownership", "Finding the correct application support team", ("poc", "support team", "owner", "application owner", "queue", "team", "contact", "routing"), "Search the internal service catalog using the application name and environment. Use the listed business owner for functional questions and the technical support queue for incidents. If no record exists, raise a Service Desk request with the application name and business area.", ("Search the service catalog.", "Confirm application and environment.", "Use the business owner for functional questions.", "Use the technical queue for incidents.", "Raise Service Desk when ownership is missing or stale."), "Service Catalog POC: service-catalog@example.com", "The service-catalog owner validates changes to application ownership and support routing.", "kb://governance/application-support-ownership"),
    CorpusRecord("CERT-001", "security", "Certificate expiry and renewal", ("certificate", "cert", "tls", "ssl", "expiry", "renewal", "keystore", "secret"), "Check the approved certificate inventory for expiry date, owner, environment, and renewal method. Open a CMP change before renewal in a controlled environment. Never paste private keys or secrets into a ticket; attach only approved evidence.", ("Check certificate inventory and expiry date.", "Confirm owner and affected environments.", "Raise a renewal change with implementation and rollback details.", "Coordinate validation with the application owner.", "Update the inventory after successful renewal."), "Security Platform POC: security-platform@example.com", "Certificate changes require application-owner approval and change-control approval; production renewal must use the approved secret-management process.", "kb://security/certificate-renewal"),
    CorpusRecord("BATCH-001", "operations", "Batch job failure and rerun request", ("batch", "job", "scheduler", "failed", "rerun", "overnight", "file", "processing"), "Check the scheduler run ID, failure timestamp, dependency status, and input-file availability. Do not rerun a financial batch without confirming duplicate-processing controls. Raise a CMP incident to the batch operations queue with the run ID and business impact.", ("Capture scheduler name and run ID.", "Check dependency and input-file status.", "Confirm duplicate-processing controls.", "Obtain batch-owner approval before rerun.", "Raise CMP to Batch Operations with evidence."), "Batch Operations POC: batch-operations@example.com", "Batch reruns require batch-owner approval; financial or production reruns require change-control approval.", "kb://operations/batch-job-failure-and-rerun"),
    CorpusRecord("ONBOARD-001", "onboarding", "New joiner application access", ("joiner", "onboarding", "new starter", "employee", "access", "day one", "provisioning"), "Use the onboarding access catalog and request only the roles required for the user's job. The line manager owns the request; application owners approve privileged roles. Raise a CMP request for applications missing from the catalog and include the start date.", ("Confirm user ID and start date.", "Select approved baseline roles.", "Add manager and business justification.", "Request application-owner approval for privileged roles.", "Track fulfillment and validate access after provisioning."), "Identity and Access POC: access-management@example.com", "Line-manager approval is required; privileged roles require application-owner approval and may require control-owner review.", "kb://onboarding/new-joiner-access"),
)


def load_sop_faq_file(path: str | Path | None = None) -> tuple[CorpusRecord, ...]:
    corpus_path = Path(path) if path is not None else Path(__file__).resolve().parents[2] / "data" / "operations_sop_faq.json"
    if not corpus_path.exists():
        return ()
    payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    return tuple(
        CorpusRecord(
            record_id=item["record_id"],
            category=item["category"],
            title=item["title"],
            keywords=tuple(item["keywords"]),
            answer=item["answer"],
            procedure=tuple(item["procedure"]),
            poc=item["poc"],
            approval=item["approval"],
            source=item["source"],
        )
        for item in payload.get("records", [])
    )


CORPUS_QA_PROMPT = """You are an enterprise IT operations support assistant.
Answer only from the supplied backend corpus records. Give direct steps, identify the correct POC,
state approval requirements, and include the source record ID. If the records do not answer the question,
say that the relevant backend information was not found and recommend raising a Service Desk/CMP request.
Never invent contacts, approvals, or procedures.
"""


class CorpusRepository:
    def __init__(self, database_path: str = ":memory:") -> None:
        if database_path != ":memory:":
            Path(database_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self.connection = sqlite3.connect(database_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("""CREATE TABLE IF NOT EXISTS support_corpus (
            record_id TEXT PRIMARY KEY, category TEXT NOT NULL, title TEXT NOT NULL,
            keywords TEXT NOT NULL, answer TEXT NOT NULL, procedure TEXT NOT NULL,
            poc TEXT NOT NULL, approval TEXT NOT NULL, source TEXT NOT NULL)""")
        self.connection.commit()

    def seed(
        self,
        records: tuple[CorpusRecord, ...] = DEFAULT_CORPUS,
        sop_faq_path: str | Path | None = None,
    ) -> None:
        all_records = records + load_sop_faq_file(sop_faq_path)
        with self._lock:
            self.connection.executemany("""INSERT OR IGNORE INTO support_corpus
                (record_id, category, title, keywords, answer, procedure, poc, approval, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", [
                (r.record_id, r.category, r.title, "|".join(r.keywords), r.answer,
                 "|".join(r.procedure), r.poc, r.approval, r.source) for r in all_records])
            self.connection.commit()

    def search(self, question: str, limit: int = 3) -> list[CorpusRecord]:
        terms = set(re.findall(r"[a-z0-9]{3,}", question.lower()))
        ranked: list[tuple[int, CorpusRecord]] = []
        with self._lock:
            rows = self.connection.execute("SELECT * FROM support_corpus").fetchall()
        for row in rows:
            title_terms = set(re.findall(r"[a-z0-9]{3,}", row["title"].lower()))
            keyword_terms = set(re.findall(r"[a-z0-9]{3,}", row["keywords"].replace("|", " ").lower()))
            answer_terms = set(re.findall(r"[a-z0-9]{3,}", row["answer"].lower()))
            score = (
                4 * len(terms & title_terms)
                + 3 * len(terms & keyword_terms)
                + len(terms & answer_terms)
            )
            if score:
                ranked.append((score, CorpusRecord(row["record_id"], row["category"], row["title"], tuple(row["keywords"].split("|")), row["answer"], tuple(row["procedure"].split("|")), row["poc"], row["approval"], row["source"])))
        ranked.sort(key=lambda item: (-item[0], item[1].record_id))
        return [record for _, record in ranked[:limit]]

    def close(self) -> None:
        with self._lock:
            self.connection.close()


class CorpusQA:
    def __init__(self, repository: CorpusRepository, llm: OpenAILLM | None = None) -> None:
        self.repository = repository
        self.llm = llm or OpenAILLM.from_environment()

    def answer(self, question: str, limit: int = 3) -> tuple[str, list[CorpusRecord]]:
        records = self.repository.search(question, limit)
        if not records:
            return "I could not find a matching backend corpus record. Raise a Service Desk/CMP request with the user ID, application, error, timestamp, and business impact.", []
        context = "\n\n".join(f"[{r.record_id}] {r.title}\nAnswer: {r.answer}\nSteps: {'; '.join(r.procedure)}\nPOC: {r.poc}\nApproval: {r.approval}\nSource: {r.source}" for r in records)
        if self.llm is not None:
            return self.llm.complete(CORPUS_QA_PROMPT, f"Question: {question}\n\nBackend records:\n{context}"), records
        best = records[0]
        answer = f"{best.answer}\n\nSteps:\n" + "\n".join(f"{i}. {step}" for i, step in enumerate(best.procedure, 1))
        return f"{answer}\n\nPOC: {best.poc}\nApproval: {best.approval}\nSource: {best.record_id} ({best.source})", records
