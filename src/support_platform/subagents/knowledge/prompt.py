"""Prompt contract for knowledge retrieval and verification."""

KNOWLEDGE_RETRIEVAL_PROMPT = (
    "Retrieve metadata, database records, or knowledge links relevant to the confirmed request."
)
KNOWLEDGE_VALIDATION_PROMPT = (
    "Check every retrieved source for relevance, title, and summary quality. Retry retrieval "
    "when the evidence does not answer the customer's request."
)
