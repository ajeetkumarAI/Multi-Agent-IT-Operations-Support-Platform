"""LangGraph subagents for database retrieval and relevance validation."""

import os
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ...agents import KnowledgeRetrievalAgent as RuleBasedKnowledgeRetrievalAgent
from ...agents import KnowledgeValidationAgent as RuleBasedKnowledgeValidationAgent
from ...database import KnowledgeRepository
from ...models import IntentProfile, KnowledgeArtifact


class RetrievalState(TypedDict):
	profile: IntentProfile | None
	attempt: int
	result: list[KnowledgeArtifact]


class ValidationState(TypedDict):
	profile: IntentProfile | None
	artifacts: list[KnowledgeArtifact]
	result: bool


class KnowledgeRetrievalAgent:
	def __init__(self, repository: KnowledgeRepository | None = None) -> None:
		self._rules = RuleBasedKnowledgeRetrievalAgent()
		self.repository = repository or KnowledgeRepository(
			os.getenv("SUPPORT_PLATFORM_DATABASE", ":memory:")
		)
		graph = StateGraph(RetrievalState)
		graph.add_node("retrieve_sources", self._retrieve_sources)
		graph.add_edge(START, "retrieve_sources")
		graph.add_edge("retrieve_sources", END)
		self.graph = graph.compile()

	def retrieve(self, profile: IntentProfile | None, attempt: int = 1) -> list[KnowledgeArtifact]:
		return self.graph.invoke({"profile": profile, "attempt": attempt})["result"]

	def _retrieve_sources(self, state: RetrievalState) -> dict[str, Any]:
		profile = state["profile"]
		if profile is None:
			return {"result": []}

		stored = self.repository.search(profile.name)
		if not stored:
			for artifact in self._rules.retrieve(profile, state["attempt"]):
				self.repository.save(profile.name, artifact)
			stored = self.repository.search(profile.name)
		return {"result": stored}


class KnowledgeValidationAgent:
	def __init__(self) -> None:
		self._rules = RuleBasedKnowledgeValidationAgent()
		graph = StateGraph(ValidationState)
		graph.add_node("validate_sources", self._validate_sources)
		graph.add_edge(START, "validate_sources")
		graph.add_edge("validate_sources", END)
		self.graph = graph.compile()

	def validate(self, profile: IntentProfile | None, artifacts: list[KnowledgeArtifact]) -> bool:
		return self.graph.invoke({"profile": profile, "artifacts": artifacts})["result"]

	def _validate_sources(self, state: ValidationState) -> dict[str, Any]:
		return {"result": self._rules.validate(state["profile"], state["artifacts"])}

__all__ = ["KnowledgeRetrievalAgent", "KnowledgeValidationAgent"]
