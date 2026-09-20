"""LangGraph subagent for validating request fields."""

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ...agents import InputValidationAgent as RuleBasedInputValidationAgent
from ...models import FollowUpRequest, IntentProfile, SupportRequest


class ValidationState(TypedDict):
	profile: IntentProfile | None
	request: SupportRequest
	result: FollowUpRequest | None


class InputValidationAgent:
	def __init__(self) -> None:
		self._rules = RuleBasedInputValidationAgent()
		graph = StateGraph(ValidationState)
		graph.add_node("validate_request", self._validate_request)
		graph.add_edge(START, "validate_request")
		graph.add_edge("validate_request", END)
		self.graph = graph.compile()

	def validate(self, profile: IntentProfile | None, request: SupportRequest) -> FollowUpRequest | None:
		return self.graph.invoke({"profile": profile, "request": request})["result"]

	def _validate_request(self, state: ValidationState) -> dict[str, Any]:
		return {"result": self._rules.validate(state["profile"], state["request"])}


class InformationGatheringAgent(InputValidationAgent):
	def gather(self, profile: IntentProfile | None, request: SupportRequest) -> FollowUpRequest | None:
		return self.validate(profile, request)

__all__ = ["InformationGatheringAgent", "InputValidationAgent"]
