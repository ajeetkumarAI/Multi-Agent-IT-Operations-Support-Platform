"""LangGraph subagent for customer identity validation."""

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ...agents import UserValidationAgent as RuleBasedUserValidationAgent
from ...models import FollowUpRequest, SupportRequest


class IdentityState(TypedDict):
	request: SupportRequest
	result: FollowUpRequest | None


class UserValidationAgent:
	def __init__(self) -> None:
		self._rules = RuleBasedUserValidationAgent()
		graph = StateGraph(IdentityState)
		graph.add_node("validate_customer", self._validate_customer)
		graph.add_edge(START, "validate_customer")
		graph.add_edge("validate_customer", END)
		self.graph = graph.compile()

	def validate(self, request: SupportRequest) -> FollowUpRequest | None:
		return self.graph.invoke({"request": request})["result"]

	def _validate_customer(self, state: IdentityState) -> dict[str, Any]:
		return {"result": self._rules.validate(state["request"])}

__all__ = ["UserValidationAgent"]
