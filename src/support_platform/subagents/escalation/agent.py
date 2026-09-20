"""LangGraph subagent for routing unresolved requests to human teams."""

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ...agents import EscalationAgent as RuleBasedEscalationAgent
from ...models import HumanEscalation, IntentProfile, SupportRequest


class EscalationState(TypedDict):
	profile: IntentProfile | None
	request: SupportRequest
	reason: str
	result: HumanEscalation


class EscalationAgent:
	"""Decides and formats a department handoff through a LangGraph node."""

	def __init__(self) -> None:
		self._rules = RuleBasedEscalationAgent()
		graph = StateGraph(EscalationState)
		graph.add_node("route_to_department", self._route_to_department)
		graph.add_edge(START, "route_to_department")
		graph.add_edge("route_to_department", END)
		self.graph = graph.compile()

	def escalate(
		self,
		profile: IntentProfile | None,
		request: SupportRequest,
		reason: str,
	) -> HumanEscalation:
		result = self.graph.invoke({"profile": profile, "request": request, "reason": reason})
		return result["result"]

	def _route_to_department(self, state: EscalationState) -> dict[str, Any]:
		return {
			"result": self._rules.escalate(
				state["profile"],
				state["request"],
				state["reason"],
			)
		}

__all__ = ["EscalationAgent"]
