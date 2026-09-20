"""LangGraph subagent for deciding automated resolution."""

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ...agents import ResolutionAgent as RuleBasedResolutionAgent
from ...models import IntentProfile, SupportRequest


class ResolutionState(TypedDict):
	profile: IntentProfile
	request: SupportRequest
	result: list[str] | None


class ResolutionAgent:
	def __init__(self) -> None:
		self._rules = RuleBasedResolutionAgent()
		graph = StateGraph(ResolutionState)
		graph.add_node("decide_resolution", self._decide_resolution)
		graph.add_edge(START, "decide_resolution")
		graph.add_edge("decide_resolution", END)
		self.graph = graph.compile()

	def resolve(self, profile: IntentProfile, request: SupportRequest) -> list[str] | None:
		return self.graph.invoke({"profile": profile, "request": request})["result"]

	def _decide_resolution(self, state: ResolutionState) -> dict[str, Any]:
		return {"result": self._rules.resolve(state["profile"], state["request"])}

__all__ = ["ResolutionAgent"]
