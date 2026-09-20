"""LangGraph subagents for intent classification and confirmation."""

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ...agents import IntentClassifierAgent as RuleBasedIntentClassifierAgent
from ...agents import IntentUnderstandingAgent as RuleBasedIntentUnderstandingAgent
from ...llm import OpenAILLM
from ...models import FollowUpRequest, IntentProfile, SupportRequest
from .prompt import INTENT_UNDERSTANDING_PROMPT


class ClassificationState(TypedDict):
	request: SupportRequest
	result: IntentProfile | None


class ConfirmationState(TypedDict):
	profile: IntentProfile | None
	request: SupportRequest
	result: FollowUpRequest | None


class IntentClassifierAgent:
	def __init__(self) -> None:
		self._rules = RuleBasedIntentClassifierAgent()
		graph = StateGraph(ClassificationState)
		graph.add_node("classify_request", self._classify_request)
		graph.add_edge(START, "classify_request")
		graph.add_edge("classify_request", END)
		self.graph = graph.compile()

	def classify(self, request: SupportRequest) -> IntentProfile | None:
		return self.graph.invoke({"request": request})["result"]

	def get_profile(self, name: str) -> IntentProfile | None:
		return self._rules.get_profile(name)

	def _classify_request(self, state: ClassificationState) -> dict[str, Any]:
		return {"result": self._rules.classify(state["request"])}


class IntentUnderstandingAgent:
	def __init__(self, classifier: IntentClassifierAgent, llm: OpenAILLM | None = None) -> None:
		self.classifier = classifier
		self._rules = RuleBasedIntentUnderstandingAgent(classifier)
		self.llm = llm or OpenAILLM.from_environment()
		understand_graph = StateGraph(ClassificationState)
		understand_graph.add_node("understand_request", self._understand_request)
		understand_graph.add_edge(START, "understand_request")
		understand_graph.add_edge("understand_request", END)
		self.understand_graph = understand_graph.compile()

		confirmation_graph = StateGraph(ConfirmationState)
		confirmation_graph.add_node("confirm_understanding", self._confirm_understanding)
		confirmation_graph.add_edge(START, "confirm_understanding")
		confirmation_graph.add_edge("confirm_understanding", END)
		self.confirmation_graph = confirmation_graph.compile()

	def understand(self, request: SupportRequest) -> IntentProfile | None:
		return self.understand_graph.invoke({"request": request})["result"]

	def request_confirmation(
		self,
		profile: IntentProfile | None,
		request: SupportRequest,
	) -> FollowUpRequest | None:
		return self.confirmation_graph.invoke({"profile": profile, "request": request})["result"]

	def _understand_request(self, state: ClassificationState) -> dict[str, Any]:
		return {"result": self._rules.understand(state["request"])}

	def _confirm_understanding(self, state: ConfirmationState) -> dict[str, Any]:
		follow_up = self._rules.request_confirmation(state["profile"], state["request"])
		if follow_up is None or self.llm is None:
			return {"result": follow_up}

		request = state["request"]
		prompt_input = f"Summary: {request.summary}\nDetails: {request.details}"
		llm_prompt = self.llm.complete(INTENT_UNDERSTANDING_PROMPT, prompt_input)
		return {
			"result": FollowUpRequest(
				missing_fields=follow_up.missing_fields,
				prompt=llm_prompt,
			)
		}

__all__ = ["IntentClassifierAgent", "IntentUnderstandingAgent"]
